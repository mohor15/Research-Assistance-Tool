"""On-demand Tier B newsletter report resolution.

This module is the ONE place a Qwen call or a source re-extraction
attempt may happen for a Tier B article, and ONLY when explicitly
invoked from the newsletter-generation workflow (never from article
selection/deselection, never from rendering).

resolve_tier_b_reports is deliberately separate from both
ui.dashboard_data.render_newsletter_markdown and
delivery.renderer.render_event_list_email -- those two stay pure
(no model call, no network call, no persistence write); this module is
the only thing that calls into agents.report.ReportAgent or performs a
bounded, network-free re-extraction, and the only thing that writes to
the tier_b_reports table.

Caching: a cached row is reused ONLY when the stored content_hash
matches the current Article.content_hash (confirmed to be sha256 of
clean_text; see the content_hash field in models.article.Article, and
note Evidence.content/.content_hash are copied verbatim from
Article.clean_text/.content_hash by evidence.conversion.
article_to_evidence, so this one hash correctly identifies the exact
text a Tier B report would be grounded in) AND the stored schema_version
matches TIER_B_REPORT_SCHEMA_VERSION. A status equal to failure cached
row is NEVER treated as a valid hit -- it is always eligible for a
fresh, user-initiated regeneration attempt, but never retried
automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from research_assistant.agents.report import ReportAgent, TierBGenerationInput
from research_assistant.config.schema import AppConfig
from research_assistant.evidence.hashing import hash_content
from research_assistant.evidence.text import normalize_whitespace
from research_assistant.llm.client import create_ollama_client
from research_assistant.llm.errors import LLMError
from research_assistant.models.article import Article
from research_assistant.models.report import TierBReport, TierBReportFailure
from research_assistant.retrieval.classification import classify
from research_assistant.retrieval.extractor import extract_article
from research_assistant.retrieval.models import CachePolicy, ExtractionStatus
from research_assistant.storage.persistence_service import PersistenceService
from research_assistant.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

TIER_B_REPORT_SCHEMA_VERSION = "tier_b_v1"


_STATUS_CACHED = "cached"
_STATUS_GENERATED = "generated"
_STATUS_INSUFFICIENT_CONTENT = "insufficient_content"
_STATUS_FAILED = "failed"
_STATUS_SKIPPED_CAP = "skipped_cap"
_STATUS_NOT_FOUND = "not_found"

# Manual fetch action's own result statuses -- deliberately distinct from
# `TierBReportResolution`'s (that dataclass is about REPORT generation;
# this one is about whether the underlying ARTICLE TEXT improved).
_FETCH_STATUS_IMPROVED = "improved"
_FETCH_STATUS_NO_IMPROVEMENT = "no_improvement"
_FETCH_STATUS_FAILED = "failed"
_FETCH_STATUS_NOT_FOUND = "not_found"
_FETCH_STATUS_TOO_LARGE = "response_too_large"


@dataclass(frozen=True)
class ManualFetchResult:
    status: str
    detail: str = ""
    new_text_chars: Optional[int] = None


@dataclass(frozen=True)
class TierBReportResolution:
    status: str
    report: Optional[TierBReport] = None
    failure: Optional[TierBReportFailure] = None


def _effective_text(article: Article, evidence_list: list) -> Optional[str]:
    if evidence_list and evidence_list[0].content:
        return evidence_list[0].content
    if article.clean_text:
        return article.clean_text
    if article.raw_text:
        return article.raw_text
    return None


def _is_sufficient(article: Article, text: str, *, min_chars: int) -> bool:
    return article.extraction_status == ExtractionStatus.SUCCESS and len(text) >= min_chars


def is_content_sufficient_for_report(service: PersistenceService, config: AppConfig, article_id: str) -> Optional[bool]:
    """Read-only, deterministic, no network/model call -- lets the UI
    decide whether to offer the explicit "fetch full article" action for
    a given Tier B selection (shown only when the already-persisted text
    is insufficient). Returns `None` if the article can't be found."""
    conn = service.database.connect()
    try:
        article = service.articles.get(conn, article_id)
        if article is None:
            return None
        evidence_list = service.evidence.get_by_article_id(conn, article_id)
    finally:
        conn.close()
    text = _effective_text(article, evidence_list)
    min_chars = config.settings.current_llm.tier_b_report_min_sufficient_chars
    return text is not None and _is_sufficient(article, text, min_chars=min_chars)


def _attempt_cache_free_reextraction(article: Article, *, cache_pages_enabled: bool) -> Optional[str]:
    from research_assistant.retrieval.cache import PageCache

    cache = PageCache(cache_dir=DATA_DIR / "cache", enabled=cache_pages_enabled)
    page = cache.get(article.url)
    if page is None or not page.success or not page.html:
        return None
    try:
        extracted = extract_article(page.html, page.url, content_type=page.content_type)
    except Exception as exc:
        logger.warning("Tier B bounded re-extraction failed for %s: %s", article.id, exc)
        return None
    if extracted.status in (ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL) and extracted.text:
        return extracted.text
    return None


def _build_generation_input(article: Article, text: str, *, content_partial: bool) -> TierBGenerationInput:
    return TierBGenerationInput(
        article_id=article.id, title=article.title, publisher=article.publisher or article.source_name,
        publication_date=article.publication_date, url=article.url, source_type=article.source_type,
        source_priority=article.source_priority, text=text, content_partial=content_partial,
    )


def resolve_tier_b_reports(
    service: PersistenceService, config: AppConfig, article_ids: List[str],
    *, report_agent: Optional[ReportAgent] = None,
) -> Dict[str, TierBReportResolution]:
    current_llm = config.settings.current_llm
    min_chars = current_llm.tier_b_report_min_sufficient_chars
    cache_pages_enabled = config.settings.retrieval.cache_pages
    max_new = current_llm.max_tier_b_reports_per_newsletter_request

    unique_ids = list(dict.fromkeys(article_ids))
    resolutions: Dict[str, TierBReportResolution] = {}
    pending_generation: List[TierBGenerationInput] = []

    conn = service.database.connect()
    try:
        for article_id in unique_ids:
            article = service.articles.get(conn, article_id)
            if article is None:
                resolutions[article_id] = TierBReportResolution(status=_STATUS_NOT_FOUND)
                continue

            cached = service.tier_b_reports.get(conn, article_id)
            if (
                cached is not None
                and cached.content_hash is not None
                and cached.content_hash == article.content_hash
                and cached.schema_version == TIER_B_REPORT_SCHEMA_VERSION
                and isinstance(cached.result, TierBReport)
            ):
                resolutions[article_id] = TierBReportResolution(status=_STATUS_CACHED, report=cached.result)
                continue

            evidence_list = service.evidence.get_by_article_id(conn, article_id)
            text = _effective_text(article, evidence_list)
            content_partial = article.extraction_status != ExtractionStatus.SUCCESS

            if text is None or not _is_sufficient(article, text, min_chars=min_chars):
                re_extracted = _attempt_cache_free_reextraction(article, cache_pages_enabled=cache_pages_enabled)
                if re_extracted is not None and len(re_extracted) >= min_chars:
                    text = re_extracted
                    content_partial = False
                else:
                    resolutions[article_id] = TierBReportResolution(status=_STATUS_INSUFFICIENT_CONTENT)
                    continue

            if len(pending_generation) >= max_new:
                resolutions[article_id] = TierBReportResolution(status=_STATUS_SKIPPED_CAP)
                continue

            pending_generation.append(_build_generation_input(article, text, content_partial=content_partial))
    finally:
        conn.close()

    if pending_generation:
        agent = report_agent or ReportAgent(create_ollama_client(config))
        generated_at = datetime.now(timezone.utc)
        results = agent.generate_tier_b_many(pending_generation)
        conn = service.database.connect()
        try:
            for item, result in zip(pending_generation, results):
                article = service.articles.get(conn, item.article_id)
                content_hash = article.content_hash if article else None
                if isinstance(result, TierBReportFailure):
                    service.tier_b_reports.upsert_failure(
                        conn, result, content_hash=content_hash,
                        schema_version=TIER_B_REPORT_SCHEMA_VERSION, generated_at=generated_at,
                    )
                    resolutions[item.article_id] = TierBReportResolution(status=_STATUS_FAILED, failure=result)
                else:
                    service.tier_b_reports.upsert_report(
                        conn, result, content_hash=content_hash,
                        schema_version=TIER_B_REPORT_SCHEMA_VERSION, generated_at=generated_at,
                    )
                    resolutions[item.article_id] = TierBReportResolution(status=_STATUS_GENERATED, report=result)
            conn.commit()
        except LLMError:
            conn.rollback()
            raise
        finally:
            conn.close()

    return resolutions


def fetch_full_article_for_tier_b(service: PersistenceService, config: AppConfig, article_id: str) -> ManualFetchResult:
    """Explicit, user-triggered live re-fetch of exactly ONE Tier B
    article's own URL -- the ONLY place this module ever makes a live
    network request (`resolve_tier_b_reports` only ever does a bounded,
    network-free re-extraction from the on-disk page cache). Never called
    automatically by `resolve_tier_b_reports`, article selection, or
    rendering -- the caller (a dedicated UI button) is responsible for
    invoking this only when the user explicitly asks for it.

    Bounded: one HTTP request (`max_retries=0`, no retry loop), one
    article, a dedicated configurable timeout
    (`tier_b_manual_fetch_timeout_seconds`) and response-size limit
    (`tier_b_manual_fetch_max_html_chars`). Respects the same robots.txt
    setting the rest of retrieval does. Never overwrites existing
    Article/Evidence text with something worse: the new extraction is
    only persisted if it is a successful (not merely partial) extraction
    AND strictly longer than whatever effective text already existed.
    On any failure, the existing Article/Evidence rows are left
    untouched and a clear, classified failure reason is returned."""
    from research_assistant.retrieval.cache import PageCache
    from research_assistant.retrieval.client import DEFAULT_USER_AGENT, HTTPClient
    from research_assistant.retrieval.robots import RobotsChecker

    conn = service.database.connect()
    try:
        article = service.articles.get(conn, article_id)
        evidence_list = service.evidence.get_by_article_id(conn, article_id)
    finally:
        conn.close()

    if article is None:
        return ManualFetchResult(status=_FETCH_STATUS_NOT_FOUND, detail="Article not found.")

    existing_text = _effective_text(article, evidence_list) or ""
    retrieval_settings = config.settings.retrieval
    current_llm = config.settings.current_llm

    http_client = HTTPClient(
        timeout_seconds=current_llm.tier_b_manual_fetch_timeout_seconds,
        max_retries=0,
        retry_delay_seconds=0.0,
        user_agent=DEFAULT_USER_AGENT,
        robots_checker=RobotsChecker(enabled=retrieval_settings.respect_robots_txt, user_agent=DEFAULT_USER_AGENT),
        cache=PageCache(cache_dir=DATA_DIR / "cache", enabled=retrieval_settings.cache_pages),
    )

    try:
        page = http_client.get(article.url, cache_policy=CachePolicy.FRESH)
    except Exception as exc:  # pragma: no cover - HTTPClient.get never raises today; defensive only
        logger.warning("Manual Tier B fetch raised unexpectedly for %s: %s", article_id, exc)
        return ManualFetchResult(status=_FETCH_STATUS_FAILED, detail=f"unexpected_error: {exc}")

    if not page.success or not page.html:
        reason = classify(page.failure) if page.failure else "no_content"
        return ManualFetchResult(status=_FETCH_STATUS_FAILED, detail=reason)

    if len(page.html) >= current_llm.tier_b_manual_fetch_max_html_chars:
        logger.info("Manual Tier B fetch for %s exceeded the response-size bound; rejected", article_id)
        return ManualFetchResult(status=_FETCH_STATUS_TOO_LARGE, detail="response_too_large")

    try:
        extracted = extract_article(page.html, page.url, content_type=page.content_type)
    except Exception as exc:
        logger.warning("Manual Tier B fetch extraction failed for %s: %s", article_id, exc)
        return ManualFetchResult(status=_FETCH_STATUS_FAILED, detail=f"extraction_error: {exc}")

    new_clean_text = normalize_whitespace(extracted.text) if extracted.text else ""
    if extracted.status != ExtractionStatus.SUCCESS or len(new_clean_text) <= len(existing_text):
        # Never overwrite better (or equal) existing content with a
        # worse/empty/partial extraction -- the existing Article/Evidence
        # rows are left exactly as they were.
        return ManualFetchResult(
            status=_FETCH_STATUS_NO_IMPROVEMENT,
            detail=f"new extraction ({len(new_clean_text)} chars, status={extracted.status.value}) "
                   f"did not improve on existing text ({len(existing_text)} chars)",
        )

    new_hash = hash_content(new_clean_text)
    conn = service.database.connect()
    try:
        updated_article = article.model_copy(
            update={
                "raw_text": extracted.text,
                "clean_text": new_clean_text,
                "content_hash": new_hash,
                "extraction_status": ExtractionStatus.SUCCESS,
                "extraction_method": extracted.extraction_method,
            }
        )
        service.articles.upsert(conn, updated_article)
        for evidence in evidence_list:
            service.evidence.upsert(
                conn, evidence.model_copy(update={"content": new_clean_text, "content_hash": new_hash}),
            )
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "Manual Tier B fetch improved article %s: %d -> %d chars",
        article_id, len(existing_text), len(new_clean_text),
    )
    return ManualFetchResult(status=_FETCH_STATUS_IMPROVED, new_text_chars=len(new_clean_text))
