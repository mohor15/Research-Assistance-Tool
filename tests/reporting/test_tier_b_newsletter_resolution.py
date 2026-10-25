"""resolve_tier_b_reports -- the ONE place a Qwen call or a bounded,
network-free re-extraction may happen for a Tier B article, and ONLY
when explicitly called (never from selection/deselection, never from
rendering). All model calls are via a scripted FakeLLMClient/FakeAgent;
nothing here ever touches real Ollama or the real network."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest

from research_assistant.agents.report import ReportAgent, TierBGenerationInput
from research_assistant.config import get_config
from research_assistant.llm.errors import LLMResponseError, LLMUnavailableError
from research_assistant.models.report import TierBReport, TierBReportFailure
from research_assistant.reporting.tier_b_newsletter_resolution import (
    TIER_B_REPORT_SCHEMA_VERSION,
    resolve_tier_b_reports,
)
from research_assistant.storage.persistence_service import PersistenceService

from tests.reporting.conftest import FakeLLMClient
from tests.storage.conftest import make_article

D = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)

_SUFFICIENT_TEXT = "A real, substantive article about an AI development. " * 20  # >200 chars
_INSUFFICIENT_TEXT = "Too short."


def _config(**current_llm_overrides):
    config = get_config().model_copy(deep=True)
    for key, value in current_llm_overrides.items():
        setattr(config.settings.current_llm, key, value)
    config.settings.retrieval.cache_pages = False  # never touch the real on-disk cache in tests
    return config


def _persist(service: PersistenceService, article_id: str, *, clean_text: str, content_hash: str = "hash1") -> None:
    conn = service.database.connect()
    service.articles.upsert(conn, make_article(article_id, clean_text=clean_text, content_hash=content_hash))
    conn.commit()
    conn.close()


def _narrative(**overrides):
    from research_assistant.agents.schemas import TierBReportNarrative

    kwargs = dict(
        opening_context="Something happened.",
        key_insights=["Insight one.", "Insight two."],
        strategic_implications="This may be relevant to markets.",
        nomura_opportunity="This may be relevant to Nomura's consideration of this area.",
    )
    kwargs.update(overrides)
    return TierBReportNarrative(**kwargs)


class _CountingReportAgent(ReportAgent):
    """Wraps the real ReportAgent's Tier B path over a FakeLLMClient,
    while recording every generate_tier_b_many call's input order."""

    def __init__(self, responses: List):
        super().__init__(FakeLLMClient(responses))
        self.batches: List[List[TierBGenerationInput]] = []

    def generate_tier_b_many(self, items):
        self.batches.append(list(items))
        return super().generate_tier_b_many(items)


# ---------------------------------------------------------------------------
# Unselected / nonexistent articles never trigger a model call
# ---------------------------------------------------------------------------


def test_nonexistent_article_resolves_to_not_found_with_zero_model_calls(persistence_service: PersistenceService):
    agent = _CountingReportAgent([])
    config = _config()

    resolutions = resolve_tier_b_reports(persistence_service, config, ["missing-art"], report_agent=agent)

    assert resolutions["missing-art"].status == "not_found"
    assert agent.batches == []


def test_empty_selection_never_constructs_or_calls_the_agent(persistence_service: PersistenceService):
    config = _config()
    resolutions = resolve_tier_b_reports(persistence_service, config, [])
    assert resolutions == {}


# ---------------------------------------------------------------------------
# Insufficient content never calls the model
# ---------------------------------------------------------------------------


def test_insufficient_content_never_calls_the_model(persistence_service: PersistenceService):
    _persist(persistence_service, "art1", clean_text=_INSUFFICIENT_TEXT)
    agent = _CountingReportAgent([])
    config = _config()

    resolutions = resolve_tier_b_reports(persistence_service, config, ["art1"], report_agent=agent)

    assert resolutions["art1"].status == "insufficient_content"
    assert resolutions["art1"].report is None
    assert agent.batches == []


# ---------------------------------------------------------------------------
# Sufficient, uncached articles generate exactly one report each, sequentially
# ---------------------------------------------------------------------------


def test_sufficient_uncached_article_generates_and_persists_a_report(persistence_service: PersistenceService):
    _persist(persistence_service, "art1", clean_text=_SUFFICIENT_TEXT)
    agent = _CountingReportAgent([_narrative()])
    config = _config()

    resolutions = resolve_tier_b_reports(persistence_service, config, ["art1"], report_agent=agent)

    assert resolutions["art1"].status == "generated"
    assert isinstance(resolutions["art1"].report, TierBReport)
    assert len(agent.batches) == 1
    assert len(agent.batches[0]) == 1

    conn = persistence_service.database.connect()
    try:
        cached = persistence_service.tier_b_reports.get(conn, "art1")
    finally:
        conn.close()
    assert cached is not None
    assert isinstance(cached.result, TierBReport)
    assert cached.schema_version == TIER_B_REPORT_SCHEMA_VERSION


def test_multiple_uncached_articles_are_generated_in_one_sequential_batch(persistence_service: PersistenceService):
    _persist(persistence_service, "art1", clean_text=_SUFFICIENT_TEXT, content_hash="h1")
    _persist(persistence_service, "art2", clean_text=_SUFFICIENT_TEXT, content_hash="h2")
    agent = _CountingReportAgent([_narrative(), _narrative()])
    config = _config()

    resolutions = resolve_tier_b_reports(persistence_service, config, ["art1", "art2"], report_agent=agent)

    assert resolutions["art1"].status == "generated"
    assert resolutions["art2"].status == "generated"
    assert len(agent.batches) == 1  # one sequential call to generate_tier_b_many, never two separate calls
    assert len(agent.batches[0]) == 2
    assert len(agent._llm.calls) == 2  # two sequential LLM calls, never concurrent


# ---------------------------------------------------------------------------
# Caching: valid cache hit -> zero model calls; invalidated by content/schema change
# ---------------------------------------------------------------------------


def test_valid_cached_report_is_reused_with_zero_model_calls(persistence_service: PersistenceService):
    _persist(persistence_service, "art1", clean_text=_SUFFICIENT_TEXT, content_hash="h1")
    from research_assistant.models.report import ReportSource
    from research_assistant.models.enums import SourcePriority, SourceType

    conn = persistence_service.database.connect()
    cached_report = TierBReport(
        article_id="art1", title="A headline", opening_context="Cached context.",
        key_insights=["Cached insight one.", "Cached insight two."],
        strategic_implications="Cached implications.", nomura_opportunity="Cached opportunity for Nomura.",
        source=ReportSource(
            publisher="Pub", title="A headline", publication_date=D, url="https://example.com/art1",
            source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
        ),
    )
    persistence_service.tier_b_reports.upsert_report(
        conn, cached_report, content_hash="h1", schema_version=TIER_B_REPORT_SCHEMA_VERSION, generated_at=D,
    )
    conn.commit()
    conn.close()

    agent = _CountingReportAgent([])
    config = _config()

    resolutions = resolve_tier_b_reports(persistence_service, config, ["art1"], report_agent=agent)

    assert resolutions["art1"].status == "cached"
    assert resolutions["art1"].report.opening_context == "Cached context."
    assert agent.batches == []


def test_changed_content_hash_invalidates_the_cache_and_regenerates(persistence_service: PersistenceService):
    _persist(persistence_service, "art1", clean_text=_SUFFICIENT_TEXT, content_hash="old-hash")
    from research_assistant.models.report import ReportSource
    from research_assistant.models.enums import SourcePriority, SourceType

    conn = persistence_service.database.connect()
    stale_report = TierBReport(
        article_id="art1", title="A headline", opening_context="Stale context.",
        key_insights=["Stale one.", "Stale two."], strategic_implications="Stale implications.",
        nomura_opportunity="Stale opportunity for Nomura.",
        source=ReportSource(
            publisher="Pub", title="A headline", publication_date=D, url="https://example.com/art1",
            source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
        ),
    )
    persistence_service.tier_b_reports.upsert_report(
        conn, stale_report, content_hash="a-different-hash", schema_version=TIER_B_REPORT_SCHEMA_VERSION,
        generated_at=D,
    )
    conn.commit()
    conn.close()

    agent = _CountingReportAgent([_narrative(opening_context="Fresh context.")])
    config = _config()

    resolutions = resolve_tier_b_reports(persistence_service, config, ["art1"], report_agent=agent)

    assert resolutions["art1"].status == "generated"
    assert resolutions["art1"].report.opening_context == "Fresh context."
    assert len(agent.batches) == 1


def test_changed_schema_version_invalidates_the_cache_and_regenerates(persistence_service: PersistenceService):
    _persist(persistence_service, "art1", clean_text=_SUFFICIENT_TEXT, content_hash="h1")
    from research_assistant.models.report import ReportSource
    from research_assistant.models.enums import SourcePriority, SourceType

    conn = persistence_service.database.connect()
    stale_report = TierBReport(
        article_id="art1", title="A headline", opening_context="Old schema context.",
        key_insights=["One.", "Two."], strategic_implications="Implications.",
        nomura_opportunity="Opportunity for Nomura.",
        source=ReportSource(
            publisher="Pub", title="A headline", publication_date=D, url="https://example.com/art1",
            source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
        ),
    )
    persistence_service.tier_b_reports.upsert_report(
        conn, stale_report, content_hash="h1", schema_version="tier_b_v0_old", generated_at=D,
    )
    conn.commit()
    conn.close()

    agent = _CountingReportAgent([_narrative()])
    config = _config()

    resolutions = resolve_tier_b_reports(persistence_service, config, ["art1"], report_agent=agent)

    assert resolutions["art1"].status == "generated"
    assert len(agent.batches) == 1


def test_a_cached_failure_is_never_a_valid_hit_and_a_retry_is_allowed(persistence_service: PersistenceService):
    _persist(persistence_service, "art1", clean_text=_SUFFICIENT_TEXT, content_hash="h1")
    conn = persistence_service.database.connect()
    failure = TierBReportFailure(article_id="art1", reason="LLMUnavailableError", detail="ollama down")
    persistence_service.tier_b_reports.upsert_failure(
        conn, failure, content_hash="h1", schema_version=TIER_B_REPORT_SCHEMA_VERSION, generated_at=D,
    )
    conn.commit()
    conn.close()

    agent = _CountingReportAgent([_narrative()])
    config = _config()

    resolutions = resolve_tier_b_reports(persistence_service, config, ["art1"], report_agent=agent)

    assert resolutions["art1"].status == "generated"  # the failure cache never short-circuits a retry
    assert len(agent.batches) == 1


# ---------------------------------------------------------------------------
# Failure handling -- per-item isolation
# ---------------------------------------------------------------------------


def test_one_articles_failure_does_not_block_the_others(persistence_service: PersistenceService):
    _persist(persistence_service, "art1", clean_text=_SUFFICIENT_TEXT, content_hash="h1")
    _persist(persistence_service, "art2", clean_text=_SUFFICIENT_TEXT, content_hash="h2")
    agent = _CountingReportAgent([LLMResponseError("bad json"), _narrative()])
    config = _config()

    resolutions = resolve_tier_b_reports(persistence_service, config, ["art1", "art2"], report_agent=agent)

    assert resolutions["art1"].status == "failed"
    assert isinstance(resolutions["art1"].failure, TierBReportFailure)
    assert resolutions["art2"].status == "generated"

    conn = persistence_service.database.connect()
    try:
        cached1 = persistence_service.tier_b_reports.get(conn, "art1")
        cached2 = persistence_service.tier_b_reports.get(conn, "art2")
    finally:
        conn.close()
    assert isinstance(cached1.result, TierBReportFailure)
    assert isinstance(cached2.result, TierBReport)


# ---------------------------------------------------------------------------
# Bounded cap on NEW generations per call
# ---------------------------------------------------------------------------


def test_new_generations_are_capped_and_the_rest_are_skipped(persistence_service: PersistenceService):
    for i in range(3):
        _persist(persistence_service, f"art{i}", clean_text=_SUFFICIENT_TEXT, content_hash=f"h{i}")
    agent = _CountingReportAgent([_narrative(), _narrative()])
    config = _config(max_tier_b_reports_per_newsletter_request=2)

    resolutions = resolve_tier_b_reports(
        persistence_service, config, ["art0", "art1", "art2"], report_agent=agent,
    )

    statuses = [resolutions[f"art{i}"].status for i in range(3)]
    assert statuses.count("generated") == 2
    assert statuses.count("skipped_cap") == 1
    assert len(agent.batches[0]) == 2


# ---------------------------------------------------------------------------
# Duplicate selection -- never a duplicate resolution/call
# ---------------------------------------------------------------------------


def test_duplicate_article_id_in_the_selection_is_resolved_and_called_once(persistence_service: PersistenceService):
    _persist(persistence_service, "art1", clean_text=_SUFFICIENT_TEXT)
    agent = _CountingReportAgent([_narrative()])
    config = _config()

    resolutions = resolve_tier_b_reports(persistence_service, config, ["art1", "art1"], report_agent=agent)

    assert resolutions["art1"].status == "generated"
    assert len(agent.batches[0]) == 1  # deduped -- never two calls for the same article


# ---------------------------------------------------------------------------
# Bounded, network-free re-extraction -- never a live fetch
# ---------------------------------------------------------------------------


def test_bounded_cache_reextraction_recovers_insufficient_persisted_text(persistence_service: PersistenceService, monkeypatch):
    _persist(persistence_service, "art1", clean_text=_INSUFFICIENT_TEXT)
    conn = persistence_service.database.connect()
    article = persistence_service.articles.get(conn, "art1")
    conn.close()

    from research_assistant.retrieval.models import ExtractedArticle, ExtractionStatus, RetrievalMethod, RetrievedPage
    import research_assistant.reporting.tier_b_newsletter_resolution as module

    richer_html = "<html><body><p>" + ("A much richer, fully extracted article body. " * 20) + "</p></body></html>"
    cached_page = RetrievedPage(
        url=article.url, final_url=article.url, status_code=200, retrieved_at=D,
        method=RetrievalMethod.HTTP, content_type="text/html", html=richer_html, success=True,
    )

    class _StubCache:
        def __init__(self, *, cache_dir, enabled):
            self.enabled = enabled

        def get(self, url):
            assert self.enabled, "cache_pages must be honored, even in the stub"
            return cached_page

    monkeypatch.setattr("research_assistant.retrieval.cache.PageCache", _StubCache)

    agent = _CountingReportAgent([_narrative()])
    config = _config()
    config.settings.retrieval.cache_pages = True

    resolutions = resolve_tier_b_reports(persistence_service, config, ["art1"], report_agent=agent)

    assert resolutions["art1"].status == "generated"
    assert len(agent.batches[0]) == 1
    assert "much richer" in agent.batches[0][0].text


def test_reextraction_disabled_when_cache_pages_setting_is_off(persistence_service: PersistenceService, monkeypatch):
    _persist(persistence_service, "art1", clean_text=_INSUFFICIENT_TEXT)

    called = {"n": 0}

    class _StubCache:
        def __init__(self, *, cache_dir, enabled):
            self.enabled = enabled

        def get(self, url):
            called["n"] += 1
            return None

    monkeypatch.setattr("research_assistant.retrieval.cache.PageCache", _StubCache)

    agent = _CountingReportAgent([])
    config = _config()  # cache_pages=False from _config()

    resolutions = resolve_tier_b_reports(persistence_service, config, ["art1"], report_agent=agent)

    assert resolutions["art1"].status == "insufficient_content"
    assert agent.batches == []


def test_resolve_tier_b_reports_never_references_the_live_http_client():
    """Structural guard: `resolve_tier_b_reports` (the automatic,
    newsletter-generation-time path) must only ever read the on-disk
    page cache directly, never RetrievalService/HTTPClient (whose
    `retrieve()`/`get()` fall back to a LIVE request on a cache miss).

    `fetch_full_article_for_tier_b` (added 2026-10-15) is a deliberate,
    narrow exception -- it IS the one explicit, user-triggered live
    fetch this feature offers, never called automatically by
    `resolve_tier_b_reports`, selection, or rendering -- so the guard is
    now scoped to `resolve_tier_b_reports`'s own source rather than the
    whole module."""
    import inspect

    from research_assistant.reporting import tier_b_newsletter_resolution as module

    source = inspect.getsource(module.resolve_tier_b_reports)
    assert "RetrievalService" not in source
    assert "HTTPClient" not in source


def test_manual_fetch_is_the_only_place_this_module_uses_the_live_http_client():
    """The live-fetch capability is confined to exactly one, explicitly
    user-triggered function -- never silently reachable from anywhere
    else in this module."""
    import inspect

    from research_assistant.reporting import tier_b_newsletter_resolution as module

    for name, obj in vars(module).items():
        if not inspect.isfunction(obj) or obj.__module__ != module.__name__:
            continue
        source = inspect.getsource(obj)
        if "HTTPClient" in source:
            assert name == "fetch_full_article_for_tier_b", (
                f"Unexpected live-HTTPClient reference in {name} -- only "
                "fetch_full_article_for_tier_b may use it."
            )
