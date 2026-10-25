"""fetch_full_article_for_tier_b -- the ONE explicit, user-triggered, live
network action this feature introduces. Bounded to exactly one request
for exactly one article's own URL, no retry loop, never automatic (never
called by resolve_tier_b_reports, selection, or rendering). All network
access is mocked here; nothing in this file touches the real network."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from research_assistant.config import get_config
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.models.evidence import Evidence
from research_assistant.reporting.tier_b_newsletter_resolution import fetch_full_article_for_tier_b
from research_assistant.retrieval.extractor import extract_article
from research_assistant.retrieval.models import ExtractionMethod, RetrievalFailure, RetrievalMethod, RetrievedPage
from research_assistant.storage.persistence_service import PersistenceService

from tests.storage.conftest import make_article

D = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)

_SHORT_TEXT = "Too short."
_RICH_HTML = "<html><body><article>" + ("A real, substantive paragraph of article content. " * 40) + "</article></body></html>"
_THIN_HTML = "<html><body><p>Still too short.</p></body></html>"


def _config():
    config = get_config().model_copy(deep=True)
    config.settings.retrieval.cache_pages = False
    return config


def _persist(service: PersistenceService, article_id: str, *, clean_text: str, url: str = "https://example.com/art1"):
    conn = service.database.connect()
    article = make_article(article_id, clean_text=clean_text, url=url, canonical_url=url)
    service.articles.upsert(conn, article)
    conn.commit()
    conn.close()
    return article


def _success_page(url: str, html_body: str) -> RetrievedPage:
    return RetrievedPage(
        url=url, final_url=url, retrieved_at=D, method=RetrievalMethod.HTTP,
        content_type="text/html", html=html_body, success=True,
    )


def _failure_page(url: str, *, reason: str) -> RetrievedPage:
    return RetrievedPage(
        url=url, final_url=url, retrieved_at=D, method=RetrievalMethod.HTTP,
        success=False, failure=RetrievalFailure(reason=reason),
    )


def test_not_found_article_returns_not_found(persistence_service: PersistenceService):
    result = fetch_full_article_for_tier_b(persistence_service, _config(), "missing")
    assert result.status == "not_found"


def test_successful_richer_fetch_improves_and_persists_article_text(persistence_service: PersistenceService):
    url = "https://example.com/rich"
    _persist(persistence_service, "art1", clean_text=_SHORT_TEXT, url=url)

    with patch(
        "research_assistant.retrieval.client.HTTPClient.get",
        return_value=_success_page(url, _RICH_HTML),
    ):
        result = fetch_full_article_for_tier_b(persistence_service, _config(), "art1")

    assert result.status == "improved"
    assert result.new_text_chars is not None and result.new_text_chars > len(_SHORT_TEXT)

    conn = persistence_service.database.connect()
    updated = persistence_service.articles.get(conn, "art1")
    conn.close()
    assert updated.clean_text is not None
    assert len(updated.clean_text) > len(_SHORT_TEXT)
    assert updated.content_hash is not None


def test_fetch_that_does_not_improve_never_overwrites_existing_content(persistence_service: PersistenceService):
    url = "https://example.com/thin"
    rich_existing = "A real, substantive paragraph of article content. " * 40
    _persist(persistence_service, "art1", clean_text=rich_existing, url=url)

    with patch(
        "research_assistant.retrieval.client.HTTPClient.get",
        return_value=_success_page(url, _THIN_HTML),
    ):
        result = fetch_full_article_for_tier_b(persistence_service, _config(), "art1")

    assert result.status == "no_improvement"

    conn = persistence_service.database.connect()
    unchanged = persistence_service.articles.get(conn, "art1")
    conn.close()
    assert unchanged.clean_text == rich_existing  # untouched


def test_network_failure_preserves_existing_content_and_reports_classified_reason(persistence_service: PersistenceService):
    url = "https://example.com/blocked"
    existing = "Some existing text."
    _persist(persistence_service, "art1", clean_text=existing, url=url)

    with patch(
        "research_assistant.retrieval.client.HTTPClient.get",
        return_value=_failure_page(url, reason="robots_disallowed"),
    ):
        result = fetch_full_article_for_tier_b(persistence_service, _config(), "art1")

    assert result.status == "failed"
    assert result.detail

    conn = persistence_service.database.connect()
    unchanged = persistence_service.articles.get(conn, "art1")
    conn.close()
    assert unchanged.clean_text == existing


def test_oversized_response_is_rejected_without_extraction(persistence_service: PersistenceService):
    url = "https://example.com/huge"
    existing = "Some existing text."
    _persist(persistence_service, "art1", clean_text=existing, url=url)

    config = _config()
    config.settings.current_llm.tier_b_manual_fetch_max_html_chars = 100
    huge_html = "<html><body>" + ("x" * 500) + "</body></html>"

    with patch(
        "research_assistant.retrieval.client.HTTPClient.get",
        return_value=_success_page(url, huge_html),
    ):
        result = fetch_full_article_for_tier_b(persistence_service, config, "art1")

    assert result.status == "response_too_large"

    conn = persistence_service.database.connect()
    unchanged = persistence_service.articles.get(conn, "art1")
    conn.close()
    assert unchanged.clean_text == existing


def test_fetch_makes_exactly_one_request_no_retry_loop(persistence_service: PersistenceService):
    url = "https://example.com/once"
    _persist(persistence_service, "art1", clean_text=_SHORT_TEXT, url=url)

    calls = {"n": 0}

    def fake_get(self, request_url, *, cache_policy):
        calls["n"] += 1
        return _failure_page(request_url, reason="connection_error")

    with patch("research_assistant.retrieval.client.HTTPClient.get", fake_get):
        fetch_full_article_for_tier_b(persistence_service, _config(), "art1")

    assert calls["n"] == 1


def test_associated_evidence_row_is_updated_in_place_not_duplicated(persistence_service: PersistenceService):
    url = "https://example.com/evidence"
    _persist(persistence_service, "art1", clean_text=_SHORT_TEXT, url=url)

    conn = persistence_service.database.connect()
    evidence = Evidence(
        evidence_id="ev1", article_id="art1", source_url=url, canonical_url=url,
        publisher="Example", title="An article title", publication_date=D, retrieved_at=D,
        content=_SHORT_TEXT, content_hash="hash1", source_type=SourceType.PRIMARY,
        source_priority=SourcePriority.HIGH, extraction_method=ExtractionMethod.BS4_FALLBACK,
    )
    persistence_service.evidence.upsert(conn, evidence)
    conn.commit()
    conn.close()

    with patch(
        "research_assistant.retrieval.client.HTTPClient.get",
        return_value=_success_page(url, _RICH_HTML),
    ):
        result = fetch_full_article_for_tier_b(persistence_service, _config(), "art1")

    assert result.status == "improved"

    conn = persistence_service.database.connect()
    rows = persistence_service.evidence.get_by_article_id(conn, "art1")
    conn.close()
    assert len(rows) == 1  # updated in place, never duplicated
    assert rows[0].evidence_id == "ev1"
    assert len(rows[0].content) > len(_SHORT_TEXT)
    assert rows[0].content_hash is not None


def test_a_real_extractor_round_trip_sanity_check():
    """Not mocking extract_article itself here -- confirms the rich/thin
    HTML fixtures above actually produce the extraction statuses the
    other tests assume, using the real extractor."""
    rich = extract_article(_RICH_HTML, "https://example.com/rich", content_type="text/html")
    thin = extract_article(_THIN_HTML, "https://example.com/thin", content_type="text/html")
    assert rich.text and len(rich.text) > len(_SHORT_TEXT)
    assert (thin.text or "") == "" or len(thin.text) < 200
