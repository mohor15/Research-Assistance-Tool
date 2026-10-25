"""TierBReportRepository persistence/caching -- one row per article_id,
PRIMARY KEY makes a duplicate report structurally impossible (an upsert,
never a second row), exactly mirroring ReportRepository's own shape for
Tier A. article_id is a real FK against articles, so every test persists
a real Article first."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.models.report import ReportSource, TierBReport, TierBReportFailure
from research_assistant.storage.persistence_service import PersistenceService

from tests.storage.conftest import make_article

D = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)


def _persist_article(service: PersistenceService, article_id: str = "art1") -> None:
    conn = service.database.connect()
    service.articles.upsert(conn, make_article(article_id))
    conn.commit()
    conn.close()


def _report(article_id: str = "art1", **overrides) -> TierBReport:
    kwargs = dict(
        article_id=article_id, title="A real headline",
        opening_context="Something happened.", key_insights=["Insight one.", "Insight two."],
        strategic_implications="This may be relevant to markets.",
        nomura_opportunity="This may be relevant to Nomura's consideration of this area.",
        source=ReportSource(
            publisher="Example Publisher", title="A real headline", publication_date=D,
            url="https://example.com/a1", source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
        ),
        content_partial=False,
    )
    kwargs.update(overrides)
    return TierBReport(**kwargs)


def test_get_returns_none_when_nothing_cached(persistence_service: PersistenceService):
    conn = persistence_service.database.connect()
    try:
        assert persistence_service.tier_b_reports.get(conn, "art1") is None
    finally:
        conn.close()


def test_upsert_report_then_get_round_trips(persistence_service: PersistenceService):
    _persist_article(persistence_service)
    conn = persistence_service.database.connect()
    try:
        persistence_service.tier_b_reports.upsert_report(
            conn, _report(), content_hash="hash1", schema_version="tier_b_v1", generated_at=D,
        )
        conn.commit()
        cached = persistence_service.tier_b_reports.get(conn, "art1")
    finally:
        conn.close()

    assert cached is not None
    assert cached.content_hash == "hash1"
    assert cached.schema_version == "tier_b_v1"
    assert isinstance(cached.result, TierBReport)
    assert cached.result.key_insights == ["Insight one.", "Insight two."]
    assert cached.result.source.url == "https://example.com/a1"


def test_upsert_report_is_an_upsert_not_a_duplicate(persistence_service: PersistenceService):
    _persist_article(persistence_service)
    conn = persistence_service.database.connect()
    try:
        persistence_service.tier_b_reports.upsert_report(
            conn, _report(), content_hash="hash1", schema_version="tier_b_v1", generated_at=D,
        )
        persistence_service.tier_b_reports.upsert_report(
            conn, _report(opening_context="Updated context."), content_hash="hash2",
            schema_version="tier_b_v1", generated_at=D,
        )
        conn.commit()
        row_count = conn.execute("SELECT COUNT(*) FROM tier_b_reports WHERE article_id = 'art1'").fetchone()[0]
        cached = persistence_service.tier_b_reports.get(conn, "art1")
    finally:
        conn.close()

    assert row_count == 1
    assert cached.content_hash == "hash2"
    assert cached.result.opening_context == "Updated context."


def test_upsert_failure_then_get_round_trips(persistence_service: PersistenceService):
    _persist_article(persistence_service)
    conn = persistence_service.database.connect()
    try:
        failure = TierBReportFailure(article_id="art1", reason="LLMUnavailableError", detail="ollama down")
        persistence_service.tier_b_reports.upsert_failure(
            conn, failure, content_hash="hash1", schema_version="tier_b_v1", generated_at=D,
        )
        conn.commit()
        cached = persistence_service.tier_b_reports.get(conn, "art1")
    finally:
        conn.close()

    assert isinstance(cached.result, TierBReportFailure)
    assert cached.result.reason == "LLMUnavailableError"
    assert cached.result.detail == "ollama down"


def test_a_failure_overwriting_a_prior_success_clears_the_success_fields(persistence_service: PersistenceService):
    _persist_article(persistence_service)
    conn = persistence_service.database.connect()
    try:
        persistence_service.tier_b_reports.upsert_report(
            conn, _report(), content_hash="hash1", schema_version="tier_b_v1", generated_at=D,
        )
        failure = TierBReportFailure(article_id="art1", reason="LLMResponseError", detail="bad json")
        persistence_service.tier_b_reports.upsert_failure(
            conn, failure, content_hash="hash1", schema_version="tier_b_v1", generated_at=D,
        )
        conn.commit()
        cached = persistence_service.tier_b_reports.get(conn, "art1")
    finally:
        conn.close()

    assert isinstance(cached.result, TierBReportFailure)


def test_a_success_overwriting_a_prior_failure_clears_the_failure_fields(persistence_service: PersistenceService):
    _persist_article(persistence_service)
    conn = persistence_service.database.connect()
    try:
        failure = TierBReportFailure(article_id="art1", reason="LLMResponseError", detail="bad json")
        persistence_service.tier_b_reports.upsert_failure(
            conn, failure, content_hash="hash1", schema_version="tier_b_v1", generated_at=D,
        )
        persistence_service.tier_b_reports.upsert_report(
            conn, _report(), content_hash="hash1", schema_version="tier_b_v1", generated_at=D,
        )
        conn.commit()
        cached = persistence_service.tier_b_reports.get(conn, "art1")
    finally:
        conn.close()

    assert isinstance(cached.result, TierBReport)


def test_tier_a_reports_table_is_completely_unaffected(persistence_service: PersistenceService):
    """Confirms the new table is genuinely separate -- writing a Tier B
    report never touches the existing `reports` table or its rows."""
    _persist_article(persistence_service)
    conn = persistence_service.database.connect()
    try:
        before = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        persistence_service.tier_b_reports.upsert_report(
            conn, _report(), content_hash="hash1", schema_version="tier_b_v1", generated_at=D,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    finally:
        conn.close()

    assert before == after == 0
