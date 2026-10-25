"""Dashboard-facing query/service functions, tested independently of
Streamlit (which is not required to be installed for these tests — see
README for why it could not be installed in this sandboxed environment).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.evidence import article_to_evidence
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, EventType
from research_assistant.models.report import Report, ReportFailure
from research_assistant.storage.persistence_service import PersistenceService
from research_assistant.ui import dashboard_data as dd

from tests.storage.conftest import D, make_article, make_digest, make_event, make_report, make_report_failure, make_research_item


def _seed_full_chain(service: PersistenceService, *, event_id="ev1", company="JPMorgan Chase", relevance=0.85, confidence=0.75, event_type=EventType.PRODUCT_LAUNCH, agent=AgentName.PEER):
    article = make_article("art-" + event_id, company=company, content_hash="hash-" + event_id)
    evidence = article_to_evidence(article)
    item = make_research_item("ri-" + event_id, article_id=article.id, company=company, agent=agent, event_type=event_type, evidence_ids=[evidence.evidence_id])
    event = make_event(event_id, research_item_ids=[item.id], article_ids=[article.id], company=company, relevance_score=relevance, confidence_score=confidence, event_type=event_type)
    report = make_report(event_id, relevance_score=relevance, confidence_score=confidence)
    service.persist_pipeline_result(articles=[article], evidence=[evidence], research_items=[item], events=[event], reports=[report])
    return event


def test_get_latest_intelligence_returns_summaries_sorted_by_relevance(persistence_service: PersistenceService):
    _seed_full_chain(persistence_service, event_id="ev1", relevance=0.5)
    _seed_full_chain(persistence_service, event_id="ev2", relevance=0.9)

    summaries = dd.get_latest_intelligence(persistence_service, limit=10)

    assert [s.id for s in summaries] == ["ev2", "ev1"]


def test_get_high_relevance_events_uses_existing_stored_score_only(persistence_service: PersistenceService):
    _seed_full_chain(persistence_service, event_id="ev1", relevance=0.9)
    _seed_full_chain(persistence_service, event_id="ev2", relevance=0.2)

    summaries = dd.get_high_relevance_events(persistence_service, min_relevance=0.5)

    assert [s.id for s in summaries] == ["ev1"]


def test_search_events_combines_filters(persistence_service: PersistenceService):
    _seed_full_chain(persistence_service, event_id="ev1", company="JPMorgan Chase", relevance=0.9)
    _seed_full_chain(persistence_service, event_id="ev2", company="OpenAI", relevance=0.9)

    summaries = dd.search_events(persistence_service, company="JPMorgan Chase", min_relevance=0.5)

    assert [s.id for s in summaries] == ["ev1"]


def test_get_market_activity_reports_counts(persistence_service: PersistenceService):
    _seed_full_chain(persistence_service, event_id="ev1", company="JPMorgan Chase", event_type=EventType.PRODUCT_LAUNCH)
    _seed_full_chain(persistence_service, event_id="ev2", company="JPMorgan Chase", event_type=EventType.FUNDING)

    activity = dd.get_market_activity(persistence_service)

    companies = {c.company: c.event_count for c in activity["by_company"]}
    assert companies["JPMorgan Chase"] == 2
    assert activity["recent_event_count"] == 2


def test_get_event_detail_returns_event_research_items_and_report(persistence_service: PersistenceService):
    _seed_full_chain(persistence_service, event_id="ev1")

    detail = dd.get_event_detail(persistence_service, "ev1")

    assert detail is not None
    assert detail.event.id == "ev1"
    assert len(detail.research_items) == 1
    assert isinstance(detail.report, Report)


def test_get_event_detail_returns_none_for_unknown_event(persistence_service: PersistenceService):
    assert dd.get_event_detail(persistence_service, "does-not-exist") is None


def test_get_event_detail_surfaces_a_report_failure(persistence_service: PersistenceService):
    article = make_article("art1")
    evidence = article_to_evidence(article)
    item = make_research_item("ri1", article_id="art1", evidence_ids=[evidence.evidence_id])
    event = make_event("ev1", research_item_ids=["ri1"], article_ids=["art1"])
    persistence_service.persist_pipeline_result(
        articles=[article], evidence=[evidence], research_items=[item], events=[event],
        report_failures=[make_report_failure("ev1")],
    )

    detail = dd.get_event_detail(persistence_service, "ev1")

    assert isinstance(detail.report, ReportFailure)


def test_get_source_evidence_returns_the_events_evidence(persistence_service: PersistenceService):
    _seed_full_chain(persistence_service, event_id="ev1")

    evidence_list = dd.get_source_evidence(persistence_service, "ev1")

    assert len(evidence_list) == 1
    assert evidence_list[0].article_id == "art-ev1"


def test_get_operational_digest_selects_the_exact_period_match_not_the_newest_row(persistence_service: PersistenceService):
    """Phase 13E: a much-wider historical/validation digest, even if it is
    the most recently generated row, must never be returned by
    `get_operational_digest` -- only a digest whose own period exactly
    matches the real operational window computed right now."""
    from research_assistant.config import get_config
    from research_assistant.scheduling.period import compute_digest_period

    config = get_config()
    now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)  # 14:30 IST -- after today's 08:00 IST morning anchor
    operational_start, operational_end = compute_digest_period(DigestType.MORNING, now, config)

    from research_assistant.models.digest import Digest

    operational = Digest(
        id="operational", digest_type=DigestType.MORNING, generated_at=now,
        period_start=operational_start, period_end=operational_end, events=[], themes=[],
    )
    stale_validation = Digest(
        id="stale", digest_type=DigestType.MORNING, generated_at=now,
        period_start=now - timedelta(days=7), period_end=now, events=[], themes=[],
    )
    # The stale/historical row is persisted LAST (so it would be "the
    # newest" by generated_at/insertion order) to prove selection is by
    # period match, never recency.
    persistence_service.persist_pipeline_result(digest=operational)
    persistence_service.persist_pipeline_result(digest=stale_validation)

    result = dd.get_operational_digest(persistence_service, DigestType.MORNING, config, now=now)

    assert result is not None
    assert result.id == "operational"


def test_get_operational_digest_returns_none_when_no_digest_matches_the_current_period(persistence_service: PersistenceService):
    """Must NOT silently fall back to any other row -- an absent operational
    digest is reported as absent, never masqueraded by a historical one."""
    from research_assistant.config import get_config
    from research_assistant.models.digest import Digest

    config = get_config()
    now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    stale_validation = Digest(
        id="stale", digest_type=DigestType.MORNING, generated_at=now,
        period_start=now - timedelta(days=7), period_end=now, events=[], themes=[],
    )
    persistence_service.persist_pipeline_result(digest=stale_validation)

    result = dd.get_operational_digest(persistence_service, DigestType.MORNING, config, now=now)

    assert result is None


def test_get_operational_digest_does_not_confuse_a_manual_run_with_the_scheduled_evening_digest(persistence_service: PersistenceService):
    """A manual run's digest (period_start anchored to morning_digest_hour,
    period_end = whatever instant the manual run happened to execute at)
    must not be mistaken for the scheduled Evening digest (whose period_end
    is always exactly the evening anchor hour) merely because both carry
    `digest_type=morning`... this test exercises the EVENING lookup
    specifically, which the manual digest's `digest_type=morning` already
    excludes by type, and additionally confirms period-mismatch would
    exclude it even if type matched."""
    from research_assistant.config import get_config
    from research_assistant.models.digest import Digest

    config = get_config()
    now = datetime(2026, 10, 1, 13, 30, tzinfo=timezone.utc)  # 19:00 IST -- before today's 20:00 IST evening anchor
    manual_run_digest = Digest(
        id="manual", digest_type=DigestType.MORNING, generated_at=now,
        period_start=now.replace(hour=2, minute=30, second=0, microsecond=0),  # 08:00 IST today in UTC
        period_end=now,  # "now", not the 20:00 IST anchor
        events=[], themes=[],
    )
    persistence_service.persist_pipeline_result(digest=manual_run_digest)

    assert dd.get_operational_digest(persistence_service, DigestType.EVENING, config, now=now) is None


def test_get_latest_digest_for_a_type(persistence_service: PersistenceService):
    event = _seed_full_chain(persistence_service, event_id="ev1")
    persistence_service.persist_pipeline_result(digest=make_digest("dg1", event_ids=[event.id], digest_type=DigestType.MORNING))

    digest = dd.get_latest_digest(persistence_service, DigestType.MORNING)

    assert digest is not None
    assert digest.id == "dg1"
    assert dd.get_latest_digest(persistence_service, DigestType.EVENING) is None


def test_get_recent_reports(persistence_service: PersistenceService):
    _seed_full_chain(persistence_service, event_id="ev1")

    reports = dd.get_recent_reports(persistence_service)

    assert len(reports) == 1


def test_get_events_for_period_uses_injected_reference_time(persistence_service: PersistenceService):
    recent = _seed_full_chain(persistence_service, event_id="ev1")

    within_window = dd.get_events_for_period(persistence_service, days=7, now=D)
    outside_window = dd.get_events_for_period(persistence_service, days=7, now=D + timedelta(days=30))

    assert [s.id for s in within_window] == ["ev1"]
    assert outside_window == []


def test_dashboard_data_module_never_imports_retrieval_llm_or_scoring():
    import research_assistant.ui.dashboard_data as dashboard_data_module
    from pathlib import Path

    lines = Path(dashboard_data_module.__file__).read_text(encoding="utf-8").splitlines()
    import_lines = [line.strip() for line in lines if line.strip().startswith(("import ", "from "))]
    assert not any("research_assistant.retrieval" in line for line in import_lines)
    assert not any("research_assistant.llm" in line for line in import_lines)
    assert not any("events.clustering" in line for line in import_lines)
    assert not any("events.scoring" in line for line in import_lines)
    assert not any(line.startswith("import streamlit") or "import streamlit" in line for line in import_lines)
