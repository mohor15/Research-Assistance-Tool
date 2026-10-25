"""Regression coverage for the reported system-wide UI crash:

    ValueError: 'current' is not a valid DigestType

Root cause (confirmed by inspection, not guessed): there is exactly ONE
`DigestType` definition (`research_assistant.models.digest.DigestType`),
and every importer -- `pipeline.runner`, `pipeline.state`,
`reporting.digest_service`, `scheduling.jobs`/`period`,
`storage.repositories.run`/`digest`, `ui.app`/`dashboard_data` -- resolves
it from that same module; it already includes `CURRENT = "current"`. The
failure was therefore never a source-level enum gap -- it was a stale,
already-running Python process (a long-lived Streamlit session) that had
imported the pre-CURRENT `DigestType` class into memory before the enum
was extended, and never reloaded it. These tests exist to prove the fix
holds at the data layer, independent of any particular running process,
and that a persisted CURRENT row never breaks a page that has nothing to
do with CURRENT mode."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.config import get_config
from research_assistant.evidence import article_to_evidence
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, EventType
from research_assistant.pipeline.runner import PipelineRunResult, RunStatus
from research_assistant.storage.persistence_service import PersistenceService
from research_assistant.ui import dashboard_data as dd

from tests.storage.conftest import make_article, make_event, make_report, make_research_item

D = datetime(2026, 10, 3, 18, 22, 14, tzinfo=timezone.utc)


def _seed_historical_chain(service: PersistenceService, *, event_id: str, company: str, agent: AgentName) -> None:
    """A normal, pre-existing historical event -- exactly what Peers/AI
    Providers/Analysts/Dashboard read, and what must keep working
    regardless of whether a CURRENT run has ever executed."""
    article = make_article("art-" + event_id, company=company, content_hash="hash-" + event_id)
    evidence = article_to_evidence(article)
    item = make_research_item(
        "ri-" + event_id, article_id=article.id, company=company, agent=agent,
        event_type=EventType.PRODUCT_LAUNCH, evidence_ids=[evidence.evidence_id],
    )
    event = make_event(
        event_id, research_item_ids=[item.id], article_ids=[article.id], company=company,
        relevance_score=0.8, confidence_score=0.7, event_type=EventType.PRODUCT_LAUNCH,
    )
    report = make_report(event_id, relevance_score=0.8, confidence_score=0.7)
    service.persist_pipeline_result(
        articles=[article], evidence=[evidence], research_items=[item], events=[event], reports=[report],
    )


def _seed_current_run(service: PersistenceService, *, run_id: str = "current-r1") -> None:
    run = PipelineRunResult(
        run_id=run_id, digest_type=DigestType.CURRENT, started_at=D, finished_at=D, status=RunStatus.SUCCESS,
        counts={"events_formed": 0},
    )
    service.persist_run(run)


def test_dashboard_s_latest_run_summary_survives_a_current_row_being_the_newest(persistence_service: PersistenceService):
    """`get_latest_run_summary` (used by the Dashboard page) queries
    `pipeline_runs` WITHOUT filtering by digest_type -- the exact query
    shape that would hit any still-broken `DigestType(row["digest_type"])`
    call the moment a CURRENT row is the most recent one."""
    _seed_current_run(persistence_service, run_id="r-current")

    summary = dd.get_latest_run_summary(persistence_service)

    assert summary is not None
    assert summary.digest_type == DigestType.CURRENT
    assert summary.run_id == "r-current"


def test_current_live_scan_page_reads_its_own_run_without_raising(persistence_service: PersistenceService):
    _seed_current_run(persistence_service, run_id="r-current")

    run = dd.get_latest_run_for_digest_type(persistence_service, DigestType.CURRENT)
    digest = dd.get_latest_digest(persistence_service, DigestType.CURRENT)

    assert run is not None
    assert run.digest_type == DigestType.CURRENT
    assert digest is None  # no digest was persisted in this seed -- must not raise, just return None


def test_peers_ai_providers_analysts_pages_are_unaffected_by_a_current_run_existing(persistence_service: PersistenceService):
    """The core property the report demanded: a CURRENT pipeline run in the
    database must never make an unrelated historical-monitoring page
    crash. This is the mechanism that actually makes the bug "system-wide":
    `get_entity_statuses` (which Peers/AI Providers/Analysts all call) goes
    through `has_any_pipeline_run_completed`/`_latest_retrieve_failures`,
    both of which call the SAME unfiltered `RunRepository.get_recent(limit=1)`
    that `get_latest_run_summary` (Dashboard) and
    `get_latest_run_for_digest_type` (CURRENT page) use -- so a CURRENT row
    being the newest run breaks every one of these pages via the one
    shared code path, exactly as reported."""
    _seed_current_run(persistence_service, run_id="r-current")  # the newest run is CURRENT
    config = get_config()

    peer_statuses = dd.get_peer_entities(persistence_service, config)
    ai_provider_statuses = dd.get_ai_provider_entities(persistence_service, config)
    analyst_statuses = dd.get_analyst_entities(persistence_service, config)

    assert len(peer_statuses) == len(config.sources.peers)
    assert len(ai_provider_statuses) == len(config.sources.ai_providers)
    assert len(analyst_statuses) == len(config.sources.analysts)


def test_dashboard_latest_intelligence_and_market_activity_are_unaffected_by_a_current_run(persistence_service: PersistenceService):
    """The rest of the Dashboard page's own data calls -- none of them
    filter by digest_type either, so none of them were ever the actual
    failure point, but this confirms the whole page's data layer survives
    a CURRENT row existing, not just the one call that was."""
    _seed_current_run(persistence_service, run_id="r-current")
    _seed_historical_chain(persistence_service, event_id="peer1", company="JPMorgan Chase", agent=AgentName.PEER)

    latest = dd.get_latest_intelligence(persistence_service, limit=10)
    activity = dd.get_market_activity(persistence_service)
    high_relevance = dd.get_high_relevance_events(persistence_service, min_relevance=0.0, limit=10)

    assert [s.id for s in latest] == ["peer1"]
    assert activity is not None
    assert [s.id for s in high_relevance] == ["peer1"]


def test_get_recent_runs_unfiltered_includes_current_rows_without_raising(persistence_service: PersistenceService):
    """`get_recent(digest_type=None)` -- the unfiltered listing explicitly
    named in the report ("do NOT filter CURRENT rows out of get_recent()")
    -- must return the CURRENT row alongside morning/evening ones, not
    raise and not silently drop it."""
    _seed_current_run(persistence_service, run_id="r-current")
    morning_run = PipelineRunResult(
        run_id="r-morning", digest_type=DigestType.MORNING, started_at=D, finished_at=D, status=RunStatus.SUCCESS,
    )
    persistence_service.persist_run(morning_run)

    conn = persistence_service.database.connect()
    try:
        recent = persistence_service.runs.get_recent(conn, limit=10)
    finally:
        conn.close()

    assert {r.run_id for r in recent} == {"r-current", "r-morning"}
    assert {r.digest_type for r in recent} == {DigestType.CURRENT, DigestType.MORNING}
