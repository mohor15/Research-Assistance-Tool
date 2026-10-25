"""Phase 17, Section 14/20: for the SAME persisted CURRENT run, the admin
Scan Explorer's "CURRENT Qualifying Events" and the user CURRENT
dashboard's event set must be EXACTLY the same -- both now resolve through
the run's own `digest_id`, never an independent reconstruction."""

from __future__ import annotations

from datetime import timedelta

from research_assistant.config import get_config
from research_assistant.evidence import article_to_evidence
from research_assistant.models.digest import Digest, DigestType
from research_assistant.models.enums import DigestPersistenceStatus
from research_assistant.pipeline.runner import PipelineRunResult, RunStatus
from research_assistant.storage.persistence_service import PersistenceService
from research_assistant.ui import dashboard_data as dd

from tests.storage.conftest import D, make_article, make_event, make_report, make_research_item

CONFIG = get_config()


def _seed_qualified_event(service: PersistenceService, *, event_id: str, run_id: str):
    article = make_article(f"art-{event_id}", content_hash=f"hash-{event_id}")
    evidence = article_to_evidence(article)
    item = make_research_item(f"ri-{event_id}", article_id=article.id, evidence_ids=[evidence.evidence_id], run_id=run_id)
    event = make_event(event_id, research_item_ids=[item.id], article_ids=[article.id], event_date=D, run_id=run_id)
    report = make_report(event_id)
    service.persist_pipeline_result(
        articles=[article], evidence=[evidence], research_items=[item], events=[event], reports=[report],
    )
    return event


def test_admin_qualifying_events_exactly_equal_user_current_events(persistence_service: PersistenceService):
    run_id = "cur1"
    qualifying = _seed_qualified_event(persistence_service, event_id="ev-qualified", run_id=run_id)
    # A second, formed-but-NOT-qualifying event (no event_date) -- must
    # appear in admin diagnostics as unqualified, but never on the user page.
    non_qualifying = _seed_qualified_event(persistence_service, event_id="ev-unqualified", run_id=run_id)
    conn = persistence_service.database.connect()
    non_qualifying_unD = non_qualifying.model_copy(update={"event_date": None})
    persistence_service.events.upsert(conn, non_qualifying_unD)
    conn.commit()
    conn.close()

    digest = Digest(
        id="dg1", digest_type=DigestType.CURRENT, generated_at=D, period_start=D - timedelta(hours=24), period_end=D,
        events=["ev-qualified"], themes=[],
    )
    persistence_service.persist_pipeline_result(digest=digest)
    run = PipelineRunResult(
        run_id=run_id, digest_type=DigestType.CURRENT, started_at=D - timedelta(hours=24), finished_at=D,
        status=RunStatus.SUCCESS, counts={"research_items_relevant": 1},
        period_start=D - timedelta(hours=24), period_end=D,
        digest_persistence_status=DigestPersistenceStatus.PERSISTED, digest_id="dg1",
    )
    persistence_service.persist_run(run)

    admin_data = dd.get_current_scan_explorer(persistence_service, CONFIG, run_id)
    admin_qualified_ids = {e.event_id for e in admin_data.events if e.digest_qualified}

    user_view = dd.get_latest_current_intelligence(persistence_service, CONFIG)
    user_event_ids = {c.event_id for c in user_view.events}

    assert admin_qualified_ids == {"ev-qualified"}
    assert user_event_ids == {"ev-qualified"}
    assert admin_qualified_ids == user_event_ids

    # The unqualified event is visible in admin diagnostics (not hidden)...
    admin_all_ids = {e.event_id for e in admin_data.events}
    assert "ev-unqualified" in admin_all_ids
    # ...but never on the user page.
    assert "ev-unqualified" not in user_event_ids
