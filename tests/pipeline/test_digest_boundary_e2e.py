"""Phase 13F ULTIMATUM 1 mandatory end-to-end test: a digest's period is a
half-open interval [period_start, period_end) -- this test builds eight
deliberately boundary-placed events (A-H, per the task's own exact
specification), runs the real `digest_generation` node, PERSISTS the result
to a real (isolated, temp-file) SQLite database, and re-reads it back out
-- proving actual persisted digest contents, not merely
`compute_digest_period`'s own math (already covered by
`tests/scheduling/test_period_semantics.py`).

IST = UTC+5:30. Morning Oct 1 2026: 2026-09-30 20:00 IST (2026-09-30T14:30Z)
-> 2026-10-01 08:00 IST (2026-10-01T02:30Z). Evening Oct 1 2026:
2026-10-01 08:00 IST (2026-10-01T02:30Z) -> 2026-10-01 20:00 IST
(2026-10-01T14:30Z).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from research_assistant.config import get_config
from research_assistant.events.clock import FixedClock
from research_assistant.models.digest import DigestType
from research_assistant.pipeline.nodes import make_digest_generation_node
from research_assistant.pipeline.state import PipelineState
from research_assistant.pipeline import PipelineServices
from research_assistant.reporting.digest_service import DigestService
from research_assistant.storage.database import Database
from research_assistant.storage.persistence_service import PersistenceService

from tests.storage.conftest import make_article, make_event, make_research_item

MORNING_START = datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)   # Sep 30 20:00 IST
MORNING_END = datetime(2026, 10, 1, 2, 30, tzinfo=timezone.utc)      # Oct 1 08:00 IST
EVENING_END = datetime(2026, 10, 1, 14, 30, tzinfo=timezone.utc)     # Oct 1 20:00 IST

EVENT_DATES = {
    "A": datetime(2026, 9, 30, 14, 29, 59, tzinfo=timezone.utc),   # Sep 30 19:59:59 IST
    "B": datetime(2026, 9, 30, 14, 30, 0, tzinfo=timezone.utc),    # Sep 30 20:00:00 IST
    "C": datetime(2026, 10, 1, 2, 29, 59, tzinfo=timezone.utc),    # Oct 1 07:59:59 IST
    "D": datetime(2026, 10, 1, 2, 30, 0, tzinfo=timezone.utc),     # Oct 1 08:00:00 IST
    "E": datetime(2026, 10, 1, 2, 30, 1, tzinfo=timezone.utc),     # Oct 1 08:00:01 IST
    "F": datetime(2026, 10, 1, 14, 29, 59, tzinfo=timezone.utc),   # Oct 1 19:59:59 IST
    "G": datetime(2026, 10, 1, 14, 30, 0, tzinfo=timezone.utc),    # Oct 1 20:00:00 IST
    "H": None,
}


def _build_events_and_persist(persistence_service: PersistenceService):
    events = []
    for label, event_date in EVENT_DATES.items():
        article = make_article(f"art-{label}", content_hash=f"hash-{label}", publication_date=event_date)
        item = make_research_item(f"ri-{label}", article_id=article.id, publication_date=event_date)
        event = make_event(f"ev-{label}", research_item_ids=[item.id], article_ids=[article.id], event_date=event_date)
        persistence_service.persist_pipeline_result(articles=[article], evidence=[], research_items=[item], events=[event])
        events.append(event)
    return events


def _generate_and_persist_digest(persistence_service: PersistenceService, events, *, digest_type: DigestType, period_start, period_end):
    config = get_config()
    services = PipelineServices(
        config=config,
        retrieval_service=None, normalization_service=None, agents={}, report_agent=None,
        event_service=None,
        digest_service=DigestService(config, clock=FixedClock(period_end)),
        clock=FixedClock(period_end),
    )
    node = make_digest_generation_node(services)
    state = PipelineState(events=events, digest_type=digest_type, period_start=period_start, period_end=period_end)
    result = node(state)
    digest = result["digest"]
    persistence_service.persist_pipeline_result(digest=digest)
    return digest


def test_mandatory_boundary_e2e_morning_and_evening_digest_persisted_contents(tmp_path: Path):
    database = Database(tmp_path / "boundary_e2e.db")
    database.initialize()
    persistence_service = PersistenceService(database)

    events = _build_events_and_persist(persistence_service)

    morning_digest = _generate_and_persist_digest(
        persistence_service, events, digest_type=DigestType.MORNING,
        period_start=MORNING_START, period_end=MORNING_END,
    )
    evening_digest = _generate_and_persist_digest(
        persistence_service, events, digest_type=DigestType.EVENING,
        period_start=MORNING_END, period_end=EVENING_END,
    )

    # Re-read back from the real database -- proving PERSISTED contents,
    # not just the in-memory node return value.
    conn = database.connect()
    try:
        reread_morning = persistence_service.digests.get(conn, morning_digest.id)
        reread_evening = persistence_service.digests.get(conn, evening_digest.id)
    finally:
        conn.close()

    assert set(reread_morning.events) == {"ev-B", "ev-C"}
    assert set(reread_evening.events) == {"ev-D", "ev-E", "ev-F"}

    for excluded_id in ("ev-A", "ev-G", "ev-H"):
        assert excluded_id not in reread_morning.events
        assert excluded_id not in reread_evening.events
