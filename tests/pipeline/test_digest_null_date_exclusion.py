"""Phase 13D: a date-bounded operational digest (Morning/Evening) must never
silently include an event whose `event_date` is None (an evergreen/
reference development with no corroborated dated action -- see the
date-grounding guardrail in `agents.base`). This is a deliberate behavior
change from the original Phase 9 "never silently dropped" backstop, which
predates the Phase 13B/13C introduction of a genuinely-null, meaningful
event date. The event itself is never deleted or hidden elsewhere -- only
excluded from `digest.events`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.events.clock import FixedClock
from research_assistant.models.digest import DigestType
from research_assistant.pipeline.nodes import make_digest_generation_node
from research_assistant.pipeline.state import PipelineState
from research_assistant.reporting.digest_service import DigestService

from tests.storage.conftest import make_event

D = datetime(2026, 10, 1, 2, 30, tzinfo=timezone.utc)
PERIOD_START = D - timedelta(hours=12)
PERIOD_END = D


class _FakeServices:
    def __init__(self, config):
        self.clock = FixedClock(D)
        self.digest_service = DigestService(config, clock=self.clock)


def test_null_event_date_is_excluded_from_a_bounded_digest(config):
    dated_event = make_event("ev-dated", event_date=PERIOD_START + timedelta(hours=1))
    undated_event = make_event("ev-undated", event_date=None)

    state = PipelineState(
        events=[dated_event, undated_event], period_start=PERIOD_START, period_end=PERIOD_END,
        digest_type=DigestType.MORNING,
    )
    node = make_digest_generation_node(_FakeServices(config))
    result = node(state)

    assert result["digest"].events == [dated_event.id]
    assert undated_event.id not in result["digest"].events
    assert result["metrics"]["events_excluded_from_digest_period"] == 1


def test_an_in_period_dated_event_still_appears_normally(config):
    """Sanity: this change only removes the null-date special case --
    genuinely in-period dated events are completely unaffected."""
    dated_event = make_event("ev-dated", event_date=PERIOD_START + timedelta(hours=1))

    state = PipelineState(
        events=[dated_event], period_start=PERIOD_START, period_end=PERIOD_END, digest_type=DigestType.MORNING,
    )
    node = make_digest_generation_node(_FakeServices(config))
    result = node(state)

    assert result["digest"].events == [dated_event.id]


def test_an_out_of_period_dated_event_remains_excluded_as_before():
    """Unaffected pre-existing behavior: a dated event outside the window
    is still excluded, same as before this phase."""
    from research_assistant.config import get_config

    config = get_config()
    out_of_period_event = make_event("ev-old", event_date=PERIOD_START - timedelta(days=5))

    state = PipelineState(
        events=[out_of_period_event], period_start=PERIOD_START, period_end=PERIOD_END, digest_type=DigestType.MORNING,
    )
    node = make_digest_generation_node(_FakeServices(config))
    result = node(state)

    assert result["digest"].events == []
    assert result["metrics"]["events_excluded_from_digest_period"] == 1
