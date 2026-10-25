"""Phase 17 (CURRENT user-page snapshot-fallback UX): the user-facing
CURRENT page must keep showing the latest USABLE completed CURRENT
snapshot -- never a blank/failed page -- while a newer scan attempt is
running, has failed, or produced no qualifying events. Every displayed
event always remains attributed to its own original run/digest; nothing
is ever merged across runs. Fixtures/fakes only -- no live network, no
LLM, no pipeline execution, no scheduler."""

from __future__ import annotations

import inspect
from datetime import timedelta

from research_assistant.config import get_config
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import DigestPersistenceStatus
from research_assistant.pipeline.runner import RunStatus
from research_assistant.storage.persistence_service import PersistenceService
from research_assistant.ui import dashboard_data as dd

from tests.storage.conftest import D, make_digest
from tests.ui.test_current_user_dashboard import _run, _seed_event

CONFIG = get_config()


def _seed_snapshot(service, *, run_id, event_ids, period_start, period_end, **run_kwargs):
    for event_id in event_ids:
        _seed_event(service, event_id=event_id)
    service.persist_pipeline_result(digest=make_digest(f"dg-{run_id}", event_ids=event_ids, digest_type=DigestType.CURRENT))
    service.persist_run(_run(run_id, digest_id=f"dg-{run_id}", period_start=period_start, period_end=period_end, **run_kwargs))


# ---------------------------------------------------------------------------
# Case A: previous snapshot (5 events), new run RUNNING
# ---------------------------------------------------------------------------


def test_case_a_running_scan_keeps_showing_the_previous_snapshot(persistence_service: PersistenceService):
    old_events = [f"ev-old-{i}" for i in range(5)]
    _seed_snapshot(
        persistence_service, run_id="cur-old", event_ids=old_events,
        period_start=D - timedelta(days=1), period_end=D,
    )
    conn = persistence_service.database.connect()
    persistence_service.current_run_lock.try_acquire(conn, run_id="cur-running", started_at=D)
    conn.close()

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.status == dd.CURRENT_STATUS_OK
    assert view.run_id == "cur-old"
    assert sorted(c.event_id for c in view.events) == sorted(old_events)
    assert view.is_newer_scan_running is True
    # No events from the (unpersisted, still-running) new run leak in --
    # there IS no persisted run row for it yet, so this is structurally
    # guaranteed, not just incidentally true.
    assert len(view.events) == 5


def test_case_a_no_scan_trigger_is_exposed_on_the_user_page():
    from research_assistant.ui.app import _render_user_current_page

    source = inspect.getsource(_render_user_current_page)
    assert "st.button" not in source
    assert "PipelineRunner" not in source
    assert "run_current_digest" not in source


# ---------------------------------------------------------------------------
# Case B: previous snapshot (5 events), new run SUCCEEDS with 3 events
# ---------------------------------------------------------------------------


def test_case_b_new_successful_run_replaces_the_previous_snapshot(persistence_service: PersistenceService):
    old_events = [f"ev-old-{i}" for i in range(5)]
    new_events = [f"ev-new-{i}" for i in range(3)]
    _seed_snapshot(
        persistence_service, run_id="cur-old", event_ids=old_events,
        period_start=D - timedelta(days=1), period_end=D,
    )
    _seed_snapshot(
        persistence_service, run_id="cur-new", event_ids=new_events,
        period_start=D, period_end=D + timedelta(hours=1),
    )

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.status == dd.CURRENT_STATUS_OK
    assert view.run_id == "cur-new"
    assert sorted(c.event_id for c in view.events) == sorted(new_events)
    assert not any(c.event_id in old_events for c in view.events)
    assert view.latest_attempt_outcome is None
    assert view.is_newer_scan_running is False


# ---------------------------------------------------------------------------
# Case C: previous snapshot (5 events), new run FAILS
# ---------------------------------------------------------------------------


def test_case_c_failed_new_run_keeps_showing_the_previous_snapshot(persistence_service: PersistenceService):
    old_events = [f"ev-old-{i}" for i in range(5)]
    _seed_snapshot(
        persistence_service, run_id="cur-old", event_ids=old_events,
        period_start=D - timedelta(days=1), period_end=D,
    )
    persistence_service.persist_run(
        _run("cur-new", status=RunStatus.FAILED, digest_id=None,
             digest_persistence_status=DigestPersistenceStatus.NOT_ATTEMPTED,
             period_start=D, period_end=D + timedelta(hours=1))
    )

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.status == dd.CURRENT_STATUS_OK
    assert view.run_id == "cur-old"
    assert sorted(c.event_id for c in view.events) == sorted(old_events)
    assert view.latest_attempt_outcome == "failed"
    assert view.is_newer_scan_running is False


# ---------------------------------------------------------------------------
# Case D: previous snapshot (5 events), new run completes NO_QUALIFYING_EVENTS
# ---------------------------------------------------------------------------


def test_case_d_no_qualifying_events_run_keeps_showing_the_previous_snapshot(persistence_service: PersistenceService):
    old_events = [f"ev-old-{i}" for i in range(5)]
    _seed_snapshot(
        persistence_service, run_id="cur-old", event_ids=old_events,
        period_start=D - timedelta(days=1), period_end=D,
    )
    persistence_service.persist_run(
        _run("cur-new", digest_id=None, digest_persistence_status=DigestPersistenceStatus.NO_QUALIFYING_EVENTS,
             period_start=D, period_end=D + timedelta(hours=1))
    )

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.status == dd.CURRENT_STATUS_OK
    # Old events retain their ORIGINAL run ownership -- never reattributed
    # to the new (empty) run.
    assert view.run_id == "cur-old"
    assert sorted(c.event_id for c in view.events) == sorted(old_events)
    assert view.latest_attempt_outcome == "no_qualifying_events"


# ---------------------------------------------------------------------------
# Case D2 (forensic finding, CURRENT run `31fa192e4d7a7175`): NO_QUALIFYING_
# EVENTS with genuine triage-relevant winners AND model timeouts/failures
# must be distinguished from a clean "nothing qualifying was found" run --
# never worded the same, even though both persist the same
# `digest_persistence_status`.
# ---------------------------------------------------------------------------


def test_case_d2_no_qualifying_events_with_model_timeouts_gets_the_incomplete_analysis_outcome(
    persistence_service: PersistenceService,
):
    old_events = [f"ev-old-{i}" for i in range(5)]
    _seed_snapshot(
        persistence_service, run_id="cur-old", event_ids=old_events,
        period_start=D - timedelta(days=1), period_end=D,
    )
    persistence_service.persist_run(
        _run(
            "cur-new", digest_id=None, digest_persistence_status=DigestPersistenceStatus.NO_QUALIFYING_EVENTS,
            period_start=D, period_end=D + timedelta(hours=1),
            counts={"triage_relevant": 7, "deep_analysis_timeouts": 2, "deep_analysis_failures": 0},
        )
    )

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.latest_attempt_outcome == "no_qualifying_events_incomplete_analysis"


def test_case_d3_no_qualifying_events_with_zero_triage_relevant_keeps_the_plain_outcome(
    persistence_service: PersistenceService,
):
    """Timeouts alone, with nothing triage ever judged relevant in the
    first place, is not 'relevant work was interrupted' -- stays the plain
    `no_qualifying_events` wording."""
    old_events = [f"ev-old-{i}" for i in range(5)]
    _seed_snapshot(
        persistence_service, run_id="cur-old", event_ids=old_events,
        period_start=D - timedelta(days=1), period_end=D,
    )
    persistence_service.persist_run(
        _run(
            "cur-new", digest_id=None, digest_persistence_status=DigestPersistenceStatus.NO_QUALIFYING_EVENTS,
            period_start=D, period_end=D + timedelta(hours=1),
            counts={"triage_relevant": 0, "deep_analysis_timeouts": 0, "deep_analysis_failures": 0},
        )
    )

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.latest_attempt_outcome == "no_qualifying_events"


# ---------------------------------------------------------------------------
# Case E: no previous usable snapshot at all
# ---------------------------------------------------------------------------


def test_case_e_no_snapshot_ever_shows_the_empty_state(persistence_service: PersistenceService):
    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.status == dd.CURRENT_STATUS_NO_RUN
    assert view.events == []
    assert view.is_newer_scan_running is False


# ---------------------------------------------------------------------------
# Case F: two old usable snapshots -> newest wins
# ---------------------------------------------------------------------------


def test_case_f_newest_of_two_usable_snapshots_wins(persistence_service: PersistenceService):
    oldest_events = ["ev-oldest-1"]
    newer_events = ["ev-newer-1"]
    _seed_snapshot(
        persistence_service, run_id="cur-oldest", event_ids=oldest_events,
        period_start=D - timedelta(days=3), period_end=D - timedelta(days=2),
    )
    _seed_snapshot(
        persistence_service, run_id="cur-newer", event_ids=newer_events,
        period_start=D - timedelta(days=1), period_end=D,
    )

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.status == dd.CURRENT_STATUS_OK
    assert view.run_id == "cur-newer"
    assert [c.event_id for c in view.events] == newer_events


# ---------------------------------------------------------------------------
# Case G: newsletter selection on a displayed FALLBACK snapshot
# ---------------------------------------------------------------------------


def test_case_g_newsletter_selection_works_on_the_displayed_fallback_snapshot(persistence_service: PersistenceService):
    old_events = ["ev-old-1", "ev-old-2"]
    _seed_snapshot(
        persistence_service, run_id="cur-old", event_ids=old_events,
        period_start=D - timedelta(days=1), period_end=D,
    )
    persistence_service.persist_run(
        _run("cur-new", status=RunStatus.FAILED, digest_id=None,
             digest_persistence_status=DigestPersistenceStatus.NOT_ATTEMPTED,
             period_start=D, period_end=D + timedelta(hours=1))
    )

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)
    assert view.run_id == "cur-old"
    displayed_ids = [c.event_id for c in view.events]

    added = dd.toggle_newsletter_selection(persistence_service, displayed_ids[0], now=D)
    assert added is True
    assert dd.get_newsletter_selection(persistence_service) == [displayed_ids[0]]

    markdown = dd.render_newsletter_markdown(persistence_service, [displayed_ids[0]])
    assert markdown.count("###") == 1
