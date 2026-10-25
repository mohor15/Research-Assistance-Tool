"""dashboard_data.get_scheduler_status: the three-way health distinction
(RUNNING / NOT_CONFIRMED / STALE) the forensic diagnosis required -- never
inferred from `pipeline_runs`, only from the persisted `scheduler_state`
heartbeat."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.storage.persistence_service import PersistenceService
from research_assistant.ui import dashboard_data as dd

NOW = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)


def test_no_heartbeat_ever_recorded_is_not_confirmed(persistence_service: PersistenceService):
    status = dd.get_scheduler_status(persistence_service, now=NOW)
    assert status.health == dd.SCHEDULER_HEALTH_NOT_CONFIRMED
    assert status.last_heartbeat_at is None


def test_a_recent_heartbeat_is_running(persistence_service: PersistenceService):
    conn = persistence_service.database.connect()
    persistence_service.scheduler_state.upsert(conn, last_heartbeat_at=NOW - timedelta(seconds=30))
    conn.close()

    status = dd.get_scheduler_status(persistence_service, now=NOW)
    assert status.health == dd.SCHEDULER_HEALTH_RUNNING


def test_a_very_old_heartbeat_is_stale(persistence_service: PersistenceService):
    conn = persistence_service.database.connect()
    persistence_service.scheduler_state.upsert(conn, last_heartbeat_at=NOW - timedelta(hours=2))
    conn.close()

    status = dd.get_scheduler_status(persistence_service, now=NOW)
    assert status.health == dd.SCHEDULER_HEALTH_STALE


def test_stale_threshold_is_configurable(persistence_service: PersistenceService):
    conn = persistence_service.database.connect()
    persistence_service.scheduler_state.upsert(conn, last_heartbeat_at=NOW - timedelta(seconds=500))
    conn.close()

    assert dd.get_scheduler_status(persistence_service, now=NOW, stale_after_seconds=600).health == dd.SCHEDULER_HEALTH_RUNNING
    assert dd.get_scheduler_status(persistence_service, now=NOW, stale_after_seconds=400).health == dd.SCHEDULER_HEALTH_STALE


def test_a_stale_scheduler_does_not_claim_healthy_merely_because_an_old_run_exists(persistence_service: PersistenceService):
    """The task's own explicit requirement: 'Do not claim healthy merely
    because an old pipeline run exists.' A STALE heartbeat must report
    STALE regardless of what last_scheduled_* fields say."""
    conn = persistence_service.database.connect()
    persistence_service.scheduler_state.upsert(
        conn, last_heartbeat_at=NOW - timedelta(hours=5),
        last_scheduled_run_id="old-run", last_scheduled_status="success",
    )
    conn.close()

    status = dd.get_scheduler_status(persistence_service, now=NOW)
    assert status.health == dd.SCHEDULER_HEALTH_STALE
    assert status.last_scheduled_run_id == "old-run"  # still reported, just not conflated with health


def test_next_run_times_and_last_scheduled_fields_are_surfaced(persistence_service: PersistenceService):
    conn = persistence_service.database.connect()
    persistence_service.scheduler_state.upsert(
        conn, last_heartbeat_at=NOW, next_morning_run_at=NOW + timedelta(hours=11),
        next_evening_run_at=NOW + timedelta(hours=23), last_scheduled_digest_type="morning",
    )
    conn.close()

    status = dd.get_scheduler_status(persistence_service, now=NOW)
    assert status.next_morning_run_at == NOW + timedelta(hours=11)
    assert status.next_evening_run_at == NOW + timedelta(hours=23)
    assert status.last_scheduled_digest_type == "morning"
