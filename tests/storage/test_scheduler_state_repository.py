"""SchedulerStateRepository: singleton-row upsert, partial updates (a
caller touching only a subset of fields must never clobber the others --
different callers in `scheduling.scheduler` update different subsets at
different times)."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.storage.database import Database
from research_assistant.storage.repositories.scheduler_state import SchedulerStateRepository

D = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)


def test_get_returns_none_before_any_upsert(database: Database):
    conn = database.connect()
    record = SchedulerStateRepository().get(conn)
    conn.close()
    assert record is None


def test_upsert_then_get_round_trips_all_fields(database: Database):
    conn = database.connect()
    SchedulerStateRepository().upsert(
        conn, started_at=D, last_heartbeat_at=D, next_morning_run_at=D, next_evening_run_at=D,
        last_scheduled_run_id="run-1", last_scheduled_execution_mode="scheduled",
        last_scheduled_digest_type="morning", last_scheduled_started_at=D, last_scheduled_finished_at=D,
        last_scheduled_status="success",
    )
    record = SchedulerStateRepository().get(conn)
    conn.close()

    assert record.started_at == D
    assert record.last_heartbeat_at == D
    assert record.last_scheduled_run_id == "run-1"
    assert record.last_scheduled_execution_mode == "scheduled"
    assert record.last_scheduled_status == "success"


def test_a_partial_upsert_does_not_clobber_previously_set_fields(database: Database):
    conn = database.connect()
    repo = SchedulerStateRepository()
    repo.upsert(conn, started_at=D, last_heartbeat_at=D)

    later = D.replace(minute=5)
    repo.upsert(conn, last_heartbeat_at=later)  # only the heartbeat tick -- started_at must survive

    record = repo.get(conn)
    conn.close()

    assert record.started_at == D  # unchanged
    assert record.last_heartbeat_at == later  # updated


def test_recording_a_scheduled_run_does_not_clobber_the_heartbeat(database: Database):
    conn = database.connect()
    repo = SchedulerStateRepository()
    repo.upsert(conn, last_heartbeat_at=D)
    repo.upsert(conn, last_scheduled_run_id="run-1", last_scheduled_status="success")

    record = repo.get(conn)
    conn.close()

    assert record.last_heartbeat_at == D
    assert record.last_scheduled_run_id == "run-1"


def test_repeated_upsert_remains_a_single_row(database: Database):
    conn = database.connect()
    repo = SchedulerStateRepository()
    repo.upsert(conn, last_heartbeat_at=D)
    repo.upsert(conn, last_heartbeat_at=D)
    repo.upsert(conn, last_heartbeat_at=D)

    count = conn.execute("SELECT COUNT(*) FROM scheduler_state").fetchone()[0]
    conn.close()
    assert count == 1
