"""RunRepository: run creation, success/partial-success/failure persistence,
and idempotent upsert-by-run_id."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.models.digest import DigestType
from research_assistant.models.enums import DigestPersistenceStatus, ExecutionMode
from research_assistant.pipeline.runner import PipelineRunResult, RunStatus
from research_assistant.pipeline.state import PipelineDiagnostic
from research_assistant.storage.database import Database
from research_assistant.storage.repositories.run import RunRepository

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def _run(run_id="r1", *, status=RunStatus.SUCCESS, diagnostics=None, counts=None) -> PipelineRunResult:
    return PipelineRunResult(
        run_id=run_id, digest_type=DigestType.MORNING, started_at=D, finished_at=D,
        status=status,
        diagnostics=diagnostics if diagnostics is not None else [],
        counts=counts if counts is not None else {"events_formed": 3},
    )


def test_persist_and_retrieve_a_successful_run(database: Database):
    conn = database.connect()
    RunRepository().upsert(conn, _run("r1", status=RunStatus.SUCCESS))
    conn.commit()

    fetched = RunRepository().get(conn, "r1")
    conn.close()

    assert fetched is not None
    assert fetched.status == RunStatus.SUCCESS
    assert fetched.counts == {"events_formed": 3}


def test_persist_and_retrieve_a_partial_success_run(database: Database):
    conn = database.connect()
    RunRepository().upsert(
        conn, _run("r1", status=RunStatus.PARTIAL_SUCCESS,
                   diagnostics=[PipelineDiagnostic(stage="retrieve", message="one source failed")]),
    )
    conn.commit()

    fetched = RunRepository().get(conn, "r1")
    conn.close()

    assert fetched.status == RunStatus.PARTIAL_SUCCESS
    assert len(fetched.diagnostics) == 1
    assert fetched.diagnostics[0].stage == "retrieve"


def test_persist_and_retrieve_a_failed_run(database: Database):
    conn = database.connect()
    RunRepository().upsert(
        conn, _run("r1", status=RunStatus.FAILED, counts={},
                   diagnostics=[PipelineDiagnostic(stage="pipeline_run", message="catastrophic failure", severity="error")]),
    )
    conn.commit()

    fetched = RunRepository().get(conn, "r1")
    conn.close()

    assert fetched.status == RunStatus.FAILED
    assert fetched.counts == {}


def test_upsert_is_idempotent_and_does_not_duplicate(database: Database):
    conn = database.connect()
    RunRepository().upsert(conn, _run("r1", status=RunStatus.SUCCESS))
    RunRepository().upsert(conn, _run("r1", status=RunStatus.PARTIAL_SUCCESS))  # same run_id, updated status
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
    fetched = RunRepository().get(conn, "r1")
    conn.close()

    assert count == 1
    assert fetched.status == RunStatus.PARTIAL_SUCCESS


def test_get_missing_run_returns_none(database: Database):
    conn = database.connect()
    fetched = RunRepository().get(conn, "does-not-exist")
    conn.close()
    assert fetched is None


def test_get_recent_orders_by_started_at_descending(database: Database):
    from datetime import timedelta

    conn = database.connect()
    RunRepository().upsert(conn, _run("r1"))
    later = PipelineRunResult(run_id="r2", digest_type=DigestType.MORNING, started_at=D + timedelta(hours=1), finished_at=D + timedelta(hours=1), status=RunStatus.SUCCESS)
    RunRepository().upsert(conn, later)
    conn.commit()

    recent = RunRepository().get_recent(conn, limit=10)
    conn.close()

    assert [r.run_id for r in recent] == ["r2", "r1"]


def test_get_recent_filters_by_digest_type(database: Database):
    conn = database.connect()
    RunRepository().upsert(conn, _run("r1"))  # morning
    evening_run = PipelineRunResult(run_id="r2", digest_type=DigestType.EVENING, started_at=D, finished_at=D, status=RunStatus.SUCCESS)
    RunRepository().upsert(conn, evening_run)
    conn.commit()

    morning_only = RunRepository().get_recent(conn, digest_type=DigestType.MORNING)
    conn.close()

    assert [r.run_id for r in morning_only] == ["r1"]


# ---------------------------------------------------------------------------
# Phase 14: explicit execution/digest-persistence metadata round-trips.
# ---------------------------------------------------------------------------


def test_execution_mode_round_trips(database: Database):
    conn = database.connect()
    run = PipelineRunResult(
        run_id="r1", digest_type=DigestType.MORNING, started_at=D, finished_at=D, status=RunStatus.SUCCESS,
        execution_mode=ExecutionMode.SCHEDULED,
    )
    RunRepository().upsert(conn, run)
    conn.commit()

    fetched = RunRepository().get(conn, "r1")
    conn.close()
    assert fetched.execution_mode == ExecutionMode.SCHEDULED


def test_period_bounds_round_trip(database: Database):
    from datetime import timedelta

    conn = database.connect()
    run = PipelineRunResult(
        run_id="r1", digest_type=DigestType.MORNING, started_at=D, finished_at=D, status=RunStatus.SUCCESS,
        period_start=D - timedelta(hours=12), period_end=D,
    )
    RunRepository().upsert(conn, run)
    conn.commit()

    fetched = RunRepository().get(conn, "r1")
    conn.close()
    assert fetched.period_start == D - timedelta(hours=12)
    assert fetched.period_end == D


def test_digest_persistence_metadata_round_trips(database: Database):
    conn = database.connect()
    run = PipelineRunResult(
        run_id="r1", digest_type=DigestType.MORNING, started_at=D, finished_at=D, status=RunStatus.SUCCESS,
        digest_attempted=True, digest_persisted=True,
        digest_persistence_status=DigestPersistenceStatus.NO_QUALIFYING_EVENTS, digest_id="dg1",
    )
    RunRepository().upsert(conn, run)
    conn.commit()

    fetched = RunRepository().get(conn, "r1")
    conn.close()
    assert fetched.digest_attempted is True
    assert fetched.digest_persisted is True
    assert fetched.digest_persistence_status == DigestPersistenceStatus.NO_QUALIFYING_EVENTS
    assert fetched.digest_id == "dg1"


# ---------------------------------------------------------------------------
# DigestType.CURRENT compatibility fix: `_row_to_run` must deserialize a
# "current" row exactly like "morning"/"evening" -- regression coverage for
# the system-wide UI crash (`ValueError: 'current' is not a valid
# DigestType`) caused by a stale long-running process holding a pre-fix
# `DigestType` class in memory, not by any source-level enum gap (confirmed:
# there is exactly one `DigestType` definition, in `models.digest`, and
# every importer -- including this repository -- resolves it from there).
# ---------------------------------------------------------------------------


def test_current_digest_type_round_trips(database: Database):
    conn = database.connect()
    run = PipelineRunResult(
        run_id="r1", digest_type=DigestType.CURRENT, started_at=D, finished_at=D, status=RunStatus.SUCCESS,
    )
    RunRepository().upsert(conn, run)
    conn.commit()

    fetched = RunRepository().get(conn, "r1")
    conn.close()
    assert fetched.digest_type == DigestType.CURRENT


def test_existing_morning_and_evening_values_still_deserialize_correctly(database: Database):
    """Confirms the CURRENT fix did not rename or otherwise disturb the
    pre-existing enum values -- both still round-trip exactly as before."""
    conn = database.connect()
    RunRepository().upsert(conn, _run("r1", status=RunStatus.SUCCESS))  # digest_type=MORNING
    evening_run = PipelineRunResult(
        run_id="r2", digest_type=DigestType.EVENING, started_at=D, finished_at=D, status=RunStatus.SUCCESS,
    )
    RunRepository().upsert(conn, evening_run)
    conn.commit()

    morning = RunRepository().get(conn, "r1")
    evening = RunRepository().get(conn, "r2")
    conn.close()
    assert morning.digest_type == DigestType.MORNING
    assert evening.digest_type == DigestType.EVENING


def test_a_raw_current_row_inserted_directly_deserializes_without_raising(database: Database):
    """Reproduces the exact reported failure shape: a `pipeline_runs` row
    that already has `digest_type='current'` in the database (as a real
    CURRENT-mode production run would persist), read back through
    `_row_to_run` without going through `RunRepository.upsert` first --
    proves the fix works for data that was ALREADY in the database, not
    only for rows this test process itself just wrote."""
    import json

    conn = database.connect()
    conn.execute(
        """
        INSERT INTO pipeline_runs (
            run_id, digest_type, started_at, finished_at, status, diagnostics_json, counts_json,
            execution_mode, period_start, period_end, digest_attempted, digest_persisted,
            digest_persistence_status, digest_id
        ) VALUES (?, 'current', ?, ?, 'success', '[]', '{}', 'manual', NULL, NULL, 0, 0, 'not_attempted', NULL)
        """,
        ("raw-current-r1", "2026-10-03T18:22:14+00:00", "2026-10-03T19:16:49+00:00"),
    )
    conn.commit()

    fetched = RunRepository().get(conn, "raw-current-r1")
    conn.close()

    assert fetched is not None
    assert fetched.digest_type == DigestType.CURRENT


def test_get_recent_filters_by_current_digest_type(database: Database):
    conn = database.connect()
    RunRepository().upsert(conn, _run("r1"))  # morning
    current_run = PipelineRunResult(
        run_id="r2", digest_type=DigestType.CURRENT, started_at=D, finished_at=D, status=RunStatus.SUCCESS,
    )
    RunRepository().upsert(conn, current_run)
    conn.commit()

    current_only = RunRepository().get_recent(conn, digest_type=DigestType.CURRENT)
    all_recent = RunRepository().get_recent(conn, limit=10)
    conn.close()

    assert [r.run_id for r in current_only] == ["r2"]
    assert {r.run_id for r in all_recent} == {"r1", "r2"}  # a CURRENT row never breaks the unfiltered listing either


def test_a_run_with_no_explicit_metadata_gets_the_documented_safe_defaults(database: Database):
    """Mirrors a pre-Phase-14 row read back after migration (see
    test_phase14_migration.py) -- defaults must be MANUAL/NOT_ATTEMPTED,
    never a guess."""
    conn = database.connect()
    RunRepository().upsert(conn, _run("r1"))
    conn.commit()

    fetched = RunRepository().get(conn, "r1")
    conn.close()
    assert fetched.execution_mode == ExecutionMode.MANUAL
    assert fetched.digest_attempted is False
    assert fetched.digest_persistence_status == DigestPersistenceStatus.NOT_ATTEMPTED
