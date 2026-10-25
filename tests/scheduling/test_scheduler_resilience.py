"""Scheduler lifecycle resilience (2026-10-08 incident: the daemon's
heartbeat went stale and 08:00/20:00 runs stopped firing).

Forensic finding from `logs/daemon_restart_stderr.log`: the process was
killed externally with no exception, no traceback, no graceful-shutdown
log line -- not a code-level crash. But the same log also showed a REAL,
fixable bug: the 2026-10-07 08:00 IST morning run was silently dropped
("was missed by 1:50:41") because APScheduler's default `misfire_grace_time`
is 1 second, and `ResearchScheduler._run_and_record` re-raised scheduled-run
exceptions instead of isolating them from the scheduler's own lifecycle.

These tests exercise the real `ResearchScheduler`/`SchedulerStateRepository`
against a real temp-file `Database` -- never a live PipelineRunner, Qwen,
or network. CURRENT has no scheduled variant at all (manual-only), so this
file remains Morning/Evening-only throughout.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pytest

apscheduler = pytest.importorskip("apscheduler")

from research_assistant.events.clock import FixedClock
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import DigestPersistenceStatus, ExecutionMode
from research_assistant.pipeline.runner import PipelineRunResult, RunStatus
from research_assistant.scheduling.scheduler import (
    EVENING_JOB_ID,
    MISFIRE_GRACE_TIME_SECONDS,
    MORNING_JOB_ID,
    ResearchScheduler,
)
from research_assistant.storage.database import Database
from research_assistant.storage.persistence_service import PersistenceService

from tests.scheduling.conftest import FakePipelineRunner

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def _service(tmp_path: Path) -> PersistenceService:
    database = Database(tmp_path / "resilience.db")
    database.initialize()
    return PersistenceService(database)


class _SequencedPipelineRunner:
    """Like `FakePipelineRunner`, but each call pops the next outcome from a
    scripted sequence -- either a `PipelineRunResult` or a raised exception
    -- so a test can script exact "Evening fails, Morning succeeds" orders."""

    def __init__(self, outcomes: List, *, now: datetime = D):
        self._outcomes = list(outcomes)
        self.calls: List[dict] = []
        self.services = FakePipelineRunner(now=now).services

    def run(
        self, *, digest_type: DigestType, period_start: datetime, period_end: datetime,
        execution_mode: ExecutionMode = ExecutionMode.MANUAL,
    ) -> PipelineRunResult:
        self.calls.append({"digest_type": digest_type, "execution_mode": execution_mode})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return PipelineRunResult(
            run_id=f"run-{len(self.calls)}", digest_type=digest_type, started_at=period_end,
            finished_at=period_end, status=outcome, counts={"events_formed": 0},
            execution_mode=execution_mode, period_start=period_start, period_end=period_end,
            digest_attempted=True, digest_persisted=True, digest_persistence_status=DigestPersistenceStatus.PERSISTED,
        )


class _UpsertFailsSchedulerState:
    """Simulates a persistence-layer failure (e.g. a locked/corrupted DB)
    isolated to exactly the scheduler_state write -- everything else about
    the database stays real and untouched."""

    def upsert(self, *_args, **_kwargs):
        raise RuntimeError("simulated scheduler_state persistence failure")


class _PersistenceServiceWithFailingSchedulerState:
    def __init__(self, inner: PersistenceService):
        self.database = inner.database
        self.scheduler_state = _UpsertFailsSchedulerState()


# ---------------------------------------------------------------------------
# Test 1 -- scheduler survives a pipeline exception
# ---------------------------------------------------------------------------


def test_scheduler_survives_a_scheduled_pipeline_exception(tmp_path):
    service = _service(tmp_path)
    runner = _SequencedPipelineRunner([RuntimeError("Ollama unreachable")])
    scheduler = ResearchScheduler(config=runner.services.config, runner=runner, persistence_service=service)
    scheduler.register_jobs()
    jobs = {job.id: job for job in scheduler.get_jobs()}

    jobs[EVENING_JOB_ID].func()  # must not raise -- the exception is isolated here

    conn = service.database.connect()
    record = service.scheduler_state.get(conn)
    conn.close()

    assert record is not None
    assert record.last_scheduled_status is not None
    assert "exception" in record.last_scheduled_status
    assert record.last_heartbeat_at is not None  # heartbeat kept ticking despite the failure
    assert MORNING_JOB_ID in {job.id for job in scheduler.get_jobs()}  # next scheduled run remains available


# ---------------------------------------------------------------------------
# Test 2 -- Morning after failed Evening
# ---------------------------------------------------------------------------


def test_morning_still_executes_after_a_failed_evening(tmp_path):
    service = _service(tmp_path)
    runner = _SequencedPipelineRunner([RuntimeError("evening failed"), RunStatus.SUCCESS])
    scheduler = ResearchScheduler(config=runner.services.config, runner=runner, persistence_service=service)
    scheduler.register_jobs()
    jobs = {job.id: job for job in scheduler.get_jobs()}

    jobs[EVENING_JOB_ID].func()
    jobs[MORNING_JOB_ID].func()  # must still run and succeed

    assert len(runner.calls) == 2
    conn = service.database.connect()
    record = service.scheduler_state.get(conn)
    conn.close()
    assert record.last_scheduled_status == "success"
    assert record.last_scheduled_digest_type == "morning"


# ---------------------------------------------------------------------------
# Test 3 -- Evening after failed Morning
# ---------------------------------------------------------------------------


def test_evening_still_executes_after_a_failed_morning(tmp_path):
    service = _service(tmp_path)
    runner = _SequencedPipelineRunner([RuntimeError("morning failed"), RunStatus.SUCCESS])
    scheduler = ResearchScheduler(config=runner.services.config, runner=runner, persistence_service=service)
    scheduler.register_jobs()
    jobs = {job.id: job for job in scheduler.get_jobs()}

    jobs[MORNING_JOB_ID].func()
    jobs[EVENING_JOB_ID].func()  # must still run and succeed

    assert len(runner.calls) == 2
    conn = service.database.connect()
    record = service.scheduler_state.get(conn)
    conn.close()
    assert record.last_scheduled_status == "success"
    assert record.last_scheduled_digest_type == "evening"


# ---------------------------------------------------------------------------
# Test 4 -- repeated schedule lifecycle keeps going
# ---------------------------------------------------------------------------


def test_repeated_evening_morning_cycle_never_kills_the_scheduler(tmp_path):
    service = _service(tmp_path)
    runner = _SequencedPipelineRunner([
        RunStatus.SUCCESS, RuntimeError("fail 1"), RunStatus.SUCCESS, RuntimeError("fail 2"),
    ])
    scheduler = ResearchScheduler(config=runner.services.config, runner=runner, persistence_service=service)
    scheduler.register_jobs()
    jobs = {job.id: job for job in scheduler.get_jobs()}

    jobs[EVENING_JOB_ID].func()
    jobs[MORNING_JOB_ID].func()
    jobs[EVENING_JOB_ID].func()
    jobs[MORNING_JOB_ID].func()

    assert len(runner.calls) == 4  # all four fired -- none were skipped or crashed the scheduler
    conn = service.database.connect()
    record = service.scheduler_state.get(conn)
    conn.close()
    assert record.last_heartbeat_at is not None


# ---------------------------------------------------------------------------
# Test 5 -- no duplicate scheduler on application reload
# ---------------------------------------------------------------------------


def test_reload_style_restart_never_creates_a_second_live_instance(tmp_path):
    """Simulates the Streamlit-rerun-style scenario the forensic brief
    raised: something calling start() again on an already-running
    scheduler must not spin up a second concurrent instance or duplicate
    jobs."""
    service = _service(tmp_path)
    runner = FakePipelineRunner()
    scheduler = ResearchScheduler(config=runner.services.config, runner=runner, persistence_service=service)

    scheduler.start()
    try:
        jobs_after_first_start = {job.id for job in scheduler.get_jobs()}
        scheduler.start()  # a second "reload" attempt
        scheduler.start()  # and a third, for good measure
        jobs_after_reloads = {job.id for job in scheduler.get_jobs()}

        assert jobs_after_first_start == jobs_after_reloads  # exactly the same job set, no duplicates
        assert scheduler.is_running is True
    finally:
        scheduler.shutdown()


# ---------------------------------------------------------------------------
# Test 6 -- heartbeat continuously updates while the scheduler is alive
# ---------------------------------------------------------------------------


def test_heartbeat_advances_on_every_tick_while_alive(tmp_path):
    service = _service(tmp_path)
    runner = FakePipelineRunner()
    clock = FixedClock(D)
    scheduler = ResearchScheduler(
        config=runner.services.config, runner=runner, persistence_service=service, clock=clock,
    )

    scheduler.start()
    try:
        conn = service.database.connect()
        first = service.scheduler_state.get(conn).last_heartbeat_at
        conn.close()

        clock._when = clock._when + timedelta(minutes=5)
        scheduler._write_heartbeat()  # simulates the next interval tick firing

        conn = service.database.connect()
        second = service.scheduler_state.get(conn).last_heartbeat_at
        conn.close()

        assert second > first
    finally:
        scheduler.shutdown()


# ---------------------------------------------------------------------------
# Test 7 -- persistence failure during a scheduled run/heartbeat write
# ---------------------------------------------------------------------------


def test_scheduler_survives_a_persistence_failure(tmp_path):
    real_service = _service(tmp_path)
    failing_service = _PersistenceServiceWithFailingSchedulerState(real_service)
    runner = _SequencedPipelineRunner([RunStatus.SUCCESS, RunStatus.SUCCESS])
    scheduler = ResearchScheduler(config=runner.services.config, runner=runner, persistence_service=failing_service)
    scheduler.register_jobs()
    jobs = {job.id: job for job in scheduler.get_jobs()}

    jobs[EVENING_JOB_ID].func()  # scheduler_state.upsert raises internally -- must not propagate
    jobs[MORNING_JOB_ID].func()  # the next scheduled run is still possible after the persistence failure

    assert len(runner.calls) == 2


# ---------------------------------------------------------------------------
# Misfire grace time -- the exact mechanism that silently dropped the real
# 2026-10-07 08:00 IST morning run (APScheduler default grace is 1 second).
# ---------------------------------------------------------------------------


def test_morning_and_evening_jobs_have_a_generous_misfire_grace_time(config):
    scheduler = ResearchScheduler(config, FakePipelineRunner())
    scheduler.register_jobs()

    jobs = {job.id: job for job in scheduler.get_jobs()}
    assert jobs[MORNING_JOB_ID].misfire_grace_time == MISFIRE_GRACE_TIME_SECONDS
    assert jobs[EVENING_JOB_ID].misfire_grace_time == MISFIRE_GRACE_TIME_SECONDS
    assert MISFIRE_GRACE_TIME_SECONDS > 60  # must be well beyond APScheduler's 1-second default
