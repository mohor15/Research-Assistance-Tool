"""ResearchScheduler: construction, timezone, morning/evening job
registration, correct digest types/period bounds via job introspection,
duplicate-start guard, clean shutdown.

`apscheduler` could not be installed in this sandboxed environment (no
outbound network access — same limitation as feedparser/trafilatura/
playwright/streamlit in earlier phases). This whole file auto-skips via
`pytest.importorskip` and will start running the moment it's installed —
`scheduling.jobs` (tested in test_jobs.py) has zero dependency on it and is
fully tested regardless.
"""

from __future__ import annotations

import pytest

apscheduler = pytest.importorskip("apscheduler")

from research_assistant.models.enums import ExecutionMode
from research_assistant.scheduling.scheduler import (
    EVENING_JOB_ID,
    HEARTBEAT_JOB_ID,
    MORNING_JOB_ID,
    ResearchScheduler,
)

from tests.scheduling.conftest import FakePipelineRunner


def test_scheduler_constructs(config):
    scheduler = ResearchScheduler(config, FakePipelineRunner())
    assert scheduler.is_running is False


def test_scheduler_uses_the_configured_timezone(config):
    scheduler = ResearchScheduler(config, FakePipelineRunner())
    assert scheduler.timezone == "Asia/Kolkata"
    assert scheduler.timezone == config.settings.system.timezone


def _next_fire_hour(trigger, *, after) -> int:
    """Uses APScheduler's own public trigger API (the same method the
    scheduler itself calls internally) rather than reaching into a cron
    trigger's private field representation."""
    next_fire_time = trigger.get_next_fire_time(None, after)
    return next_fire_time.hour


def test_morning_job_is_registered_at_the_configured_hour(config):
    import datetime as dt

    scheduler = ResearchScheduler(config, FakePipelineRunner())
    scheduler.register_jobs()

    jobs = {job.id: job for job in scheduler.get_jobs()}
    assert MORNING_JOB_ID in jobs
    reference = dt.datetime(2025, 1, 1, tzinfo=jobs[MORNING_JOB_ID].trigger.timezone)
    assert _next_fire_hour(jobs[MORNING_JOB_ID].trigger, after=reference) == config.settings.research.morning_digest_hour


def test_evening_job_is_registered_at_the_configured_hour(config):
    import datetime as dt

    scheduler = ResearchScheduler(config, FakePipelineRunner())
    scheduler.register_jobs()

    jobs = {job.id: job for job in scheduler.get_jobs()}
    assert EVENING_JOB_ID in jobs
    reference = dt.datetime(2025, 1, 1, tzinfo=jobs[EVENING_JOB_ID].trigger.timezone)
    assert _next_fire_hour(jobs[EVENING_JOB_ID].trigger, after=reference) == config.settings.research.evening_digest_hour


def test_both_jobs_call_pipeline_runner_when_fired(config):
    runner = FakePipelineRunner()
    scheduler = ResearchScheduler(config, runner)
    scheduler.register_jobs()

    jobs = {job.id: job for job in scheduler.get_jobs()}
    jobs[MORNING_JOB_ID].func()
    jobs[EVENING_JOB_ID].func()

    assert len(runner.calls) == 2
    digest_types = {call["digest_type"].value for call in runner.calls}
    assert digest_types == {"morning", "evening"}


def test_scheduled_jobs_are_recorded_with_execution_mode_scheduled(config):
    """Phase 14: closes the forensic-diagnosis finding directly -- the ONLY
    code path that should ever produce `execution_mode=SCHEDULED` is this
    scheduler's own cron callback."""
    runner = FakePipelineRunner()
    scheduler = ResearchScheduler(config, runner)
    scheduler.register_jobs()

    jobs = {job.id: job for job in scheduler.get_jobs()}
    jobs[MORNING_JOB_ID].func()

    assert runner.calls[0]["execution_mode"] == ExecutionMode.SCHEDULED


def test_heartbeat_job_is_registered(config):
    scheduler = ResearchScheduler(config, FakePipelineRunner())
    scheduler.register_jobs()

    jobs = {job.id: job for job in scheduler.get_jobs()}
    assert HEARTBEAT_JOB_ID in jobs


def test_start_and_shutdown_toggle_is_running(config):
    scheduler = ResearchScheduler(config, FakePipelineRunner())

    scheduler.start()
    assert scheduler.is_running is True

    scheduler.shutdown()
    assert scheduler.is_running is False


def test_starting_twice_does_not_create_a_duplicate_instance(config):
    scheduler = ResearchScheduler(config, FakePipelineRunner())
    scheduler.start()
    scheduler.start()  # must not raise, must not create a second underlying scheduler

    assert scheduler.is_running is True
    scheduler.shutdown()


def test_shutdown_before_start_is_a_safe_no_op(config):
    scheduler = ResearchScheduler(config, FakePipelineRunner())
    scheduler.shutdown()  # must not raise
    assert scheduler.is_running is False


# ---------------------------------------------------------------------------
# CURRENT mode isolation: CURRENT has no scheduled variant at all -- it is
# only ever triggered manually (the UI's "Run Live Current Scan" button),
# and its automatic email delivery fires on completion regardless of
# execution mode (see `PipelineRunner._deliver_digest_if_eligible`), not via
# any scheduler job. The scheduler itself must never register, own, or
# invoke a CURRENT job.
# ---------------------------------------------------------------------------


def test_scheduler_registers_exactly_two_digest_jobs_never_a_current_job(config):
    scheduler = ResearchScheduler(config, FakePipelineRunner())
    scheduler.register_jobs()

    job_ids = {job.id for job in scheduler.get_jobs()}
    assert job_ids == {MORNING_JOB_ID, EVENING_JOB_ID, HEARTBEAT_JOB_ID}


def test_scheduler_module_never_imports_run_current_digest():
    """Structural guard: the scheduler module must never reference
    `run_current_digest` -- CURRENT is manual-only, and its automatic
    email delivery is wired directly in `PipelineRunner`, not here."""
    from pathlib import Path

    import research_assistant.scheduling.scheduler as scheduler_module

    source = Path(scheduler_module.__file__).read_text(encoding="utf-8")
    assert "run_current_digest" not in source
