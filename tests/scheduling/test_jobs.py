"""scheduling.jobs: independently callable, correct digest types, no
APScheduler dependency at all. Period-bound semantics (anchored to the
configured morning/evening hours, not a rolling window from "now") are
covered in detail in test_period_semantics.py; this file only checks that
the jobs correctly delegate to `compute_digest_period` and to the runner."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from research_assistant.models.digest import DigestType
from research_assistant.models.enums import ExecutionMode
from research_assistant.scheduling.jobs import (
    run_current_digest,
    run_evening_digest,
    run_manual_digest,
    run_morning_digest,
)

from tests.scheduling.conftest import D


def test_run_morning_digest_calls_runner_with_morning_type(fake_runner):
    result = run_morning_digest(fake_runner)

    assert len(fake_runner.calls) == 1
    assert fake_runner.calls[0]["digest_type"] == DigestType.MORNING
    assert result.digest_type == DigestType.MORNING


def test_run_evening_digest_calls_runner_with_evening_type(fake_runner):
    result = run_evening_digest(fake_runner)

    assert len(fake_runner.calls) == 1
    assert fake_runner.calls[0]["digest_type"] == DigestType.EVENING
    assert result.digest_type == DigestType.EVENING


def test_morning_digest_period_is_anchored_to_the_configured_hour_not_to_now(fake_runner):
    """D is 2025-09-10T12:00:00Z == 17:30 IST -- well after the configured
    08:00 morning cutoff. period_end must be that cutoff, not `now` itself
    (the bug this phase fixes: period_end used to always equal `now`)."""
    run_morning_digest(fake_runner)
    call = fake_runner.calls[0]

    expected_end = datetime(2025, 9, 10, 8, 0, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(timezone.utc)
    assert call["period_end"] == expected_end
    assert call["period_end"] != D


def test_morning_digest_period_covers_a_twelve_hour_window(fake_runner):
    run_morning_digest(fake_runner)
    call = fake_runner.calls[0]
    assert call["period_end"] - call["period_start"] == timedelta(hours=12)


def test_evening_digest_period_covers_a_twelve_hour_window(fake_runner):
    run_evening_digest(fake_runner)
    call = fake_runner.calls[0]
    assert call["period_end"] - call["period_start"] == timedelta(hours=12)


def test_jobs_are_independently_callable_without_a_scheduler_process(fake_runner):
    """The whole point of Part 2: these must be callable directly in tests,
    with no APScheduler instance involved anywhere in this test."""
    morning_result = run_morning_digest(fake_runner)
    evening_result = run_evening_digest(fake_runner)

    assert morning_result.status.value == "success"
    assert evening_result.status.value == "success"
    assert len(fake_runner.calls) == 2


def test_run_manual_digest_calls_runner_with_morning_type_and_period_ending_now(fake_runner):
    """Manual "Run Now" (Phase 1): period_end is `now` itself (unlike the
    scheduled digests, which anchor period_end to a configured hour)."""
    result = run_manual_digest(fake_runner)

    assert len(fake_runner.calls) == 1
    assert fake_runner.calls[0]["digest_type"] == DigestType.MORNING
    assert fake_runner.calls[0]["period_end"] == D
    assert result.digest_type == DigestType.MORNING


# ---------------------------------------------------------------------------
# Phase 14: execution_mode -- defaults to MANUAL everywhere (preserving the
# existing CLI's behavior exactly), only ever SCHEDULED when explicitly
# passed (which only `ResearchScheduler`'s own cron callback does).
# ---------------------------------------------------------------------------


def test_run_morning_digest_defaults_to_manual_execution_mode(fake_runner):
    result = run_morning_digest(fake_runner)
    assert result.execution_mode == ExecutionMode.MANUAL
    assert fake_runner.calls[0]["execution_mode"] == ExecutionMode.MANUAL


def test_run_evening_digest_defaults_to_manual_execution_mode(fake_runner):
    result = run_evening_digest(fake_runner)
    assert result.execution_mode == ExecutionMode.MANUAL


def test_run_morning_digest_records_scheduled_when_explicitly_passed(fake_runner):
    result = run_morning_digest(fake_runner, execution_mode=ExecutionMode.SCHEDULED)
    assert result.execution_mode == ExecutionMode.SCHEDULED
    assert fake_runner.calls[0]["execution_mode"] == ExecutionMode.SCHEDULED


def test_run_evening_digest_records_scheduled_when_explicitly_passed(fake_runner):
    result = run_evening_digest(fake_runner, execution_mode=ExecutionMode.SCHEDULED)
    assert result.execution_mode == ExecutionMode.SCHEDULED


def test_run_manual_digest_is_always_manual_even_if_a_scheduled_mode_were_hypothetically_desired(fake_runner):
    """`run_manual_digest` takes no `execution_mode` parameter at all --
    a manual "Run Now" can never be mistaken for a scheduler-triggered run."""
    result = run_manual_digest(fake_runner)
    assert result.execution_mode == ExecutionMode.MANUAL


# ---------------------------------------------------------------------------
# CURRENT mode (final production gate "Live Current" extension): a pure
# rolling window ending at `now`, independent of the
# morning_digest_hour/evening_digest_hour anchors the other jobs use. Width
# comes from `config.settings.research.current_window_hours` (default 48 --
# management requirement, widened from the original fixed 24).
# ---------------------------------------------------------------------------


def test_run_current_digest_calls_runner_with_current_type(fake_runner):
    result = run_current_digest(fake_runner)

    assert len(fake_runner.calls) == 1
    assert fake_runner.calls[0]["digest_type"] == DigestType.CURRENT
    assert result.digest_type == DigestType.CURRENT


def test_current_digest_period_is_a_rolling_48_hour_window_ending_at_now(fake_runner):
    """Unlike morning/evening (anchored to a configured hour) or manual
    (anchored to the morning hour), CURRENT's period_end is exactly `now`
    and period_start is exactly `current_window_hours` (default 48) before
    it -- never anchored to 08:00/20:00 at all."""
    run_current_digest(fake_runner)
    call = fake_runner.calls[0]

    assert call["period_end"] == D
    assert call["period_end"] - call["period_start"] == timedelta(hours=48)


def test_current_digest_period_width_follows_the_configured_window_hours(fake_runner):
    """Changing `research.current_window_hours` changes the period actually
    passed to the runner -- proves `run_current_digest` reads the
    configured value instead of a second hardcoded constant."""
    from copy import deepcopy

    config = deepcopy(fake_runner.services.config)
    config.settings.research.current_window_hours = 24
    fake_runner.services.config = config

    run_current_digest(fake_runner)
    call = fake_runner.calls[0]

    assert call["period_end"] == D
    assert call["period_end"] - call["period_start"] == timedelta(hours=24)


def test_run_current_digest_defaults_to_manual_execution_mode(fake_runner):
    result = run_current_digest(fake_runner)
    assert result.execution_mode == ExecutionMode.MANUAL
    assert fake_runner.calls[0]["execution_mode"] == ExecutionMode.MANUAL


def test_current_digest_never_waits_for_or_depends_on_the_scheduler():
    """CURRENT must execute immediately against whatever runner it is
    given -- this module has no APScheduler/ResearchScheduler import at
    all (verified below) and `run_current_digest` takes no scheduler
    argument, so there is structurally nothing for it to wait on."""
    import inspect

    assert "scheduler" not in inspect.signature(run_current_digest).parameters


def test_jobs_module_has_no_apscheduler_dependency():
    from pathlib import Path

    import research_assistant.scheduling.jobs as jobs_module

    lines = Path(jobs_module.__file__).read_text(encoding="utf-8").splitlines()
    import_lines = [line.strip() for line in lines if line.strip().startswith(("import ", "from "))]
    assert not any("apscheduler" in line.lower() for line in import_lines)
