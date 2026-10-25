"""Explicit, independently-callable scheduled jobs.

`run_morning_digest`/`run_evening_digest` compute their own period bounds
deterministically from the runner's own injected `Clock` and configuration
(never a bare `datetime.now()`, and never "the last 12 hours from now"), and
delegate entirely to `PipelineRunner` — no retrieval, agent, scoring,
report, or database logic lives here, and this module has no dependency on
APScheduler at all, so it is fully testable without it installed (see
`scheduling.scheduler` for the actual scheduler).

Period semantics (`compute_digest_period`): a digest's period is anchored to
the configured `morning_digest_hour`/`evening_digest_hour` in the configured
IANA timezone (`config.settings.system.timezone`), not to the wall-clock
time the job happens to run at. Concretely, a digest's `period_end` is the
most recently completed occurrence of its own anchor hour at-or-before now,
and its `period_start` is the immediately preceding occurrence of the
*other* digest's anchor hour. This means:

* A morning job firing at exactly 08:00 IST reports 20:00 IST yesterday ->
  08:00 IST today, as intended.
* A manual `morning` run executed later the same day (e.g. 20:32 IST) still
  reports that same 20:00-yesterday -> 08:00-today window, because today's
  08:00 occurrence is still the most recent completed one — it does NOT
  fall back to "the last 12 hours from now".
* A manual run executed *before* its anchor hour (e.g. a "morning" run at
  03:00 IST, before today's 08:00) reaches back to *yesterday's* occurrence,
  since today's hasn't happened yet — a digest can never cover a period
  that hasn't completed yet.
"""

from __future__ import annotations

from research_assistant.models.digest import DigestType
from research_assistant.models.enums import ExecutionMode
from research_assistant.pipeline.runner import PipelineRunner, PipelineRunResult

# Phase 13E: the pure period-math moved to `scheduling.period` (no
# pipeline/retrieval/agents/LLM dependency chain), so lightweight callers
# like `ui.dashboard_data` can compute the exact operational digest period
# without pulling in this module's much heavier dependencies. Re-exported
# here unchanged so every existing `from research_assistant.scheduling.jobs
# import compute_digest_period` (etc.) import keeps working verbatim.
from research_assistant.scheduling.period import (  # noqa: F401
    _most_recent_occurrence,
    _preceding_occurrence,
    compute_current_period,
    compute_digest_period,
    compute_manual_run_period,
)


def run_manual_digest(runner: PipelineRunner) -> PipelineRunResult:
    """Manual "Run Now" — today's business-day start through the current
    instant. Uses `DigestType.MORNING` as the digest label (the manual
    window is anchored to the same business-day-start hour the morning
    digest uses); this is a CLI/API entry point only -- no scheduling or
    email delivery is wired to it. Always `execution_mode=MANUAL` -- a
    manual run can never masquerade as a scheduler-triggered one."""
    now = runner.services.clock.now()
    period_start, period_end = compute_manual_run_period(now, runner.services.config)
    return runner.run(
        digest_type=DigestType.MORNING, period_start=period_start, period_end=period_end,
        execution_mode=ExecutionMode.MANUAL,
    )


def run_morning_digest(runner: PipelineRunner, *, execution_mode: ExecutionMode = ExecutionMode.MANUAL) -> PipelineRunResult:
    """Covers the prior evening-digest cutoff through the morning-digest
    cutoff. `execution_mode` defaults to MANUAL (preserving the existing
    CLI's behavior, `python -m research_assistant.scheduling.cli
    --digest-type morning`) -- only `ResearchScheduler`'s own cron callback
    passes `execution_mode=ExecutionMode.SCHEDULED`."""
    now = runner.services.clock.now()
    period_start, period_end = compute_digest_period(DigestType.MORNING, now, runner.services.config)
    return runner.run(
        digest_type=DigestType.MORNING, period_start=period_start, period_end=period_end,
        execution_mode=execution_mode,
    )


def run_evening_digest(runner: PipelineRunner, *, execution_mode: ExecutionMode = ExecutionMode.MANUAL) -> PipelineRunResult:
    """Covers the morning-digest cutoff through the evening-digest cutoff.
    See `run_morning_digest` for `execution_mode` semantics."""
    now = runner.services.clock.now()
    period_start, period_end = compute_digest_period(DigestType.EVENING, now, runner.services.config)
    return runner.run(
        digest_type=DigestType.EVENING, period_start=period_start, period_end=period_end,
        execution_mode=execution_mode,
    )


def run_current_digest(runner: PipelineRunner, *, execution_mode: ExecutionMode = ExecutionMode.MANUAL) -> PipelineRunResult:
    """LIVE CURRENT mode (final production gate extension): a manually
    triggered, rolling scan ending at the exact instant invoked --
    `[now - current_window_hours, now)`, width controlled by
    `config.settings.research.current_window_hours` (default 48; management
    requirement, widened from the original fixed 24) -- computed from
    `compute_current_period`, which is deliberately independent of the
    08:00/20:00 scheduler anchors. This executes immediately against
    `runner`; it never waits for APScheduler, the next scheduled run, or an
    existing digest. `execution_mode` defaults to MANUAL, exactly like every
    other job in this module except `ResearchScheduler`'s own cron callback
    -- CURRENT has no scheduled variant at all, so in practice it is always
    MANUAL."""
    now = runner.services.clock.now()
    window_hours = runner.services.config.settings.research.current_window_hours
    period_start, period_end = compute_current_period(now, window_hours)
    return runner.run(
        digest_type=DigestType.CURRENT, period_start=period_start, period_end=period_end,
        execution_mode=execution_mode,
    )
