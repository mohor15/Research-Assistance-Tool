"""Pure digest-period math (Phase 13E extraction from `scheduling.jobs`) --
no dependency on `pipeline.runner` or anything that pulls in retrieval/
agents/LLM, so lightweight callers (notably `ui.dashboard_data`, which
structurally must never import retrieval/agents/LLM) can compute the exact
operational digest period without that heavy dependency chain.

`scheduling.jobs` re-exports everything from here unchanged, so existing
`from research_assistant.scheduling.jobs import compute_digest_period`
imports keep working exactly as before -- this is a pure refactor, not a
behavior change.

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

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from research_assistant.config.schema import AppConfig
from research_assistant.models.digest import DigestType


def _most_recent_occurrence(now_local: datetime, hour: int) -> datetime:
    """The latest wall-clock instant at `hour`:00 that is <= `now_local`."""
    candidate = now_local.replace(hour=hour, minute=0, second=0, microsecond=0)
    if candidate > now_local:
        candidate -= timedelta(days=1)
    return candidate


def _preceding_occurrence(period_end_local: datetime, hour: int) -> datetime:
    """The latest wall-clock instant at `hour`:00 strictly before `period_end_local`."""
    candidate = period_end_local.replace(hour=hour, minute=0, second=0, microsecond=0)
    if candidate >= period_end_local:
        candidate -= timedelta(days=1)
    return candidate


def compute_digest_period(
    digest_type: DigestType, now_utc: datetime, config: AppConfig
) -> tuple[datetime, datetime]:
    """Deterministic digest-period bounds (both timezone-aware, in UTC),
    anchored to `config.settings.research.morning_digest_hour` /
    `.evening_digest_hour` in `config.settings.system.timezone`. See the
    module docstring for the exact semantics."""
    tz = ZoneInfo(config.settings.system.timezone)
    now_local = now_utc.astimezone(tz)
    research = config.settings.research

    if digest_type == DigestType.MORNING:
        period_end_local = _most_recent_occurrence(now_local, research.morning_digest_hour)
        period_start_local = _preceding_occurrence(period_end_local, research.evening_digest_hour)
    else:
        period_end_local = _most_recent_occurrence(now_local, research.evening_digest_hour)
        period_start_local = _preceding_occurrence(period_end_local, research.morning_digest_hour)

    return period_start_local.astimezone(timezone.utc), period_end_local.astimezone(timezone.utc)


def compute_manual_run_period(now_utc: datetime, config: AppConfig) -> tuple[datetime, datetime]:
    """Manual "Run Now" period (Phase 1): from the most recently completed
    `morning_digest_hour` (today's business-day start, or yesterday's if
    that hasn't happened yet) through the current instant.

    This is deliberately NOT a rolling lookback window ("the last N hours")
    -- `period_start` is always anchored to the configured business-day
    start hour, exactly like `compute_digest_period`'s anchoring, just with
    `period_end` fixed at "now" instead of another configured hour. Examples
    (Asia/Kolkata, morning_digest_hour=8):
      * run at 21:30 IST -> 08:00 IST today .. 21:30 IST today
      * run at 14:00 IST -> 08:00 IST today .. 14:00 IST today
      * run at 09:30 IST -> 08:00 IST today .. 09:30 IST today
      * run at 03:00 IST (before today's 08:00) -> 08:00 IST yesterday .. 03:00 IST today
    """
    tz = ZoneInfo(config.settings.system.timezone)
    now_local = now_utc.astimezone(tz)
    period_start_local = _most_recent_occurrence(now_local, config.settings.research.morning_digest_hour)
    return period_start_local.astimezone(timezone.utc), now_utc


def compute_current_period(now_utc: datetime, window_hours: int = 48) -> tuple[datetime, datetime]:
    """CURRENT mode's period (final production gate, live-current extension):
    a pure rolling window ending at the exact trigger instant --
    `[now_utc - window_hours, now_utc)`. Deliberately NOT anchored to
    `morning_digest_hour`/`evening_digest_hour` like `compute_digest_period`/
    `compute_manual_run_period` -- CURRENT is "the last N hours, right now",
    not "since the last business-day boundary", and must never wait for or
    be affected by the scheduler's own anchor hours.

    `window_hours` defaults to 48 (management requirement, widened from the
    original 24) but callers that have a config available should pass
    `config.settings.research.current_window_hours` explicitly (see
    `scheduling.jobs.run_current_digest`) so the configured value is what
    actually governs production runs -- this default only covers callers
    (tests, ad hoc scripts) with no config in hand."""
    return now_utc - timedelta(hours=window_hours), now_utc
