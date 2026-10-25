"""Digest-period semantics (bug fix): a digest's period must be anchored to
the configured morning/evening hours in the configured IANA timezone, never
to "the last 12 hours from whenever the job happened to run". This was the
root cause of a manual morning run executed at ~20:32 IST producing period
03:02..15:02 (a rolling window ending at invocation time) instead of the
intended 20:00-yesterday -> 08:00-today window.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from research_assistant.config import get_config
from research_assistant.models.digest import DigestType
from research_assistant.scheduling.jobs import compute_digest_period, compute_manual_run_period

IST = ZoneInfo("Asia/Kolkata")


def _ist(year, month, day, hour, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=IST)


def _config():
    # Real config: system.timezone=Asia/Kolkata, research.morning_digest_hour=8,
    # research.evening_digest_hour=20 (config/settings.yaml, unmodified).
    return get_config()


def test_morning_run_at_the_scheduled_morning_time():
    now = _ist(2026, 3, 5, 8, 0).astimezone(timezone.utc)
    start, end = compute_digest_period(DigestType.MORNING, now, _config())

    assert end == _ist(2026, 3, 5, 8, 0).astimezone(timezone.utc)
    assert start == _ist(2026, 3, 4, 20, 0).astimezone(timezone.utc)


def test_evening_run_at_the_scheduled_evening_time():
    now = _ist(2026, 3, 5, 20, 0).astimezone(timezone.utc)
    start, end = compute_digest_period(DigestType.EVENING, now, _config())

    assert end == _ist(2026, 3, 5, 20, 0).astimezone(timezone.utc)
    assert start == _ist(2026, 3, 5, 8, 0).astimezone(timezone.utc)


def test_phase_13c_october_1_2026_morning_and_evening_windows_do_not_expand_to_execution_time():
    """Phase 13C Part 9 verification, pinned to the task's own literal date
    and execution times: on 2026-10-01, a Morning Digest run at 13:09 IST or
    14:43 IST (both well after the 08:00 window closed) must still report
    exactly 2026-09-30 20:00 IST -> 2026-10-01 08:00 IST, never expanding to
    the execution time; an Evening Digest run after its own 20:00 anchor has
    passed must report exactly 2026-10-01 08:00 IST -> 2026-10-01 20:00 IST."""
    expected_morning_start = _ist(2026, 9, 30, 20, 0).astimezone(timezone.utc)
    expected_morning_end = _ist(2026, 10, 1, 8, 0).astimezone(timezone.utc)

    for execution_time in (_ist(2026, 10, 1, 13, 9), _ist(2026, 10, 1, 14, 43)):
        start, end = compute_digest_period(DigestType.MORNING, execution_time.astimezone(timezone.utc), _config())
        assert start == expected_morning_start
        assert end == expected_morning_end
        assert end != execution_time.astimezone(timezone.utc)

    expected_evening_start = _ist(2026, 10, 1, 8, 0).astimezone(timezone.utc)
    expected_evening_end = _ist(2026, 10, 1, 20, 0).astimezone(timezone.utc)
    for execution_time in (_ist(2026, 10, 1, 20, 30), _ist(2026, 10, 1, 23, 59), _ist(2026, 10, 2, 7, 59)):
        start, end = compute_digest_period(DigestType.EVENING, execution_time.astimezone(timezone.utc), _config())
        assert start == expected_evening_start
        assert end == expected_evening_end


def test_manual_morning_execution_later_in_the_day_still_reports_the_correct_window():
    """The reported bug, reproduced exactly: a manual morning run executed at
    ~20:32 IST must still report yesterday-20:00 -> today-08:00, not a
    rolling 12-hour window ending at 20:32."""
    now = _ist(2026, 3, 5, 20, 32).astimezone(timezone.utc)
    start, end = compute_digest_period(DigestType.MORNING, now, _config())

    assert end == _ist(2026, 3, 5, 8, 0).astimezone(timezone.utc)
    assert start == _ist(2026, 3, 4, 20, 0).astimezone(timezone.utc)
    assert end != now
    assert (end - start) == timedelta(hours=12)


def test_manual_evening_execution_later_in_the_day_still_reports_the_correct_window():
    now = _ist(2026, 3, 5, 23, 45).astimezone(timezone.utc)
    start, end = compute_digest_period(DigestType.EVENING, now, _config())

    assert end == _ist(2026, 3, 5, 20, 0).astimezone(timezone.utc)
    assert start == _ist(2026, 3, 5, 8, 0).astimezone(timezone.utc)
    assert end != now


def test_manual_morning_execution_earlier_in_the_day_reaches_back_to_yesterday():
    """A digest can never cover a period that hasn't completed yet: run at
    03:00 IST (before today's 08:00 morning cutoff) must resolve to
    yesterday's already-completed morning window, not today's future one."""
    now = _ist(2026, 3, 5, 3, 0).astimezone(timezone.utc)
    start, end = compute_digest_period(DigestType.MORNING, now, _config())

    assert end == _ist(2026, 3, 4, 8, 0).astimezone(timezone.utc)
    assert start == _ist(2026, 3, 3, 20, 0).astimezone(timezone.utc)
    assert end < now


def test_midnight_date_boundary_for_evening_digest():
    """Run just after midnight (00:30 IST) asking for the evening digest:
    both today's 20:00 and today's 08:00 are still in the future, so both
    bounds must fall on the previous calendar day."""
    now = _ist(2026, 3, 5, 0, 30).astimezone(timezone.utc)
    start, end = compute_digest_period(DigestType.EVENING, now, _config())

    assert end == _ist(2026, 3, 4, 20, 0).astimezone(timezone.utc)
    assert start == _ist(2026, 3, 4, 8, 0).astimezone(timezone.utc)


def test_timezone_conversion_is_correct_for_the_half_hour_ist_offset():
    """Asia/Kolkata is UTC+5:30 (a half-hour offset) -- confirms the
    conversion is done via real timezone arithmetic, not a hardcoded whole
    number of hours."""
    now = _ist(2026, 3, 5, 8, 0).astimezone(timezone.utc)
    start, end = compute_digest_period(DigestType.MORNING, now, _config())

    assert end == datetime(2026, 3, 5, 2, 30, tzinfo=timezone.utc)
    assert start == datetime(2026, 3, 4, 14, 30, tzinfo=timezone.utc)
    assert end.tzinfo is not None
    assert start.tzinfo is not None


def test_repeated_manual_runs_within_the_same_window_are_idempotent():
    """Two manual "morning" runs on the same calendar day, at different
    times, must resolve to the identical period (and therefore the
    identical deterministic run_id via generate_run_id) -- this is what
    keeps a re-run idempotent rather than creating a second run record."""
    config = _config()
    early = _ist(2026, 3, 5, 9, 0).astimezone(timezone.utc)
    late = _ist(2026, 3, 5, 18, 0).astimezone(timezone.utc)

    period_a = compute_digest_period(DigestType.MORNING, early, config)
    period_b = compute_digest_period(DigestType.MORNING, late, config)

    assert period_a == period_b


def test_evening_and_morning_periods_are_non_overlapping_and_adjacent():
    morning_start, morning_end = compute_digest_period(
        DigestType.MORNING, _ist(2026, 3, 5, 8, 0).astimezone(timezone.utc), _config()
    )
    evening_start, evening_end = compute_digest_period(
        DigestType.EVENING, _ist(2026, 3, 5, 20, 0).astimezone(timezone.utc), _config()
    )

    assert morning_end == evening_start  # today's 08:00 is both boundaries
    assert morning_start < morning_end <= evening_start < evening_end


# ---------------------------------------------------------------------------
# Manual "Run Now" period (Phase 1): period_start = today's business-day
# start (morning_digest_hour), period_end = current instant. NOT a rolling
# lookback window.
# ---------------------------------------------------------------------------


def test_manual_run_at_2130_covers_0800_to_2130():
    now = _ist(2026, 3, 5, 21, 30).astimezone(timezone.utc)
    start, end = compute_manual_run_period(now, _config())

    assert start == _ist(2026, 3, 5, 8, 0).astimezone(timezone.utc)
    assert end == now


def test_manual_run_at_1400_covers_0800_to_1400():
    now = _ist(2026, 3, 5, 14, 0).astimezone(timezone.utc)
    start, end = compute_manual_run_period(now, _config())

    assert start == _ist(2026, 3, 5, 8, 0).astimezone(timezone.utc)
    assert end == now


def test_manual_run_at_0930_covers_0800_to_0930():
    now = _ist(2026, 3, 5, 9, 30).astimezone(timezone.utc)
    start, end = compute_manual_run_period(now, _config())

    assert start == _ist(2026, 3, 5, 8, 0).astimezone(timezone.utc)
    assert end == now


def test_manual_run_period_is_not_a_rolling_lookback_window():
    """A manual run late in the day must NOT produce a 12-hour (or any
    fixed-width) rolling window -- the window widens as the day goes on,
    always anchored to the fixed 08:00 start."""
    early = compute_manual_run_period(_ist(2026, 3, 5, 9, 0).astimezone(timezone.utc), _config())
    late = compute_manual_run_period(_ist(2026, 3, 5, 21, 0).astimezone(timezone.utc), _config())

    assert early[0] == late[0] == _ist(2026, 3, 5, 8, 0).astimezone(timezone.utc)  # same start
    assert (late[1] - late[0]) > (early[1] - early[0])  # window widened, not fixed-width


def test_manual_run_before_0800_reaches_back_to_yesterdays_start():
    """A digest can never cover a period that hasn't completed yet: a
    manual run at 03:00 IST (before today's 08:00) anchors to yesterday's
    08:00, not today's future one."""
    now = _ist(2026, 3, 5, 3, 0).astimezone(timezone.utc)
    start, end = compute_manual_run_period(now, _config())

    assert start == _ist(2026, 3, 4, 8, 0).astimezone(timezone.utc)
    assert end == now


def test_manual_run_at_exactly_0800_covers_a_zero_width_window_at_the_boundary():
    now = _ist(2026, 3, 5, 8, 0).astimezone(timezone.utc)
    start, end = compute_manual_run_period(now, _config())

    assert start == end == now
