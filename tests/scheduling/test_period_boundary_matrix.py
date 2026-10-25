"""Phase 13F ULTIMATUM 1 "TIME-WINDOW TEST MATRIX": exact second-precision
boundary coverage for `compute_digest_period`, at every clock time the task
explicitly lists. Complements `test_period_semantics.py` (which already
covers the whole-hour anchors) with the sub-second boundary cases."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from research_assistant.config import get_config
from research_assistant.models.digest import DigestType
from research_assistant.scheduling.period import compute_digest_period

IST = ZoneInfo("Asia/Kolkata")


def _ist(year, month, day, hour, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=IST)


def _config():
    return get_config()


MORNING_START = _ist(2026, 9, 30, 20, 0, 0)
MORNING_END = _ist(2026, 10, 1, 8, 0, 0)
EVENING_START = MORNING_END
EVENING_END = _ist(2026, 10, 1, 20, 0, 0)


@pytest.mark.parametrize(
    "clock_time",
    [
        _ist(2026, 10, 1, 16, 18, 0),
        _ist(2026, 10, 1, 19, 0, 0),
        _ist(2026, 10, 1, 19, 59, 59),
        _ist(2026, 10, 1, 20, 0, 0),
        _ist(2026, 10, 1, 20, 0, 1),
        _ist(2026, 10, 1, 23, 59, 0),
        _ist(2026, 10, 2, 0, 0, 0),
        _ist(2026, 10, 2, 7, 59, 59),
    ],
)
def test_morning_window_before_the_0800_anchor_resolves_to_the_prior_evening_start(clock_time):
    """Every one of these times falls on/after Oct 1's own 08:00 IST anchor
    but strictly before Oct 2's 08:00 IST anchor -- the most recently
    COMPLETED morning anchor throughout this whole span is Oct 1's own
    08:00, so every one of these clock times (spanning 16:18, the 20:00
    evening boundary on either side, 23:59, midnight, and just before the
    *next* 08:00, per the task's own exact matrix) must resolve to the
    SAME period: Sep 30 20:00 -> Oct 1 08:00 IST."""
    start, end = compute_digest_period(DigestType.MORNING, clock_time.astimezone(timezone.utc), _config())
    assert start == MORNING_START.astimezone(timezone.utc)
    assert end == MORNING_END.astimezone(timezone.utc)


@pytest.mark.parametrize(
    "clock_time",
    [
        _ist(2026, 10, 1, 8, 0, 0),
        _ist(2026, 10, 1, 8, 0, 1),
    ],
)
def test_morning_window_at_or_after_the_0800_anchor_is_the_just_completed_one(clock_time):
    start, end = compute_digest_period(DigestType.MORNING, clock_time.astimezone(timezone.utc), _config())
    assert start == MORNING_START.astimezone(timezone.utc)
    assert end == MORNING_END.astimezone(timezone.utc)


@pytest.mark.parametrize(
    "clock_time",
    [
        _ist(2026, 10, 1, 8, 0, 0),
        _ist(2026, 10, 1, 12, 0, 0),
        _ist(2026, 10, 1, 19, 0, 0),
        _ist(2026, 10, 1, 19, 59, 59),
    ],
)
def test_evening_window_before_the_2000_anchor_reaches_back_to_yesterdays_completed_evening_window(clock_time):
    """Mirrors the already-tested MORNING case
    (`test_manual_morning_execution_earlier_in_the_day_reaches_back_to_yesterday`
    in test_period_semantics.py): a digest can never cover a period that
    hasn't completed yet. At any of these times, TODAY's 08:00->20:00
    evening window is still in progress, so the function must resolve to
    YESTERDAY's already-completed 08:00->20:00 window, never a partial/
    in-progress "today so far" window."""
    start, end = compute_digest_period(DigestType.EVENING, clock_time.astimezone(timezone.utc), _config())
    assert start == _ist(2026, 9, 30, 8, 0, 0).astimezone(timezone.utc)
    assert end == _ist(2026, 9, 30, 20, 0, 0).astimezone(timezone.utc)


@pytest.mark.parametrize(
    "clock_time",
    [
        _ist(2026, 10, 1, 20, 0, 0),
        _ist(2026, 10, 1, 20, 0, 1),
        _ist(2026, 10, 1, 23, 59, 0),
    ],
)
def test_evening_window_at_or_after_the_2000_anchor_is_the_just_completed_one(clock_time):
    start, end = compute_digest_period(DigestType.EVENING, clock_time.astimezone(timezone.utc), _config())
    assert start == EVENING_START.astimezone(timezone.utc)
    assert end == EVENING_END.astimezone(timezone.utc)
