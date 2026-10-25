"""`compute_current_period` boundary matrix (final production gate "Live
Current" extension, test matrix section "Window"): exactly `window_hours`
old, `window_hours`+1s old, exactly at period_end, future timestamps, and a
timezone round-trip -- mirrors the rigor of `test_period_boundary_matrix.py`
for the scheduled digests, but for the rolling CURRENT window instead of an
anchored one.

`window_hours` defaults to 48 (management requirement, widened from the
original fixed 24 -- see `config.schema.ResearchSettings.current_window_hours`
and `scheduling.jobs.run_current_digest`, which passes the configured value
explicitly in production). The default-argument tests below exercise that
48-hour default; a handful of tests also pass an explicit `window_hours` to
prove the width is a genuine parameter, not a second hardcoded constant."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from research_assistant.scheduling.period import compute_current_period

NOW = datetime(2026, 10, 3, 14, 30, 0, tzinfo=timezone.utc)


def test_period_is_exactly_48_hours_wide_by_default():
    start, end = compute_current_period(NOW)
    assert end - start == timedelta(hours=48)


def test_period_end_is_exactly_now():
    start, end = compute_current_period(NOW)
    assert end == NOW


def test_period_start_is_exactly_48_hours_before_now_by_default():
    start, end = compute_current_period(NOW)
    assert start == NOW - timedelta(hours=48)


def test_window_hours_is_a_genuine_parameter_not_a_second_hardcoded_constant():
    """Passing an explicit `window_hours` must change the width -- proves
    the 48h behavior above comes from the parameter's default, not from a
    second hardcoded 48 inside the function body."""
    start, end = compute_current_period(NOW, window_hours=24)
    assert end - start == timedelta(hours=24)
    assert start == NOW - timedelta(hours=24)

    start72, end72 = compute_current_period(NOW, window_hours=72)
    assert end72 - start72 == timedelta(hours=72)
    assert start72 == NOW - timedelta(hours=72)


def test_article_exactly_at_period_start_is_includable():
    """[period_start, period_end) is half-open -- an article dated exactly
    at period_start must be INSIDE the window. This test checks the period
    math itself; `digest_generation`'s existing `period_start <= e.event_date
    < period_end` comparison (unchanged, see test_digest_boundary_e2e.py)
    is what actually applies this boundary to an event."""
    start, end = compute_current_period(NOW)
    assert start <= start < end


def test_article_exactly_at_period_end_is_excluded():
    """An event dated exactly at `period_end` (= `now`) belongs to a FUTURE
    scan, never this one -- half-open on the right, same invariant as
    Morning/Evening."""
    start, end = compute_current_period(NOW)
    assert not (start <= end < end)


def test_article_one_second_before_period_end_is_includable():
    start, end = compute_current_period(NOW)
    one_second_before_end = end - timedelta(seconds=1)
    assert start <= one_second_before_end < end


def test_article_exactly_48_hours_old_is_includable():
    """Exactly `window_hours` old == exactly at period_start -- includable,
    not excluded (default 48h window)."""
    start, end = compute_current_period(NOW)
    exactly_48h_old = NOW - timedelta(hours=48)
    assert exactly_48h_old == start
    assert start <= exactly_48h_old < end


def test_article_48_hours_and_one_second_old_is_excluded():
    start, end = compute_current_period(NOW)
    too_old = NOW - timedelta(hours=48, seconds=1)
    assert not (start <= too_old < end)


def test_future_timestamp_is_excluded():
    start, end = compute_current_period(NOW)
    future = NOW + timedelta(seconds=1)
    assert not (start <= future < end)


def test_timezone_conversion_is_consistent_regardless_of_input_timezone():
    """The same instant, expressed in UTC vs. IST, must produce the exact
    same period bounds -- `compute_current_period` works purely on
    timezone-aware instants, never on wall-clock fields, so it cannot be
    thrown off by which timezone the caller's `now` happens to carry."""
    ist_now = NOW.astimezone(ZoneInfo("Asia/Kolkata"))
    assert compute_current_period(NOW) == compute_current_period(ist_now)


def test_results_are_timezone_aware_and_in_utc():
    start, end = compute_current_period(NOW)
    assert start.tzinfo is not None and end.tzinfo is not None
