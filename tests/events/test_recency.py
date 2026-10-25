"""Recency scoring: injected reference time only, never `datetime.now()`.
Bucketed decay, boundary conditions, future-dated events, missing dates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from research_assistant.events.clock import FixedClock
from research_assistant.events.scoring import recency_score

REFERENCE = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def test_very_recent_event_scores_maximum():
    clock = FixedClock(REFERENCE)
    assert recency_score(REFERENCE, clock=clock) == 1.0
    assert recency_score(REFERENCE - timedelta(hours=6), clock=clock) == 1.0


def test_older_event_scores_lower():
    clock = FixedClock(REFERENCE)
    assert recency_score(REFERENCE - timedelta(days=10), clock=clock) == 0.5
    assert recency_score(REFERENCE - timedelta(days=20), clock=clock) == 0.3


def test_very_old_event_scores_zero():
    clock = FixedClock(REFERENCE)
    assert recency_score(REFERENCE - timedelta(days=200), clock=clock) == 0.0


@pytest.mark.parametrize(
    "age_days,expected",
    [
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0001, 0.9),
        (3.0, 0.9),
        (3.0001, 0.7),
        (7.0, 0.7),
        (7.0001, 0.5),
        (14.0, 0.5),
        (14.0001, 0.3),
        (30.0, 0.3),
        (30.0001, 0.1),
        (90.0, 0.1),
        (90.0001, 0.0),
    ],
)
def test_bucket_boundaries_are_exact(age_days, expected):
    clock = FixedClock(REFERENCE)
    event_date = REFERENCE - timedelta(days=age_days)
    assert recency_score(event_date, clock=clock) == expected


def test_future_dated_event_is_clamped_to_maximally_recent_not_penalized():
    clock = FixedClock(REFERENCE)
    future_date = REFERENCE + timedelta(days=5)
    assert recency_score(future_date, clock=clock) == 1.0


def test_missing_event_date_scores_zero_not_assumed_recent():
    clock = FixedClock(REFERENCE)
    assert recency_score(None, clock=clock) == 0.0


def test_recency_never_calls_system_clock_directly():
    """Uses an injected FixedClock exclusively — verifies the test itself
    doesn't accidentally depend on wall-clock time by checking two calls
    separated by nothing still agree (would only fail if `now()` drifted)."""
    clock = FixedClock(REFERENCE)
    first = recency_score(REFERENCE - timedelta(days=2), clock=clock)
    second = recency_score(REFERENCE - timedelta(days=2), clock=clock)
    assert first == second == 0.9
