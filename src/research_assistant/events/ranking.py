"""Deterministic Event ranking.

Sort order (highest first, no subjective "best" label attached anywhere):

1. `relevance_score` — descending
2. `confidence_score` — descending (tie-break)
3. `event_date` — descending, i.e. more recent events sort earlier among
   ties (a missing date sorts as though it were the oldest possible date,
   i.e. last)
4. `id` — ascending lexicographic (final, fully deterministic tie-break)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from research_assistant.events.models import Event

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)
_OLDEST_POSSIBLE = _EPOCH


def _seconds_since_epoch(value: datetime) -> float:
    # Plain datetime subtraction (never `.timestamp()`, which can overflow
    # on some platforms for dates far from 1970) — safe for any tz-aware
    # datetime, including datetime.min.
    return (value - _EPOCH).total_seconds()


def _sort_key(event: Event):
    date_key = event.event_date or _OLDEST_POSSIBLE
    return (-event.relevance_score, -event.confidence_score, -_seconds_since_epoch(date_key), event.id)


def rank_events(events: List[Event]) -> List[Event]:
    return sorted(events, key=_sort_key)
