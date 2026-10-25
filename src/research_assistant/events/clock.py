"""An injectable clock, so recency scoring never calls `datetime.now()`
directly and tests get fully deterministic, reproducible results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FixedClock:
    """Always returns the same instant — the only clock unit tests should use."""

    def __init__(self, when: datetime) -> None:
        if when.tzinfo is None:
            raise ValueError("FixedClock requires a timezone-aware datetime")
        self._when = when

    def now(self) -> datetime:
        return self._when
