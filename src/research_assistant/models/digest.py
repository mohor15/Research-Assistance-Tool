"""The canonical Digest model — a structured collection of ranked Events for
one scheduled run. Scheduling itself (Phase 8) is out of scope here; this
model just represents the result of one digest-building call.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class DigestType(str, Enum):
    MORNING = "morning"
    EVENING = "evening"
    CURRENT = "current"
    """A manually-triggered, rolling digest (`[T-window, T)`, window width from
    `config.settings.research.current_window_hours`, default 48), independent
    of the 08:00/20:00 scheduler anchors -- see `scheduling.period.compute_current_period`."""


class DigestTheme(BaseModel):
    """A recurring pattern across the digest's events.

    `label` is drawn from config/topics.yaml's `strategic_themes` — a
    closed, configured vocabulary — never invented by the LLM. `event_ids`
    is the deterministic (keyword-matched) set of events that mention the
    theme; `description` may optionally be LLM-synthesized prose *about*
    that already-determined set, never a redefinition of the set itself.
    """

    label: str
    description: Optional[str] = None
    event_ids: List[str] = Field(default_factory=list)


class Digest(BaseModel):
    id: str
    digest_type: DigestType
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    events: List[str] = Field(default_factory=list)
    """Event ids included in this digest, in ranked order."""
    themes: List[DigestTheme] = Field(default_factory=list)

    @field_validator("generated_at", "period_start", "period_end")
    @classmethod
    def _require_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("digest datetime fields must be timezone-aware")
        return value
