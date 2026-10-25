"""DigestService — a collection of ranked Events -> a structured Digest.

No LLM call: theme descriptions are deterministic templates (see
`reporting.themes`). Scheduling (which run is "morning" vs "evening", and
when it fires) is explicitly Phase 8's job — this service only ever builds
one digest from whatever events/period bounds the caller supplies.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import List, Optional

from research_assistant.config.schema import AppConfig
from research_assistant.events.clock import Clock, SystemClock
from research_assistant.events.models import Event
from research_assistant.models.digest import Digest, DigestType
from research_assistant.reporting.themes import build_theme_candidates

_ID_LENGTH = 16


class DigestService:
    def __init__(self, config: AppConfig, *, clock: Optional[Clock] = None) -> None:
        self._config = config
        self._clock = clock or SystemClock()

    def build_digest(
        self,
        events: List[Event],
        *,
        digest_type: DigestType,
        period_start: datetime,
        period_end: datetime,
    ) -> Digest:
        themes = build_theme_candidates(events, self._config.topics)
        event_ids = [event.id for event in events]
        return Digest(
            id=_generate_digest_id(digest_type, period_start, period_end, event_ids),
            digest_type=digest_type,
            generated_at=self._clock.now(),
            period_start=period_start,
            period_end=period_end,
            events=event_ids,
            themes=themes,
        )


def _generate_digest_id(
    digest_type: DigestType, period_start: datetime, period_end: datetime, event_ids: List[str]
) -> str:
    basis = f"{digest_type.value}:{period_start.isoformat()}:{period_end.isoformat()}:{','.join(sorted(event_ids))}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:_ID_LENGTH]
