"""The canonical Article model — Phase 2's core deliverable.

An `Article` is the deterministic, source-grounded representation of one
retrieved publication/page, produced by
`research_assistant.evidence.normalize_article` from Phase 1's
`ExtractedArticle` + `RetrievedPage` + `SourceDefinition`. It deliberately
carries no relevance/strategic interpretation, no clustering, and no
score — those belong to later phases. Source quality is represented by
reusing `SourceType`/`SourcePriority` from `models.enums` rather than
introducing a duplicate concept.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.models import ExtractionMethod, ExtractionStatus


class DatePrecision(str, Enum):
    """How much to trust `Article.publication_date`.

    `DATE_ONLY` is a best-effort heuristic (the parsed timestamp lands
    exactly at midnight UTC), since Phase 1's date parsing already collapses
    a bare "2025-09-01" and a genuine midnight timestamp into the same
    `datetime` — the raw source string isn't preserved upstream. This is
    documented, not fabricated: when in doubt, precision degrades toward
    `UNKNOWN` rather than claiming false confidence.
    """

    EXACT = "exact"
    DATE_ONLY = "date_only"
    MONTH = "month"
    """Only the calendar month is known (e.g. recovered from a source's own
    URL structure, such as Bank of America's `/press-releases/YYYY/MM/`
    convention) -- the day-of-month in `publication_date` is a storage
    placeholder (always the 1st), never a real fact. Code that enforces
    exact-period eligibility must treat this the same as `UNKNOWN` for that
    purpose -- see `events/builder.py::_select_event_date` and
    `agents/base.py::_build_research_item`."""
    UNKNOWN = "unknown"


class Article(BaseModel):
    id: str
    """Deterministic id derived from `canonical_url` — refetching the same
    URL always produces the same Article id."""

    url: str
    canonical_url: str
    domain: str
    source_name: str
    source_type: SourceType
    source_priority: SourcePriority

    publisher: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None

    publication_date: Optional[datetime] = None
    """None, never guessed, when the source gave no usable date."""
    publication_date_precision: DatePrecision = DatePrecision.UNKNOWN
    retrieved_at: datetime

    raw_text: Optional[str] = None
    clean_text: Optional[str] = None
    content_hash: Optional[str] = None
    """sha256 of `clean_text`; None when there was no usable content to hash."""

    extraction_method: ExtractionMethod = ExtractionMethod.NONE
    extraction_status: ExtractionStatus = ExtractionStatus.FAILED

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("publication_date", "retrieved_at")
    @classmethod
    def _require_timezone_aware(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is not None and value.tzinfo is None:
            raise ValueError("datetime fields must be timezone-aware")
        return value
