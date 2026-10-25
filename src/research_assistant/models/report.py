"""The canonical Report model — Phase 6's structured, manager-ready output.

Every factual list field (`companies`, `people`, `products`, `technologies`,
`partnerships`, `deal_value`, `geography`) and every `sources` entry is
populated deterministically by Python code from the underlying `Event`/
`ResearchItem`s — never invented or re-derived by the LLM. The LLM only
ever writes the four narrative prose fields (`opening_context`, the
`details.narrative`, `strategic_implications`, `nomura_opportunity`),
grounded in those same deterministic facts. This is what keeps
`Report -> Event -> ResearchItem -> Evidence -> Article -> source URL`
traceable instead of collapsing into an untraceable text blob.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from research_assistant.models.enums import SourcePriority, SourceType


class ReportSource(BaseModel):
    publisher: Optional[str] = None
    title: Optional[str] = None
    publication_date: Optional[datetime] = None
    url: str
    source_type: SourceType
    source_priority: SourcePriority

    @field_validator("publication_date")
    @classmethod
    def _require_timezone_aware(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is not None and value.tzinfo is None:
            raise ValueError("publication_date must be timezone-aware")
        return value


class ReportDetails(BaseModel):
    """Deterministic facts (Python-aggregated from the Event's ResearchItems)
    plus the LLM-written narrative grounded in exactly those facts."""

    companies: List[str] = Field(default_factory=list)
    people: List[str] = Field(default_factory=list)
    products: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    partnerships: List[str] = Field(default_factory=list)
    deal_value: Optional[str] = None
    geography: List[str] = Field(default_factory=list)
    narrative: str = Field(min_length=1)
    key_insights: List[str] = Field(default_factory=list)
    """Bounded, sanitized takeaways (2026-10-15 quality fix). Defaults to
    `[]` so every Report row persisted before this field existed still
    deserializes correctly (`ReportDetails.model_validate` on old
    `details_json` simply fills this in as empty) -- the renderer omits
    the section entirely when it's empty, never fabricating one."""


class Report(BaseModel):
    event_id: str
    title: str
    opening_context: str = Field(min_length=1)
    details: ReportDetails
    strategic_implications: str = Field(min_length=1)
    nomura_opportunity: str = Field(min_length=1)
    sources: List[ReportSource] = Field(default_factory=list)
    relevance_score: float
    confidence_score: float


class ReportFailure(BaseModel):
    """Returned instead of a Report when the LLM call fails — never a
    fabricated report. The underlying Event is preserved (by `event_id`) so
    the orchestrator/caller can retry or surface the gap explicitly rather
    than silently losing the event."""

    event_id: str
    reason: str
    detail: str


class TierBReport(BaseModel):
    """Structured report for ONE Tier B article -- no Event, no
    ResearchItems exist for it (that is definitionally what makes it Tier
    B), so there is no deterministically-aggregated entity/partnership
    fact list to carry the way `ReportDetails` does for Tier A. Only the
    narrative sections the model actually wrote (sanitized the same way
    Tier A's are) plus deterministic, Python-built provenance
    (`source`, reused `ReportSource`) are ever included -- nothing here is
    invented. `content_partial` is a deterministic flag (never an LLM
    judgment) set by the resolver when the source article's own
    `extraction_status` was not a full `SUCCESS` -- it tells the reader the
    underlying text may itself be incomplete, never a fabricated caveat."""

    article_id: str
    title: Optional[str] = None
    opening_context: str = Field(min_length=1)
    key_insights: List[str] = Field(min_length=1)
    strategic_implications: str = Field(min_length=1)
    nomura_opportunity: str = Field(min_length=1)
    source: ReportSource
    content_partial: bool = False


class TierBReportFailure(BaseModel):
    """Returned instead of a `TierBReport` when generation fails (model
    error, timeout, or invalid structured output) -- never a fabricated
    report. The underlying article selection is preserved by the caller
    (the resolver never removes a user's newsletter selection on failure)."""

    article_id: str
    reason: str
    detail: str
