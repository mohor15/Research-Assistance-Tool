"""The canonical Event model — Phase 5's core deliverable.

An `Event` is one underlying real-world development that may be described
by multiple `ResearchItem`s (in turn each grounded in an `Article`/
`Evidence`). It is deliberately independent of the Report Agent/LangGraph
(Phase 6): it holds scored, structured data, not prose.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from research_assistant.models.enums import EventType, GroundingType, SourcePriority, SourceType


class SourceRef(BaseModel):
    """One cluster member's provenance — always traceable back to its
    ResearchItem/Article and the original source URL."""

    research_item_id: str
    article_id: str
    url: str
    publisher: Optional[str] = None
    source_type: SourceType
    source_priority: SourcePriority


class ClusterSignal(str, Enum):
    """Which signal(s) contributed to a clustering decision — see events.similarity."""

    ENTITY_OVERLAP = "entity_overlap"
    EVENT_TYPE_MATCH = "event_type_match"
    DATE_PROXIMITY = "date_proximity"
    TITLE_SIMILARITY = "title_similarity"
    CONTENT_SIMILARITY = "content_similarity"


class PairwiseMatch(BaseModel):
    """Why two research items were grouped together — the clustering
    algorithm's explainability record for one pair."""

    research_item_id_a: str
    research_item_id_b: str
    combined_score: float
    matched_signals: List[ClusterSignal] = Field(default_factory=list)


class ClusterMetadata(BaseModel):
    member_count: int
    matches: List[PairwiseMatch] = Field(default_factory=list)
    algorithm_version: str = "v1"


class RelevanceBreakdown(BaseModel):
    """Mirrors config/settings.yaml's ranking.relevance keys exactly."""

    ai_relevance: float
    financial_services_relevance: float
    strategic_significance: float
    recency: float
    novelty: float
    weighted_total: float


class ConfidenceBreakdown(BaseModel):
    """Mirrors config/settings.yaml's ranking.confidence keys exactly."""

    primary_source_confirmation: float
    multiple_source_corroboration: float
    publication_date_certainty: float
    entity_extraction_confidence: float
    event_classification_confidence: float
    content_completeness: float
    weighted_total: float


class EventScoringBreakdown(BaseModel):
    relevance: RelevanceBreakdown
    confidence: ConfidenceBreakdown


class Event(BaseModel):
    id: str
    """Deterministic — derived from the sorted set of member research_item ids."""

    canonical_title: str
    event_type: Optional[EventType] = None
    primary_company: str
    related_companies: List[str] = Field(default_factory=list)
    event_date: Optional[datetime] = None
    description: Optional[str] = None

    research_items: List[str] = Field(default_factory=list)
    articles: List[str] = Field(default_factory=list)

    primary_source: SourceRef
    secondary_sources: List[SourceRef] = Field(default_factory=list)

    relevance_score: float
    confidence_score: float
    scoring_breakdown: EventScoringBreakdown
    cluster_metadata: ClusterMetadata

    grounding_type: GroundingType = GroundingType.CONTENT
    """CONTENT if ANY member ResearchItem is content-grounded (real
    evidence backs at least part of this event), else LINK (every member
    is metadata-only) -- see `events.builder._select_grounding_type`."""

    run_id: Optional[str] = None
    """Explicit run ownership: which `PipelineRunResult.run_id` formed this
    event (stamped once, in `pipeline.nodes.event_scoring`, from
    `PipelineState.run_id`). `None` for every pre-existing row and for any
    event built outside a real pipeline run (e.g. a unit test). This --
    never a mutable Article/Evidence timestamp -- is the authoritative
    answer to "which run produced this event?", and is what Scan
    Explorer's admin diagnostics must use instead of reconstructing
    membership from `Article.retrieved_at`."""

    @field_validator("event_date")
    @classmethod
    def _require_timezone_aware(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is not None and value.tzinfo is None:
            raise ValueError("event_date must be timezone-aware")
        return value
