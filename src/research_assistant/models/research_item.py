"""The canonical ResearchItem model — Phase 3's structured agent output.

Deliberately separates three kinds of information:

- **Deterministic/provenance fields** (`article_id`, `publisher`, `url`,
  `publication_date`, `company`, `evidence`, `source_quality`) — copied
  directly from the input `Article`/`Evidence`, never produced by the LLM.
- **Agent interpretation** (`relevance`, `category`, `summary`,
  `financial_services_relevance`, `nomura_relevance`,
  `strategic_significance`) — the agent's qualitative judgment, explicitly
  NOT the Phase 5 deterministic weighted score.
- **Extracted facts** (`event_type`, `entities`, `people`, `products`,
  `deal_value`, `partnerships`, `technologies`, `geography`) — the agent's
  best-effort extraction from evidence text; may be empty/None rather than
  guessed.

No `event_id` or clustering fields exist here — that is Phase 5's job.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from research_assistant.models.enums import AgentName, EntityType, EventType, GroundingType, SourcePriority, SourceType


class RelevanceLabel(str, Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    AMBIGUOUS = "ambiguous"


class RelevanceDecision(BaseModel):
    label: RelevanceLabel
    reason: str = Field(min_length=1, description="Brief agent explanation for the relevance label.")


class RelevanceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RelevanceAssessment(BaseModel):
    """A qualitative agent judgment — NOT the Phase 5 deterministic weighted score."""

    level: RelevanceLevel
    rationale: str = Field(min_length=1)


class TopicCategory(str, Enum):
    """Mirrors config/topics.yaml's ai_topics buckets plus a financial-services
    and catch-all bucket, so the agent classifies into a closed vocabulary
    instead of free text."""

    CORE_AI = "core_ai"
    MODELS = "models"
    INFRASTRUCTURE = "infrastructure"
    ENTERPRISE_AI = "enterprise_ai"
    GOVERNANCE = "governance"
    FINANCIAL_SERVICES = "financial_services"
    OTHER = "other"


class TypedEntity(BaseModel):
    """One extracted entity name plus its closed-vocabulary type (Phase
    13B) -- a bare `List[str]` of entity names can't distinguish NVIDIA
    (company) from "Agentic AI" (technology) or "Claude.ai" (product), so
    `events.builder._collect_related_companies` had no reliable way to know
    which extracted names were actually companies. `type` is always the
    agent's own judgment from the evidence text, never guessed from the
    name alone."""

    name: str = Field(min_length=1)
    type: EntityType


class SourceQuality(BaseModel):
    """Deterministic — copied from the source Article, never decided by the LLM."""

    source_type: SourceType
    source_priority: SourcePriority


class ReportedFinding(BaseModel):
    """One finding/claim as reported by a consulting publication.

    `is_quantitative` flags a statement containing a number, percentage,
    survey result, market-size figure, or forecast — the agent reports it
    as *reported evidence*, never independently validates or reinterprets
    it as established fact (Phase 4's AnalystAgent).
    """

    statement: str = Field(min_length=1)
    is_quantitative: bool = False


class AnalystFindings(BaseModel):
    """Analyst-specific structured extraction — optional, populated only by
    AnalystAgent, so PeerAgent/AIProductAgent/StartupAgent output is
    unaffected. Historical/contextual claims may only appear here when
    explicitly present in the evidence; nothing is invented from general
    knowledge, and no field here is a prediction of the agent's own."""

    thesis: Optional[str] = None
    key_findings: List[str] = Field(default_factory=list)
    evidence_points: List[ReportedFinding] = Field(default_factory=list)
    industry_pattern: Optional[str] = None
    # Merged `financial_services_implication` + `capital_markets_implication`
    # into one field (CPU-only qwen2.5:3b deep-analysis cost reduction): the
    # two were never read independently anywhere downstream (stored only,
    # inside the nullable `analyst_findings_json` blob column -- no fixed DB
    # columns depend on either name), and both ask the same question
    # ("what does the evidence itself say about FS/capital-markets
    # relevance"), so asking for them as two separate free-text paragraphs
    # was pure extra generation cost for qwen2.5:3b with no distinct
    # downstream consumer for either. A JSON blob loaded from a row written
    # before this change simply has no `financial_implication` key, which
    # `Optional[str] = None` already handles.
    financial_implication: Optional[str] = None
    strategic_themes: List[str] = Field(default_factory=list)


class ResearchItem(BaseModel):
    id: str
    agent: AgentName
    article_id: str
    publisher: Optional[str] = None
    title: Optional[str] = None
    url: str
    publication_date: Optional[datetime] = None
    company: str

    relevance: RelevanceDecision
    category: TopicCategory
    event_type: Optional[EventType] = None
    summary: Optional[str] = None

    is_dated_development: bool
    """Phase 13B: True only when the evidence itself establishes a specific,
    dated action/announcement (not merely that today's retrieval happened
    to find this content) -- an evergreen/reference page (e.g. a static
    product landing page) must be False even when it is otherwise
    `relevance.label == RELEVANT`. Drives whether `article.publication_date`
    is trusted as this item's contribution to an Event's `event_date`
    (see `events.builder._select_event_date`) -- never silently defaulted."""
    development_date_rationale: Optional[str] = None

    entities: List[TypedEntity] = Field(default_factory=list)
    people: List[str] = Field(default_factory=list)
    products: List[str] = Field(default_factory=list)
    deal_value: Optional[str] = None
    partnerships: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    geography: List[str] = Field(default_factory=list)

    financial_services_relevance: RelevanceAssessment
    nomura_relevance: RelevanceAssessment
    strategic_significance: RelevanceAssessment

    evidence: List[str] = Field(default_factory=list, description="Evidence ids this item is grounded in.")
    source_quality: SourceQuality

    analyst_findings: Optional[AnalystFindings] = None
    """Populated only by AnalystAgent (Phase 4); None for every other agent."""

    grounding_type: GroundingType = GroundingType.CONTENT
    """CONTENT (default -- every pre-existing caller/test is unaffected) or
    LINK (see `agents.link_grounded`). Never implies `is_dated_development`
    one way or the other -- a LINK item can be a dated development when its
    publication date is reliable, exactly like a CONTENT item."""

    run_id: Optional[str] = None
    """Explicit run ownership: which `PipelineRunResult.run_id` produced
    this item (stamped once, in `pipeline.nodes.agent_analysis`, from
    `PipelineState.run_id`). `None` for every pre-existing row (predates
    this field) and for any caller that builds a ResearchItem outside a
    real pipeline run (e.g. a unit test). Never inferred from a mutable
    timestamp like `retrieved_at` -- this is the one authoritative answer
    to "which run produced this?"."""

    relevance_override: Optional[str] = None
    """Set only when the deterministic scope safeguard in `agents.base`
    (`_resolve_relevance_override`) overrode a deep-analysis
    `relevance.label == IRRELEVANT` verdict to RELEVANT -- never set
    otherwise (including for an item whose relevance was never IRRELEVANT
    in the first place). Records the disagreement explicitly rather than
    hiding it: e.g. "qwen_relevance=irrelevant deterministic_scope_signal=
    in_scope final_relevance=relevant override_reason=...". `None` for
    every pre-existing row (predates this field) and for any item the
    safeguard never touched."""

    @field_validator("publication_date")
    @classmethod
    def _require_timezone_aware(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is not None and value.tzinfo is None:
            raise ValueError("publication_date must be timezone-aware")
        return value
