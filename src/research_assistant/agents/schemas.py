"""The schema an agent's LLM call must return, validated by `OllamaClient.generate_structured`.

Deliberately excludes everything the agent already knows deterministically
from the input Article/Evidence (id, url, publisher, publication_date,
company, evidence ids, source_quality) — the model is only ever asked for
its interpretation/extraction, never to restate provenance it could get
wrong or invent.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from research_assistant.models.enums import EventType
from research_assistant.models.research_item import (
    AnalystFindings,
    RelevanceAssessment,
    RelevanceDecision,
    TopicCategory,
    TypedEntity,
)


class AgentAssessment(BaseModel):
    relevance: RelevanceDecision
    category: TopicCategory
    event_type: Optional[EventType] = None
    summary: Optional[str] = None

    is_dated_development: bool
    development_date_rationale: Optional[str] = None

    entities: List[TypedEntity] = Field(default_factory=list)
    people: List[str] = Field(default_factory=list)
    products: List[str] = Field(default_factory=list)
    deal_value: Optional[str] = None
    partnerships: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    geography: List[str] = Field(default_factory=list)

    financial_services_relevance: RelevanceAssessment
    strategic_significance: RelevanceAssessment
    # `nomura_relevance` deliberately NOT requested from the model (CPU-only
    # qwen2.5:3b deep-analysis cost reduction): grep across events/scoring.py,
    # events/builder.py, events/similarity.py, agents/report.py, and ui/*.py
    # shows it is used NOWHERE downstream except being persisted -- only
    # `.level` on `financial_services_relevance`/`strategic_significance`
    # ever feeds Phase 5 scoring, and neither `ResearchItem.nomura_relevance`
    # nor its `.rationale` is read by any builder/report/UI code path. It is
    # derived deterministically instead (`agents.base._derive_nomura_relevance`,
    # mirroring `strategic_significance`) so the `research_items` table's
    # pre-existing NOT NULL columns stay populated without spending a third
    # LLM-generated free-text paragraph on a field nothing downstream reads.


class AnalystAssessment(AgentAssessment):
    """AnalystAgent's structured output — everything `AgentAssessment` has,
    plus the analyst-specific findings extraction. A separate subclass
    (rather than adding an optional field to the shared schema) keeps
    PeerAgent/AIProductAgent/StartupAgent's schema untouched."""

    analyst_findings: AnalystFindings


# Forbidden prescriptive phrasing for `nomura_opportunity` — a defense-in-depth
# guardrail beyond the prompt instructions: even if the model ignores the
# system prompt, output containing these phrases fails validation (raising
# LLMResponseError) rather than silently reaching a report.
_FORBIDDEN_PRESCRIPTIVE_PHRASES = (
    "nomura should", "nomura must", "nomura needs to", "nomura has to", "nomura will need to",
)


def _validate_diplomatic_nomura_opportunity(value: str) -> str:
    """Shared by `ReportNarrative` and `TierBReportNarrative` -- the same
    defense-in-depth guardrail applies regardless of whether the narrative
    was grounded on a full Event or a single Tier B article."""
    lowered = value.lower()
    for phrase in _FORBIDDEN_PRESCRIPTIVE_PHRASES:
        if phrase in lowered:
            raise ValueError(
                f"nomura_opportunity must not contain prescriptive language ({phrase!r}); "
                "use diplomatic framing such as 'may be relevant to Nomura's consideration of...'"
            )
    return value


_MIN_KEY_INSIGHTS = 2
_MAX_KEY_INSIGHTS = 4
_MAX_KEY_INSIGHT_CHARS = 280


def _bounded_nonempty_insights(value: List[str]) -> List[str]:
    """Shared by `ReportNarrative` and `TierBReportNarrative` -- a bounded,
    non-empty, length-capped list of short takeaways, so a malformed/
    runaway generation fails schema validation rather than reaching a
    report (same defense-in-depth spirit as `_validate_diplomatic_
    nomura_opportunity`)."""
    cleaned = []
    for item in value:
        stripped = item.strip()
        if not stripped:
            raise ValueError("key_insights items must be non-empty")
        if len(stripped) > _MAX_KEY_INSIGHT_CHARS:
            raise ValueError(f"key_insights item exceeds {_MAX_KEY_INSIGHT_CHARS} characters")
        cleaned.append(stripped)
    return cleaned


class ReportNarrative(BaseModel):
    """ReportAgent's structured output — five narrative sections.

    Every factual list (companies/people/products/...) and every source URL
    is deterministic, Python-aggregated from the Event's ResearchItems
    elsewhere (`agents.report`) — never part of this schema, so the model
    is never even asked to restate or invent them. `key_insights` (added
    2026-10-15, Tier A/Tier B quality-parity fix) is a bounded, REQUIRED
    list for every fresh generation (this schema is only ever the live
    LLM output, never reloaded from old persisted rows -- the persisted
    `models.report.ReportDetails.key_insights` is the one with a `[]`
    default, for backward compatibility with reports written before this
    field existed)."""

    opening_context: str = Field(min_length=1)
    details_narrative: str = Field(min_length=1)
    key_insights: List[str] = Field(min_length=_MIN_KEY_INSIGHTS, max_length=_MAX_KEY_INSIGHTS)
    strategic_implications: str = Field(min_length=1)
    nomura_opportunity: str = Field(min_length=1)

    @field_validator("key_insights")
    @classmethod
    def _bounded_insights(cls, value: List[str]) -> List[str]:
        return _bounded_nonempty_insights(value)

    @field_validator("nomura_opportunity")
    @classmethod
    def _must_be_diplomatic_not_prescriptive(cls, value: str) -> str:
        return _validate_diplomatic_nomura_opportunity(value)


_TIER_B_MAX_NARRATIVE_CHARS = 1500
_TIER_B_MAX_NOMURA_OPPORTUNITY_CHARS = 800
_TIER_B_MIN_KEY_INSIGHTS = _MIN_KEY_INSIGHTS
_TIER_B_MAX_KEY_INSIGHTS = _MAX_KEY_INSIGHTS
_TIER_B_MAX_KEY_INSIGHT_CHARS = _MAX_KEY_INSIGHT_CHARS


class TierBReportNarrative(BaseModel):
    """ReportAgent's Tier B structured output -- a single-article, no-Event
    narrative (a Tier B article has no ResearchItems/Event to aggregate
    deterministic facts from the way Tier A's report-building does). The
    single `details_narrative` prose field is replaced with a bounded
    `key_insights` list (~2-4 short, specific, evidence-grounded takeaways
    per the feature spec) -- `opening_context`, `strategic_implications`,
    and `nomura_opportunity` keep the exact same meaning and the exact same
    diplomatic/non-prescriptive guardrail as `ReportNarrative`'s. Every
    field is bounded in length so a malformed/runaway generation fails
    schema validation rather than reaching a report."""

    opening_context: str = Field(min_length=1, max_length=_TIER_B_MAX_NARRATIVE_CHARS)
    key_insights: List[str] = Field(min_length=_TIER_B_MIN_KEY_INSIGHTS, max_length=_TIER_B_MAX_KEY_INSIGHTS)
    strategic_implications: str = Field(min_length=1, max_length=_TIER_B_MAX_NARRATIVE_CHARS)
    nomura_opportunity: str = Field(min_length=1, max_length=_TIER_B_MAX_NOMURA_OPPORTUNITY_CHARS)

    @field_validator("key_insights")
    @classmethod
    def _bounded_nonempty_insights(cls, value: List[str]) -> List[str]:
        cleaned = []
        for item in value:
            stripped = item.strip()
            if not stripped:
                raise ValueError("key_insights items must be non-empty")
            if len(stripped) > _TIER_B_MAX_KEY_INSIGHT_CHARS:
                raise ValueError(f"key_insights item exceeds {_TIER_B_MAX_KEY_INSIGHT_CHARS} characters")
            cleaned.append(stripped)
        return cleaned

    @field_validator("nomura_opportunity")
    @classmethod
    def _must_be_diplomatic_not_prescriptive(cls, value: str) -> str:
        return _validate_diplomatic_nomura_opportunity(value)
