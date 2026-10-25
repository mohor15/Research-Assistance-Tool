"""Deterministic relevance/confidence scoring.

Weights are read from `config/settings.yaml` (`ranking.relevance` /
`ranking.confidence`) — never hardcoded here and never produced by an LLM.
Every component score is a pure function of already-existing, deterministic
data (`ResearchItem` qualitative fields, `Article`-derived source quality,
an injected `Clock`, and the current scoring batch for novelty) — nothing
here makes a network call or an LLM call.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from research_assistant.events.builder import EventDraft
from research_assistant.events.clock import Clock
from research_assistant.events.models import ConfidenceBreakdown, RelevanceBreakdown
from research_assistant.models.enums import GroundingType, SourceType
from research_assistant.models.research_item import RelevanceLabel, RelevanceLevel, ResearchItem, TopicCategory

# A purely metadata-only (LINK-grounded) event can never claim full
# confidence, no matter how the individual component scores above happen
# to land -- those components already trend low naturally for a
# link-grounded item (no entities/technologies/products were extracted,
# since none were invented), but this ceiling makes the distinction
# explicit and guaranteed rather than incidental.
_LINK_GROUNDED_CONFIDENCE_CEILING = 0.6

# --------------------------------------------------------------------------
# Explicit, documented, deterministic qualitative -> numeric mappings.
# Reused for both financial_services_relevance and strategic_significance,
# since both are RelevanceAssessment.level (the same 3-level enum).
# --------------------------------------------------------------------------

RELEVANCE_LEVEL_TO_SCORE: Dict[RelevanceLevel, float] = {
    RelevanceLevel.LOW: 0.25,
    RelevanceLevel.MEDIUM: 0.50,
    RelevanceLevel.HIGH: 0.75,
}

# The agent's own RELEVANT/IRRELEVANT/AMBIGUOUS call, as a numeric factor.
# IRRELEVANT items never reach event clustering (agents already drop them),
# but the mapping is total for safety/completeness.
RELEVANCE_LABEL_TO_SCORE: Dict[RelevanceLabel, float] = {
    RelevanceLabel.RELEVANT: 1.0,
    RelevanceLabel.AMBIGUOUS: 0.6,
    RelevanceLabel.IRRELEVANT: 0.0,
}

# ai_relevance's category multiplier: any real AI-topic bucket (everything
# except the closed-vocabulary catch-all OTHER) counts at full weight.
_CATEGORY_MULTIPLIER: Dict[TopicCategory, float] = {category: 1.0 for category in TopicCategory}
_CATEGORY_MULTIPLIER[TopicCategory.OTHER] = 0.4

# Recency buckets: (max_age_in_days, score), evaluated in order — first
# match wins. Beyond the last bucket, recency is 0.0.
_RECENCY_BUCKETS = [
    (1.0, 1.0),
    (3.0, 0.9),
    (7.0, 0.7),
    (14.0, 0.5),
    (30.0, 0.3),
    (90.0, 0.1),
]

_SOURCE_TYPE_CONFIRMATION: Dict[SourceType, float] = {
    SourceType.PRIMARY: 1.0,
    SourceType.ANALYST: 0.8,
    SourceType.SECONDARY: 0.5,
    SourceType.PRESS_DISTRIBUTION: 0.3,
}

_ENTITY_CONFIDENCE_FULL_COUNT = 3  # 3+ distinct extracted names -> full confidence
_CONTENT_COMPLETENESS_CHECKLIST_SIZE = 5


# --------------------------------------------------------------------------
# Relevance components
# --------------------------------------------------------------------------


def ai_relevance_score(items: List[ResearchItem]) -> float:
    """category-multiplier * relevance-label-score, taking the strongest
    signal across the cluster (an event is as AI-relevant as its most
    clearly-AI-relevant contributing analysis). Derived entirely from the
    agents' own structured classification — no keyword counting."""
    if not items:
        return 0.0
    scores = [
        _CATEGORY_MULTIPLIER.get(item.category, 1.0) * RELEVANCE_LABEL_TO_SCORE.get(item.relevance.label, 0.0)
        for item in items
    ]
    return max(scores)


def financial_services_relevance_score(items: List[ResearchItem]) -> float:
    if not items:
        return 0.0
    return max(RELEVANCE_LEVEL_TO_SCORE[item.financial_services_relevance.level] for item in items)


def strategic_significance_score(items: List[ResearchItem]) -> float:
    if not items:
        return 0.0
    return max(RELEVANCE_LEVEL_TO_SCORE[item.strategic_significance.level] for item in items)


def recency_score(event_date, *, clock: Clock) -> float:
    """1.0 for same-day, decaying through documented buckets, 0.0 beyond 90
    days. A missing event_date scores 0.0 (unknown recency is not assumed
    recent). A future-dated event (age < 0, e.g. clock/timezone skew) is
    clamped to age=0 rather than penalized."""
    if event_date is None:
        return 0.0
    age_days = (clock.now() - event_date).total_seconds() / 86400.0
    if age_days < 0:
        age_days = 0.0
    for max_age, score in _RECENCY_BUCKETS:
        if age_days <= max_age:
            return score
    return 0.0


def novelty_score(draft: EventDraft, other_drafts: List[EventDraft]) -> float:
    """Batch-relative novelty (no external historical database): 1.0 unless
    an EARLIER-dated event in the same batch already shares this event's
    primary_company and event_type — i.e. this is a recurring theme already
    represented earlier in the batch, not a first appearance. A missing
    event_date has no comparison basis and defaults to 1.0 (not penalized).

    Limitation: this is relative to the current scoring batch only. An
    event that is a rehash of something covered in a PREVIOUS run (outside
    this batch) is not detected — see Phase 5's known limitations.
    """
    if draft.event_date is None:
        return 1.0
    for other in other_drafts:
        if other.id == draft.id:
            continue
        if other.event_date is None or other.primary_company != draft.primary_company:
            continue
        if other.event_type != draft.event_type:
            continue
        if other.event_date < draft.event_date:
            return 0.4
    return 1.0


# --------------------------------------------------------------------------
# Confidence components
# --------------------------------------------------------------------------


def primary_source_confirmation(draft: EventDraft) -> float:
    return _SOURCE_TYPE_CONFIRMATION.get(draft.primary_source.source_type, 0.3)


def multiple_source_corroboration(draft: EventDraft) -> float:
    total_sources = 1 + len(draft.secondary_sources)
    if total_sources >= 3:
        return 1.0
    if total_sources == 2:
        return 0.6
    return 0.3


def publication_date_certainty(draft: EventDraft) -> float:
    """Binary presence/absence of `event_date`. Limitation: does not yet
    distinguish an exact timestamp from a date-only value (Article's
    DatePrecision is not currently propagated through ResearchItem) —
    documented in Phase 5's known limitations."""
    return 1.0 if draft.event_date is not None else 0.0


def entity_extraction_confidence(items: List[ResearchItem]) -> float:
    names = set()
    for item in items:
        names.update(e.name for e in item.entities)  # typed (name, type) list (Phase 13B); name only
        names.update(item.people)
        names.update(item.products)
        names.update(item.technologies)
    return min(1.0, len(names) / _ENTITY_CONFIDENCE_FULL_COUNT)


def event_classification_confidence(event_type) -> float:
    return 1.0 if event_type is not None else 0.5


def content_completeness(items: List[ResearchItem]) -> float:
    checklist = [
        any(item.summary for item in items),
        any(item.event_type is not None for item in items),
        any(item.entities for item in items),
        any(item.technologies or item.partnerships or item.products for item in items),
        any(item.geography for item in items),
    ]
    return sum(1 for present in checklist if present) / _CONTENT_COMPLETENESS_CHECKLIST_SIZE


# --------------------------------------------------------------------------
# Weighted breakdowns
# --------------------------------------------------------------------------


def compute_relevance_breakdown(
    draft: EventDraft, items: List[ResearchItem], other_drafts: List[EventDraft], *, weights, clock: Clock
) -> RelevanceBreakdown:
    ai = ai_relevance_score(items)
    fs = financial_services_relevance_score(items)
    strategic = strategic_significance_score(items)
    recency = recency_score(draft.event_date, clock=clock)
    novelty = novelty_score(draft, other_drafts)

    weighted_total = (
        weights.ai_relevance * ai
        + weights.financial_services_relevance * fs
        + weights.strategic_significance * strategic
        + weights.recency * recency
        + weights.novelty * novelty
    )
    return RelevanceBreakdown(
        ai_relevance=round(ai, 4),
        financial_services_relevance=round(fs, 4),
        strategic_significance=round(strategic, 4),
        recency=round(recency, 4),
        novelty=round(novelty, 4),
        weighted_total=round(min(1.0, max(0.0, weighted_total)), 4),
    )


def compute_confidence_breakdown(draft: EventDraft, items: List[ResearchItem], *, weights) -> ConfidenceBreakdown:
    psc = primary_source_confirmation(draft)
    msc = multiple_source_corroboration(draft)
    pdc = publication_date_certainty(draft)
    eec = entity_extraction_confidence(items)
    ecc = event_classification_confidence(draft.event_type)
    cc = content_completeness(items)

    weighted_total = (
        weights.primary_source_confirmation * psc
        + weights.multiple_source_corroboration * msc
        + weights.publication_date_certainty * pdc
        + weights.entity_extraction_confidence * eec
        + weights.event_classification_confidence * ecc
        + weights.content_completeness * cc
    )
    weighted_total = min(1.0, max(0.0, weighted_total))
    if items and all(item.grounding_type == GroundingType.LINK for item in items):
        weighted_total = min(weighted_total, _LINK_GROUNDED_CONFIDENCE_CEILING)
    return ConfidenceBreakdown(
        primary_source_confirmation=round(psc, 4),
        multiple_source_corroboration=round(msc, 4),
        publication_date_certainty=round(pdc, 4),
        entity_extraction_confidence=round(eec, 4),
        event_classification_confidence=round(ecc, 4),
        content_completeness=round(cc, 4),
        weighted_total=round(weighted_total, 4),
    )
