"""Deterministic Event ranking: relevance desc, confidence tie-break, date
tie-break, id as the final deterministic tie-break. No subjective labeling."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.events.models import ClusterMetadata, Event, EventScoringBreakdown, SourceRef
from research_assistant.events.ranking import rank_events
from research_assistant.models.enums import SourcePriority, SourceType

from tests.events.test_scoring_relevance import D as _D  # reuse a fixed reference datetime


def _event(id_, *, relevance, confidence, event_date=None):
    ref = SourceRef(
        research_item_id=f"ri-{id_}", article_id=f"art-{id_}", url=f"https://example.com/{id_}",
        source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
    )
    breakdown = EventScoringBreakdown(
        relevance={"ai_relevance": 0, "financial_services_relevance": 0, "strategic_significance": 0, "recency": 0, "novelty": 0, "weighted_total": relevance},
        confidence={"primary_source_confirmation": 0, "multiple_source_corroboration": 0, "publication_date_certainty": 0, "entity_extraction_confidence": 0, "event_classification_confidence": 0, "content_completeness": 0, "weighted_total": confidence},
    )
    return Event(
        id=id_, canonical_title="t", primary_company="X", event_date=event_date,
        research_items=[f"ri-{id_}"], articles=[f"art-{id_}"], primary_source=ref,
        relevance_score=relevance, confidence_score=confidence,
        scoring_breakdown=breakdown, cluster_metadata=ClusterMetadata(member_count=1),
    )


def test_higher_relevance_score_ranks_first():
    low = _event("e1", relevance=0.3, confidence=0.9)
    high = _event("e2", relevance=0.9, confidence=0.1)

    ranked = rank_events([low, high])

    assert [e.id for e in ranked] == ["e2", "e1"]


def test_confidence_breaks_ties_when_relevance_equal():
    lower_conf = _event("e1", relevance=0.5, confidence=0.2)
    higher_conf = _event("e2", relevance=0.5, confidence=0.8)

    ranked = rank_events([lower_conf, higher_conf])

    assert [e.id for e in ranked] == ["e2", "e1"]


def test_more_recent_event_date_breaks_ties_when_relevance_and_confidence_equal():
    from datetime import timedelta

    older = _event("e1", relevance=0.5, confidence=0.5, event_date=_D)
    newer = _event("e2", relevance=0.5, confidence=0.5, event_date=_D + timedelta(days=3))

    ranked = rank_events([older, newer])

    assert [e.id for e in ranked] == ["e2", "e1"]


def test_missing_event_date_sorts_after_dated_events_in_a_tie():
    dated = _event("e1", relevance=0.5, confidence=0.5, event_date=_D)
    undated = _event("e2", relevance=0.5, confidence=0.5, event_date=None)

    ranked = rank_events([undated, dated])

    assert [e.id for e in ranked] == ["e1", "e2"]


def test_event_id_is_the_final_deterministic_tiebreaker():
    a = _event("zzz", relevance=0.5, confidence=0.5, event_date=_D)
    b = _event("aaa", relevance=0.5, confidence=0.5, event_date=_D)

    ranked = rank_events([a, b])

    assert [e.id for e in ranked] == ["aaa", "zzz"]


def test_ranking_is_stable_and_deterministic_across_repeated_calls():
    events = [
        _event("e1", relevance=0.5, confidence=0.5, event_date=_D),
        _event("e2", relevance=0.9, confidence=0.1, event_date=_D),
        _event("e3", relevance=0.5, confidence=0.9, event_date=_D),
    ]

    first = [e.id for e in rank_events(events)]
    second = [e.id for e in rank_events(list(reversed(events)))]

    assert first == second == ["e2", "e3", "e1"]


def test_ranking_never_attaches_a_subjective_label():
    event = _event("e1", relevance=0.9, confidence=0.9, event_date=_D)
    assert not hasattr(event, "is_best")
    assert not hasattr(event, "label")
