"""The single entry point later phases (Report Agent, Phase 6) should use.

    EventService(config).build_events(research_items)  ->  ranked, scored Events

Mirrors `RetrievalService`/`NormalizationService` from earlier phases: one
class, injectable dependencies (`clock`), no hidden global state.
"""

from __future__ import annotations

from typing import List, Optional

from research_assistant.config.schema import AppConfig
from research_assistant.events.builder import EventDraft, build_event_draft, flatten_non_company_vocabulary
from research_assistant.events.clock import Clock, SystemClock
from research_assistant.events.clustering import cluster_research_items
from research_assistant.events.models import Event, EventScoringBreakdown
from research_assistant.events.ranking import rank_events
from research_assistant.events.scoring import compute_confidence_breakdown, compute_relevance_breakdown
from research_assistant.models.research_item import ResearchItem


class EventService:
    def __init__(self, config: AppConfig, *, clock: Optional[Clock] = None) -> None:
        self._config = config
        self._clock = clock or SystemClock()
        self._non_company_vocabulary = flatten_non_company_vocabulary(config.topics)

    def build_events(self, research_items: List[ResearchItem]) -> List[Event]:
        """Cluster -> score -> rank. Returns Events already in ranked order."""
        clusters = cluster_research_items(research_items)
        items_by_id = {item.id: item for item in research_items}

        drafts = [
            build_event_draft(cluster, excluded_terms=self._non_company_vocabulary) for cluster in clusters
        ]
        events = [self._score_draft(draft, items_by_id, drafts) for draft in drafts]
        return rank_events(events)

    def _score_draft(self, draft: EventDraft, items_by_id, all_drafts: List[EventDraft]) -> Event:
        member_items = [items_by_id[item_id] for item_id in draft.research_items]
        relevance = compute_relevance_breakdown(
            draft, member_items, all_drafts, weights=self._config.settings.ranking.relevance, clock=self._clock
        )
        confidence = compute_confidence_breakdown(
            draft, member_items, weights=self._config.settings.ranking.confidence
        )
        return Event(
            id=draft.id,
            canonical_title=draft.canonical_title,
            event_type=draft.event_type,
            primary_company=draft.primary_company,
            related_companies=draft.related_companies,
            event_date=draft.event_date,
            description=draft.description,
            research_items=draft.research_items,
            articles=draft.articles,
            primary_source=draft.primary_source,
            secondary_sources=draft.secondary_sources,
            relevance_score=relevance.weighted_total,
            confidence_score=confidence.weighted_total,
            scoring_breakdown=EventScoringBreakdown(relevance=relevance, confidence=confidence),
            cluster_metadata=draft.cluster_metadata,
            grounding_type=draft.grounding_type,
        )
