"""Shared builders for reporting-package tests (ReportAgent + DigestService).
No network, no LLM — every LLM-facing test uses a scripted FakeLLMClient."""

from __future__ import annotations

from typing import List, Optional, Type, TypeVar

import pytest
from pydantic import BaseModel

from research_assistant.config import get_config
from research_assistant.events.models import (
    ClusterMetadata,
    ConfidenceBreakdown,
    Event,
    EventScoringBreakdown,
    RelevanceBreakdown,
    SourceRef,
)
from research_assistant.models.enums import AgentName, EntityType, EventType, SourcePriority, SourceType
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    ResearchItem,
    SourceQuality,
    TopicCategory,
    TypedEntity,
)

# Re-exposes the storage-package `persistence_service`/`database` fixtures
# for reporting-package tests (the Tier B report-resolution orchestration
# tests need a real, temp-directory-backed PersistenceService) without
# duplicating the fixture definitions -- same pattern as `tests/ui/conftest.py`.
from tests.storage.conftest import database, persistence_service  # noqa: E402,F401

def _as_typed_entities(names: Optional[List[str]]) -> List[TypedEntity]:
    """Test convenience: plain entity-name strings default to type COMPANY
    (the overwhelmingly common case in these fixtures) unless a caller
    needs a different type, in which case it passes `TypedEntity`s directly."""
    if not names:
        return []
    return [n if isinstance(n, TypedEntity) else TypedEntity(name=n, type=EntityType.COMPANY) for n in names]

T = TypeVar("T", bound=BaseModel)


class FakeLLMClient:
    """Scripted stand-in for `OllamaClient`. `responses` is a list popped in
    order; an item that is an Exception is raised instead of returned."""

    def __init__(self, responses: List):
        self._responses = list(responses)
        self.calls: List[tuple] = []

    def generate_structured(self, *, system: str, prompt: str, schema: Type[T]) -> T:
        self.calls.append((system, prompt, schema))
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def config():
    return get_config()


def make_research_item(
    item_id: str,
    *,
    company: str = "JPMorgan Chase",
    title: Optional[str] = "JPMorgan launches AI copilot",
    event_type: Optional[EventType] = EventType.PRODUCT_LAUNCH,
    publication_date=None,
    source_type: SourceType = SourceType.PRIMARY,
    source_priority: SourcePriority = SourcePriority.HIGH,
    entities: Optional[List[str]] = None,
    people: Optional[List[str]] = None,
    products: Optional[List[str]] = None,
    partnerships: Optional[List[str]] = None,
    technologies: Optional[List[str]] = None,
    geography: Optional[List[str]] = None,
    deal_value: Optional[str] = None,
    summary: Optional[str] = "JPMorgan launched a GenAI copilot for advisors.",
    url: Optional[str] = None,
    article_id: Optional[str] = None,
    publisher: Optional[str] = None,
    is_dated_development: bool = True,
) -> ResearchItem:
    return ResearchItem(
        id=item_id,
        agent=AgentName.PEER,
        article_id=article_id or f"art-{item_id}",
        publisher=publisher if publisher is not None else company,
        title=title,
        url=url or f"https://example.com/{item_id}",
        publication_date=publication_date,
        company=company,
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="test"),
        category=TopicCategory.ENTERPRISE_AI,
        event_type=event_type,
        summary=summary,
        is_dated_development=is_dated_development,
        entities=_as_typed_entities(entities),
        people=people or [],
        products=products or [],
        deal_value=deal_value,
        partnerships=partnerships or [],
        technologies=technologies or [],
        geography=geography or [],
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="test"),
        nomura_relevance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="test"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="test"),
        evidence=[f"ev-{item_id}"],
        source_quality=SourceQuality(source_type=source_type, source_priority=source_priority),
    )


def make_event(
    event_id: str,
    *,
    research_items: List[ResearchItem],
    canonical_title: str = "JPMorgan launches AI copilot",
    event_type: Optional[EventType] = EventType.PRODUCT_LAUNCH,
    primary_company: str = "JPMorgan Chase",
    related_companies: Optional[List[str]] = None,
    event_date=None,
    description: Optional[str] = "JPMorgan launched a GenAI copilot for advisors.",
    relevance_score: float = 0.8,
    confidence_score: float = 0.7,
) -> Event:
    primary_item = research_items[0]
    primary_source = SourceRef(
        research_item_id=primary_item.id, article_id=primary_item.article_id, url=primary_item.url,
        publisher=primary_item.publisher, source_type=primary_item.source_quality.source_type,
        source_priority=primary_item.source_quality.source_priority,
    )
    secondary_sources = [
        SourceRef(
            research_item_id=item.id, article_id=item.article_id, url=item.url, publisher=item.publisher,
            source_type=item.source_quality.source_type, source_priority=item.source_quality.source_priority,
        )
        for item in research_items[1:]
    ]
    return Event(
        id=event_id,
        canonical_title=canonical_title,
        event_type=event_type,
        primary_company=primary_company,
        related_companies=related_companies or [],
        event_date=event_date,
        description=description,
        research_items=[item.id for item in research_items],
        articles=[item.article_id for item in research_items],
        primary_source=primary_source,
        secondary_sources=secondary_sources,
        relevance_score=relevance_score,
        confidence_score=confidence_score,
        scoring_breakdown=EventScoringBreakdown(
            relevance=RelevanceBreakdown(
                ai_relevance=1.0, financial_services_relevance=0.75, strategic_significance=0.5,
                recency=1.0, novelty=1.0, weighted_total=relevance_score,
            ),
            confidence=ConfidenceBreakdown(
                primary_source_confirmation=1.0, multiple_source_corroboration=0.3,
                publication_date_certainty=1.0, entity_extraction_confidence=0.3,
                event_classification_confidence=1.0, content_completeness=0.6,
                weighted_total=confidence_score,
            ),
        ),
        cluster_metadata=ClusterMetadata(member_count=len(research_items)),
    )


def make_narrative(**overrides):
    from research_assistant.agents.schemas import ReportNarrative

    kwargs = dict(
        opening_context="JPMorgan Chase launched a generative AI copilot for financial advisors.",
        details_narrative="The copilot helps advisors draft research summaries; no deal value was disclosed.",
        key_insights=[
            "The copilot targets research-summary drafting specifically, not general-purpose chat.",
            "No deal value or external partner was disclosed alongside the launch.",
        ],
        strategic_implications="This may signal a broader trend of GenAI tooling adoption among peer banks.",
        nomura_opportunity="This development may be relevant to Nomura's consideration of similar advisor tooling.",
    )
    kwargs.update(overrides)
    return ReportNarrative(**kwargs)
