"""Shared builders for events-package tests. No network, no LLM, no clock reliance
(every test that cares about time uses `research_assistant.events.FixedClock`)."""

from __future__ import annotations

from typing import List, Optional

import pytest

from research_assistant.config import get_config
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


def _as_typed_entities(names: Optional[List[str]]) -> List[TypedEntity]:
    """Test convenience: plain entity-name strings default to type COMPANY
    unless a caller passes `TypedEntity`s directly."""
    if not names:
        return []
    return [n if isinstance(n, TypedEntity) else TypedEntity(name=n, type=EntityType.COMPANY) for n in names]


@pytest.fixture
def config():
    return get_config()


def make_item(
    item_id: str,
    *,
    company: str,
    title: Optional[str] = None,
    event_type: Optional[EventType] = None,
    publication_date=None,
    source_type: SourceType = SourceType.PRIMARY,
    source_priority: SourcePriority = SourcePriority.HIGH,
    entities: Optional[List[str]] = None,
    people: Optional[List[str]] = None,
    products: Optional[List[str]] = None,
    partnerships: Optional[List[str]] = None,
    technologies: Optional[List[str]] = None,
    geography: Optional[List[str]] = None,
    summary: Optional[str] = None,
    relevance_label: RelevanceLabel = RelevanceLabel.RELEVANT,
    category: TopicCategory = TopicCategory.ENTERPRISE_AI,
    fs_level: RelevanceLevel = RelevanceLevel.HIGH,
    strategic_level: RelevanceLevel = RelevanceLevel.MEDIUM,
    agent: AgentName = AgentName.PEER,
    article_id: Optional[str] = None,
    publisher: Optional[str] = None,
    url: Optional[str] = None,
    is_dated_development: bool = True,
) -> ResearchItem:
    return ResearchItem(
        id=item_id,
        agent=agent,
        article_id=article_id or f"art-{item_id}",
        publisher=publisher if publisher is not None else company,
        title=title,
        url=url or f"https://example.com/{item_id}",
        publication_date=publication_date,
        company=company,
        relevance=RelevanceDecision(label=relevance_label, reason="test"),
        category=category,
        event_type=event_type,
        summary=summary if summary is not None else title,
        is_dated_development=is_dated_development,
        entities=_as_typed_entities(entities),
        people=people or [],
        products=products or [],
        deal_value=None,
        partnerships=partnerships or [],
        technologies=technologies or [],
        geography=geography or [],
        financial_services_relevance=RelevanceAssessment(level=fs_level, rationale="test"),
        nomura_relevance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="test"),
        strategic_significance=RelevanceAssessment(level=strategic_level, rationale="test"),
        evidence=[f"ev-{item_id}"],
        source_quality=SourceQuality(source_type=source_type, source_priority=source_priority),
    )
