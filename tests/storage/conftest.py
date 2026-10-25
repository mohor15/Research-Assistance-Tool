"""Shared builders for storage-package tests. Every test uses a fresh
temp-directory SQLite file — never a shared/real database."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pytest

from research_assistant.events.models import (
    ClusterMetadata,
    ConfidenceBreakdown,
    Event,
    EventScoringBreakdown,
    RelevanceBreakdown,
    SourceRef,
)
from research_assistant.models.article import Article, DatePrecision
from research_assistant.models.digest import Digest, DigestTheme, DigestType
from research_assistant.models.enums import AgentName, EntityType, EventType, SourcePriority, SourceType
from research_assistant.models.report import Report, ReportDetails, ReportFailure, ReportSource
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
from research_assistant.retrieval.models import ExtractionMethod, ExtractionStatus
from research_assistant.storage.database import Database
from research_assistant.storage.persistence_service import PersistenceService

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def database(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    db.initialize()
    return db


@pytest.fixture
def persistence_service(database: Database) -> PersistenceService:
    return PersistenceService(database)


def make_article(
    article_id: str = "art1", *, company: str = "JPMorgan Chase", url: Optional[str] = None,
    canonical_url: Optional[str] = None, content_hash: Optional[str] = "hash1",
    publication_date=D, clean_text: str = "Clean article text.",
) -> Article:
    return Article(
        id=article_id, url=url or f"https://example.com/{article_id}",
        canonical_url=canonical_url or url or f"https://example.com/{article_id}",
        domain="example.com", source_name=company, source_type=SourceType.PRIMARY,
        source_priority=SourcePriority.HIGH, publisher=company, title="An article title", author=None,
        publication_date=publication_date, publication_date_precision=DatePrecision.EXACT, retrieved_at=D,
        raw_text="raw text", clean_text=clean_text, content_hash=content_hash,
        extraction_method=ExtractionMethod.BS4_FALLBACK, extraction_status=ExtractionStatus.SUCCESS,
        metadata={"reason": "test"},
    )


def make_research_item(
    item_id: str = "ri1", *, article_id: str = "art1", company: str = "JPMorgan Chase",
    agent: AgentName = AgentName.PEER, event_type: Optional[EventType] = EventType.PRODUCT_LAUNCH,
    publication_date=D, entities: Optional[List[str]] = None, evidence_ids: Optional[List[str]] = None,
    source_type: SourceType = SourceType.PRIMARY, source_priority: SourcePriority = SourcePriority.HIGH,
    is_dated_development: bool = True, run_id: Optional[str] = None,
    relevance_override: Optional[str] = None,
) -> ResearchItem:
    return ResearchItem(
        id=item_id, agent=agent, article_id=article_id, publisher=company, title="Research item title",
        url=f"https://example.com/{article_id}", publication_date=publication_date, company=company,
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="test reason"),
        category=TopicCategory.ENTERPRISE_AI, event_type=event_type, summary="A summary.",
        is_dated_development=is_dated_development,
        entities=_as_typed_entities(entities) or [TypedEntity(name="Entity A", type=EntityType.COMPANY)],
        people=["Jane Doe"], products=["Product X"],
        deal_value="$10 million", partnerships=["Partner Co"], technologies=["GenAI"], geography=["United States"],
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="test"),
        nomura_relevance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="test"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="test"),
        evidence=evidence_ids if evidence_ids is not None else ["ev1"],
        source_quality=SourceQuality(source_type=source_type, source_priority=source_priority),
        run_id=run_id,
        relevance_override=relevance_override,
    )


def make_event(
    event_id: str = "ev1", *, research_item_ids: Optional[List[str]] = None, article_ids: Optional[List[str]] = None,
    company: str = "JPMorgan Chase", event_type: Optional[EventType] = EventType.PRODUCT_LAUNCH,
    event_date=D, relevance_score: float = 0.85, confidence_score: float = 0.75,
    secondary_source_ids: Optional[List[str]] = None, run_id: Optional[str] = None,
) -> Event:
    research_item_ids = research_item_ids or ["ri1"]
    article_ids = article_ids or ["art1"]
    primary_source = SourceRef(
        research_item_id=research_item_ids[0], article_id=article_ids[0],
        url=f"https://example.com/{article_ids[0]}", publisher=company,
        source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
    )
    secondary_sources = [
        SourceRef(
            research_item_id=rid, article_id=aid, url=f"https://example.com/{aid}", publisher="Reuters",
            source_type=SourceType.SECONDARY, source_priority=SourcePriority.HIGH,
        )
        for rid, aid in (secondary_source_ids or [])
    ]
    return Event(
        id=event_id, canonical_title="An event title", event_type=event_type, primary_company=company,
        related_companies=["Related Co"], event_date=event_date, description="An event description.",
        research_items=research_item_ids, articles=article_ids, primary_source=primary_source,
        secondary_sources=secondary_sources, relevance_score=relevance_score, confidence_score=confidence_score,
        scoring_breakdown=EventScoringBreakdown(
            relevance=RelevanceBreakdown(
                ai_relevance=1.0, financial_services_relevance=0.75, strategic_significance=0.5,
                recency=1.0, novelty=1.0, weighted_total=relevance_score,
            ),
            confidence=ConfidenceBreakdown(
                primary_source_confirmation=1.0, multiple_source_corroboration=0.3,
                publication_date_certainty=1.0, entity_extraction_confidence=0.3,
                event_classification_confidence=1.0, content_completeness=0.6, weighted_total=confidence_score,
            ),
        ),
        cluster_metadata=ClusterMetadata(member_count=len(research_item_ids)),
        run_id=run_id,
    )


def make_report(event_id: str = "ev1", *, relevance_score: float = 0.85, confidence_score: float = 0.75) -> Report:
    return Report(
        event_id=event_id, title="An event title", opening_context="Opening context.",
        details=ReportDetails(companies=["JPMorgan Chase"], people=["Jane Doe"], narrative="Details narrative."),
        strategic_implications="Strategic implications.",
        nomura_opportunity="This may be relevant to Nomura's consideration of similar tooling.",
        sources=[
            ReportSource(
                publisher="JPMorgan Chase", title="An article title", publication_date=D,
                url="https://example.com/art1", source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
            )
        ],
        relevance_score=relevance_score, confidence_score=confidence_score,
    )


def make_report_failure(event_id: str = "ev1") -> ReportFailure:
    return ReportFailure(event_id=event_id, reason="LLMUnavailableError", detail="Ollama not running")


def make_digest(digest_id: str = "dg1", *, event_ids: Optional[List[str]] = None, digest_type: DigestType = DigestType.MORNING) -> Digest:
    event_ids = event_ids or ["ev1"]
    return Digest(
        id=digest_id, digest_type=digest_type, generated_at=D, period_start=D, period_end=D,
        events=event_ids, themes=[DigestTheme(label="enterprise AI", description="1 event", event_ids=event_ids)],
    )
