"""AppTest target for `tests/ui/test_digest_event_rendering.py` -- renders
`_render_digest_event` directly against hand-built domain objects, never
touching `_get_persistence_service()`/`get_config()` (and therefore never
touching the real `data/research.db` or its `st.cache_resource` cache)."""

from datetime import datetime, timezone

import streamlit as st

from research_assistant.events.models import (
    ClusterMetadata,
    ConfidenceBreakdown,
    Event,
    EventScoringBreakdown,
    RelevanceBreakdown,
    SourceRef,
)
from research_assistant.models.enums import EventType, SourceType, SourcePriority
from research_assistant.models.report import Report, ReportDetails, ReportFailure
from research_assistant.ui import dashboard_data as dd
from research_assistant.ui.app import _render_digest_event

D = datetime(2026, 9, 28, tzinfo=timezone.utc)


def _event(event_id: str) -> Event:
    primary_source = SourceRef(
        research_item_id="ri1", article_id="art1", url="https://example.com/real-article",
        publisher="Example Publisher", source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
    )
    return Event(
        id=event_id, canonical_title="Example Event Title", event_type=EventType.PRODUCT_LAUNCH,
        primary_company="Example Co", related_companies=[], event_date=D, description=None,
        research_items=["ri1"], articles=["art1"], primary_source=primary_source, secondary_sources=[],
        relevance_score=0.84, confidence_score=0.62,
        scoring_breakdown=EventScoringBreakdown(
            relevance=RelevanceBreakdown(
                ai_relevance=1.0, financial_services_relevance=0.75, strategic_significance=0.5,
                recency=1.0, novelty=1.0, weighted_total=0.84,
            ),
            confidence=ConfidenceBreakdown(
                primary_source_confirmation=1.0, multiple_source_corroboration=0.3,
                publication_date_certainty=1.0, entity_extraction_confidence=1.0,
                event_classification_confidence=0.5, content_completeness=0.4, weighted_total=0.62,
            ),
        ),
        cluster_metadata=ClusterMetadata(member_count=1, matches=[]),
    )


case = st.query_params.get("case", "with_report")

if case == "with_report":
    report = Report(
        event_id="ev1", title="Example Event Title", opening_context="REAL_OPENING_CONTEXT_MARKER",
        details=ReportDetails(narrative="details narrative"),
        strategic_implications="REAL_STRATEGIC_IMPLICATIONS_MARKER",
        nomura_opportunity="x", sources=[], relevance_score=0.84, confidence_score=0.62,
    )
    _render_digest_event(dd.EventDetail(event=_event("ev1"), research_items=[], report=report))
elif case == "report_failure":
    failure = ReportFailure(event_id="ev2", reason="llm_unavailable", detail="Ollama unreachable")
    _render_digest_event(dd.EventDetail(event=_event("ev2"), research_items=[], report=failure))
elif case == "no_report":
    _render_digest_event(dd.EventDetail(event=_event("ev3"), research_items=[], report=None))
