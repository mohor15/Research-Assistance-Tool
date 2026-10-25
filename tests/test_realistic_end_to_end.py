"""Phase 8 Part 12 — a single, controlled, realistic end-to-end test
exercising retrieval -> normalization -> evidence -> research agents ->
event clustering -> scoring -> ranking -> report generation -> digest ->
persistence, entirely with deterministic fake/scripted services (no live
internet, no live Ollama, no paid API), verifying all nine required
assertions explicitly.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.report import ReportAgent
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.config import get_config
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.evidence.service import NormalizationService
from research_assistant.models.enums import AgentName, SourcePriority, SourceType
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
)
from research_assistant.pipeline import PipelineServices, PipelineState, build_graph
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition
from research_assistant.storage import Database, PersistenceService

from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)
ORIGINAL_URL = "https://www.jpmorganchase.com/newsroom/ai-copilot?utm_source=twitter"


class _FakeLLMClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate_structured(self, *, system, prompt, schema):
        return self._responses.pop(0)


class _FakeRetrievalService:
    def list_sources(self):
        return [
            SourceDefinition(
                name="JPMorgan Chase", domain="jpmorganchase.com", urls=["https://jpmorganchase.com/newsroom"],
                source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
            )
        ]

    def retrieve_source(self, source, *, cache_policy=None):
        return [
            ExtractedArticle(
                url=ORIGINAL_URL, title="JPMorgan launches AI copilot",
                text="JPMorgan Chase today launched a new generative AI copilot for financial advisors, "
                     "built on internal infrastructure to help draft client research summaries.",
                # Phase 13F: strictly before the test's own period_end=D --
                # a digest period is now the half-open interval
                # [period_start, period_end), so an article/event dated
                # exactly at period_end would fall on the excluded boundary.
                published_at=D - timedelta(hours=1),
            )
        ]


def _relevant_assessment() -> AgentAssessment:
    return AgentAssessment(
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="Concrete GenAI copilot launch."),
        category=TopicCategory.ENTERPRISE_AI,
        summary="JPMorgan launched a GenAI copilot for financial advisors.",
        products=["AI copilot"],
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="Direct FS application."),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="Signals adoption trend."),
    )


def _build_services(database):
    config = get_config()
    return PipelineServices(
        config=config,
        retrieval_service=_FakeRetrievalService(),
        normalization_service=NormalizationService(),
        agents={AgentName.PEER: PeerAgent(_FakeLLMClient([_relevant_assessment()]), config)},
        event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
        digest_service=DigestService(config, clock=FixedClock(D)),
        clock=FixedClock(D),
        persistence_service=PersistenceService(database),
    )


def test_realistic_end_to_end_pipeline_with_all_required_assertions(tmp_path: Path):
    database = Database(tmp_path / "e2e.db")
    database.initialize()
    services = _build_services(database)
    graph = build_graph(services)

    state = PipelineState(period_start=D - timedelta(hours=12), period_end=D)
    result = graph.invoke(state)

    # 1. Event is created.
    assert len(result["events"]) == 1
    event = result["events"][0]

    # 2. Event score is preserved (matches EventService's own deterministic output).
    direct_events = services.event_service.build_events(result["research_items"])
    assert event.relevance_score == direct_events[0].relevance_score
    assert event.confidence_score == direct_events[0].confidence_score

    # 3. Report is generated.
    assert len(result["reports"]) == 1
    report = result["reports"][0]
    assert report.event_id == event.id

    # 4. Digest is generated.
    assert result["digest"] is not None
    assert result["digest"].events == [event.id]

    # 5. Data is persisted.
    conn = database.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM digests").fetchone()[0] == 1
    finally:
        conn.close()

    # 6. Provenance survives (Report -> Event -> ResearchItem -> Evidence -> Article).
    persisted_event = services.persistence_service.events.get(database.connect(), event.id)
    assert persisted_event.research_items == event.research_items
    assert persisted_event.articles == event.articles
    persisted_research_item = services.persistence_service.research_items.get(database.connect(), event.research_items[0])
    assert persisted_research_item.article_id == event.articles[0]
    persisted_evidence = services.persistence_service.evidence.get_by_article_id(database.connect(), event.articles[0])
    assert len(persisted_evidence) == 1
    assert persisted_evidence[0].article_id == event.articles[0]

    # 7. Original source URL survives exactly (including its tracking query string).
    assert event.primary_source.url == ORIGINAL_URL
    assert report.sources[0].url == ORIGINAL_URL
    assert result["articles"][0].url == ORIGINAL_URL

    # 8. Second execution is idempotent.
    services_2 = _build_services(database)
    graph_2 = build_graph(services_2)
    graph_2.invoke(PipelineState(period_start=D - timedelta(hours=12), period_end=D))

    conn = database.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM digests").fetchone()[0] == 1
    finally:
        conn.close()

    # 9. Partial failure does not destroy successful outputs.
    #    Simulate the report agent failing on a second, independent run against
    #    a fresh database — the already-computed Event must survive regardless.
    from research_assistant.llm.errors import LLMUnavailableError

    database_3 = Database(tmp_path / "e2e_partial.db")
    database_3.initialize()
    config = get_config()
    services_3 = PipelineServices(
        config=config, retrieval_service=_FakeRetrievalService(), normalization_service=NormalizationService(),
        agents={AgentName.PEER: PeerAgent(_FakeLLMClient([_relevant_assessment()]), config)},
        event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([LLMUnavailableError("Ollama down")])),
        digest_service=DigestService(config, clock=FixedClock(D)), clock=FixedClock(D),
        persistence_service=PersistenceService(database_3),
    )
    result_3 = build_graph(services_3).invoke(PipelineState(period_start=D - timedelta(hours=12), period_end=D))

    assert len(result_3["events"]) == 1  # the event itself still exists
    assert len(result_3["reports"]) == 0
    assert len(result_3["report_failures"]) == 1
    conn = database_3.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1  # persisted despite the report failure
    finally:
        conn.close()
