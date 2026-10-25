"""Pipeline persistence integration: the persistence node works, the graph
still constructs, an end-to-end mocked pipeline persists its results, and
repeated execution is idempotent."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.report import ReportAgent
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
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
from research_assistant.storage import Database, PersistenceService
from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition

from tests.pipeline.conftest import FakeRetrievalService, jpmorgan_source, real_normalization_service
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


class _PeerFakeLLMClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_structured(self, *, system, prompt, schema):
        self.calls.append((system, prompt, schema))
        return self._responses.pop(0)


def _relevant_assessment():
    return AgentAssessment(
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="Concrete GenAI launch."),
        category=TopicCategory.ENTERPRISE_AI,
        summary="JPMorgan launched a GenAI copilot for advisors.",
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="x"),
    )


def _jpmorgan_article() -> ExtractedArticle:
    return ExtractedArticle(
        url="https://jpmorganchase.com/a", title="JPMorgan launches AI copilot",
        text="JPMorgan Chase today launched a new generative AI copilot for financial advisors.",
        published_at=D,
    )


def _build_services(config, database):
    return PipelineServices(
        config=config,
        retrieval_service=FakeRetrievalService(sources=[jpmorgan_source()], articles_by_source={"JPMorgan Chase": [_jpmorgan_article()]}),
        normalization_service=real_normalization_service(),
        agents={AgentName.PEER: PeerAgent(_PeerFakeLLMClient([_relevant_assessment()]), config)},
        event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
        digest_service=DigestService(config, clock=FixedClock(D)),
        clock=FixedClock(D),
        persistence_service=PersistenceService(database),
    )


def test_graph_still_constructs_with_persistence_service_attached(config, tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    services = _build_services(config, database)

    graph = build_graph(services)

    assert graph is not None


def test_persistence_node_is_a_no_op_when_persistence_service_is_none(config, tmp_path: Path):
    """Backward compatibility with Phase 6: omitting persistence_service must
    not change any other behavior."""
    services = _build_services(config, Database(tmp_path / "unused.db"))
    services.persistence_service = None
    graph = build_graph(services)

    result = graph.invoke(PipelineState())

    assert result["diagnostics"] == []
    assert len(result["events"]) == 1


def test_end_to_end_mocked_pipeline_persists_its_results(config, tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    services = _build_services(config, database)
    graph = build_graph(services)

    result = graph.invoke(PipelineState())

    assert result["diagnostics"] == []
    conn = database.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM digests").fetchone()[0] == 1
    finally:
        conn.close()


def test_repeated_pipeline_execution_is_idempotent(config, tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    services = _build_services(config, database)
    graph = build_graph(services)

    graph.invoke(PipelineState())
    graph.invoke(PipelineState())  # same fake retrieval data -> same deterministic ids

    conn = database.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 1
    finally:
        conn.close()


def test_persistence_failure_is_recorded_as_a_diagnostic_not_raised(config, tmp_path: Path, monkeypatch):
    database = Database(tmp_path / "test.db")
    database.initialize()
    services = _build_services(config, database)

    def _broken_persist(**kwargs):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(services.persistence_service, "persist_pipeline_result", _broken_persist)
    graph = build_graph(services)

    result = graph.invoke(PipelineState())  # must not raise

    assert len(result["events"]) == 1  # computed output still returned
    assert any(d.stage == "persistence" for d in result["diagnostics"])
