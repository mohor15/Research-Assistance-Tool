"""LangGraph orchestration: the 12 required scenarios. Every service is a
fake/scripted stand-in — no network, no live Ollama, no live retrieval."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from research_assistant.agents.ai_product import AIProductAgent
from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.report import ReportAgent
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.llm.errors import LLMUnavailableError
from research_assistant.models.enums import AgentName, SourcePriority, SourceType
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
)
from research_assistant.pipeline import PipelineServices, PipelineState, build_graph
from research_assistant.pipeline.state import PipelineDiagnostic
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition

from tests.pipeline.conftest import D, FakeRetrievalService, jpmorgan_source, real_normalization_service
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative


def _relevant_assessment(**overrides):
    kwargs = dict(
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="Concrete GenAI launch."),
        category=TopicCategory.ENTERPRISE_AI,
        summary="JPMorgan launched a GenAI copilot for advisors.",
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="x"),
    )
    kwargs.update(overrides)
    return AgentAssessment(**kwargs)


def _jpmorgan_article() -> ExtractedArticle:
    # Phase 13F: published strictly before `D` (not exactly at it) -- a
    # digest's period is now a half-open interval [period_start, period_end),
    # and `D` is also the fixed clock's "now"/`period_end` in these tests,
    # so an article dated exactly at `D` would fall on the excluded boundary.
    return ExtractedArticle(
        url="https://jpmorganchase.com/a", title="JPMorgan launches AI copilot",
        text="JPMorgan Chase today launched a new generative AI copilot for financial advisors.",
        published_at=D - timedelta(hours=1),
    )


def _build_services(config, *, peer_llm_responses, report_llm_responses, retrieval_service=None, agents=None):
    peer_agent = PeerAgent(FakeLLMClient(peer_llm_responses), config)
    report_agent = ReportAgent(ReportingFakeLLMClient(report_llm_responses))
    return PipelineServices(
        config=config,
        retrieval_service=retrieval_service or FakeRetrievalService(
            sources=[jpmorgan_source()], articles_by_source={"JPMorgan Chase": [_jpmorgan_article()]}
        ),
        normalization_service=real_normalization_service(),
        agents=agents if agents is not None else {AgentName.PEER: peer_agent},
        event_service=EventService(config, clock=FixedClock(D)),
        report_agent=report_agent,
        digest_service=DigestService(config, clock=FixedClock(D)),
        clock=FixedClock(D),
    )


class FakeLLMClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_structured(self, *, system, prompt, schema):
        self.calls.append((system, prompt, schema))
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_graph_construction_succeeds(config):
    services = _build_services(config, peer_llm_responses=[], report_llm_responses=[])
    graph = build_graph(services)
    assert graph is not None


def test_graph_state_accepts_valid_construction_and_rejects_invalid():
    import pytest
    from pydantic import ValidationError

    state = PipelineState()
    assert state.articles == []
    assert state.digest is None

    with pytest.raises(ValidationError):
        PipelineState(digest_type="not-a-real-type")


def test_successful_end_to_end_execution_with_mocked_services(config):
    services = _build_services(
        config, peer_llm_responses=[_relevant_assessment()], report_llm_responses=[make_narrative()]
    )
    graph = build_graph(services)

    result = graph.invoke(PipelineState())

    assert len(result["sources"]) == 1
    assert len(result["articles"]) == 1
    assert len(result["evidence"]) == 1
    assert len(result["research_items"]) == 1
    assert len(result["events"]) == 1
    assert len(result["reports"]) == 1
    assert result["digest"] is not None
    assert result["digest"].events == [result["events"][0].id]
    assert result["diagnostics"] == []


def test_retrieval_failure_for_one_source_does_not_abort_the_pipeline(config):
    sources = [
        jpmorgan_source(),
        SourceDefinition(
            name="Broken Source", domain="broken.example.com", urls=["https://broken.example.com/news"],
            source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
        ),
    ]
    retrieval_service = FakeRetrievalService(
        sources=sources,
        articles_by_source={"JPMorgan Chase": [_jpmorgan_article()]},
        failing_sources={"Broken Source"},
    )
    services = _build_services(
        config, peer_llm_responses=[_relevant_assessment()], report_llm_responses=[make_narrative()],
        retrieval_service=retrieval_service,
    )
    graph = build_graph(services)

    result = graph.invoke(PipelineState())

    assert len(result["articles"]) == 1  # JPMorgan's article still made it through
    assert any("Broken Source" in d.message for d in result["diagnostics"])
    assert any(d.stage == "retrieve" for d in result["diagnostics"])


def test_agent_failure_for_one_item_does_not_abort_the_pipeline(config):
    services = _build_services(
        config,
        peer_llm_responses=[LLMUnavailableError("Ollama not running")],
        report_llm_responses=[],
    )
    graph = build_graph(services)

    result = graph.invoke(PipelineState())

    assert result["research_items"] == []  # the one article's agent call failed
    assert result["events"] == []
    assert any(d.stage == "agent_analysis" for d in result["diagnostics"])


def test_report_failure_for_one_event_does_not_abort_the_pipeline(config):
    services = _build_services(
        config,
        peer_llm_responses=[_relevant_assessment()],
        report_llm_responses=[LLMUnavailableError("Ollama not running")],
    )
    graph = build_graph(services)

    result = graph.invoke(PipelineState())

    assert len(result["events"]) == 1
    assert result["reports"] == []
    assert len(result["report_failures"]) == 1
    assert result["report_failures"][0].event_id == result["events"][0].id
    assert result["digest"] is not None  # digest generation still runs


def test_partial_failure_across_multiple_sources_processes_what_succeeds(config):
    other_source = SourceDefinition(
        name="OpenAI", domain="openai.com", urls=["https://openai.com/news"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.AI_PROVIDER,
    )
    retrieval_service = FakeRetrievalService(
        sources=[jpmorgan_source(), other_source],
        articles_by_source={
            "JPMorgan Chase": [_jpmorgan_article()],
            "OpenAI": [ExtractedArticle(
                url="https://openai.com/a", title="OpenAI releases new model",
                text="OpenAI today released a new generative AI foundation model for enterprise customers.",
                published_at=D,
            )],
        },
    )
    # AIProductAgent's own LLM call raises an unexpected (non-LLMError)
    # exception — the orchestration layer must catch this too, not just the
    # anticipated LLMError subtypes, and continue with the rest of the batch.
    ai_product_agent = AIProductAgent(FakeLLMClient([RuntimeError("unexpected agent failure")]), config)
    peer_agent = PeerAgent(FakeLLMClient([_relevant_assessment()]), config)
    services = PipelineServices(
        config=config, retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.PEER: peer_agent, AgentName.AI_PRODUCT: ai_product_agent},
        event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
        digest_service=DigestService(config, clock=FixedClock(D)), clock=FixedClock(D),
    )
    graph = build_graph(services)

    result = graph.invoke(PipelineState())

    # JPMorgan succeeded; OpenAI's agent call raised an unexpected exception
    # that was caught per-item — no crash, and the pipeline still yields
    # exactly the JPMorgan event/report plus a diagnostic for the failure.
    assert len(result["events"]) == 1
    assert result["events"][0].primary_company == "JPMorgan Chase"
    assert any("unexpected agent failure" in d.message for d in result["diagnostics"])


def test_disabled_agent_is_never_invoked(config):
    modified_config = config.model_copy(deep=True)
    modified_config.settings.agents.peer = modified_config.settings.agents.peer.model_copy(update={"enabled": False})

    peer_llm = FakeLLMClient([])  # would raise AssertionError if ever called
    services = _build_services(modified_config, peer_llm_responses=[], report_llm_responses=[])
    services.agents[AgentName.PEER] = PeerAgent(peer_llm, modified_config)

    graph = build_graph(services)
    result = graph.invoke(PipelineState())

    assert result["research_items"] == []
    assert len(peer_llm.calls) == 0  # disabled -> never even attempted


def test_top_k_configuration_limits_processed_articles_and_records_a_diagnostic(config):
    modified_config = config.model_copy(deep=True)
    modified_config.settings.agents.peer = modified_config.settings.agents.peer.model_copy(update={"top_k": 1})

    articles = [
        ExtractedArticle(url=f"https://jpmorganchase.com/{i}", title=f"Story {i}", text=f"GenAI story number {i} about AI.", published_at=D)
        for i in range(3)
    ]
    retrieval_service = FakeRetrievalService(sources=[jpmorgan_source()], articles_by_source={"JPMorgan Chase": articles})
    services = _build_services(
        modified_config, peer_llm_responses=[_relevant_assessment()], report_llm_responses=[make_narrative()],
        retrieval_service=retrieval_service,
    )
    graph = build_graph(services)

    result = graph.invoke(PipelineState())

    assert len(result["articles"]) == 3  # all retrieved/normalized...
    assert len(result["research_items"]) == 1  # ...but only top_k=1 actually analyzed
    assert any("top_k=1" in d.message for d in result["diagnostics"])


def test_graph_holds_no_hidden_global_state(config):
    """Two independently-built graphs from independently-built services never
    share state — confirms PipelineServices/state are pure DI, not module globals."""
    services_a = _build_services(config, peer_llm_responses=[_relevant_assessment()], report_llm_responses=[make_narrative()])
    services_b = _build_services(config, peer_llm_responses=[_relevant_assessment()], report_llm_responses=[make_narrative()])

    graph_a = build_graph(services_a)
    graph_b = build_graph(services_b)

    result_a = graph_a.invoke(PipelineState())
    result_b = graph_b.invoke(PipelineState())

    assert result_a["events"][0].id == result_b["events"][0].id  # same input -> same deterministic output
    assert services_a is not services_b


def test_nodes_module_contains_no_retrieval_or_llm_http_logic():
    import research_assistant.pipeline.nodes as nodes_module

    source = Path(nodes_module.__file__).read_text(encoding="utf-8")
    assert "requests.get" not in source
    assert "requests.post" not in source
    assert "BeautifulSoup" not in source
    assert "feedparser" not in source


def test_event_scoring_node_output_matches_direct_event_service_call(config):
    services = _build_services(config, peer_llm_responses=[_relevant_assessment()], report_llm_responses=[make_narrative()])
    graph = build_graph(services)

    graph_result = graph.invoke(PipelineState())

    direct_events = services.event_service.build_events(graph_result["research_items"])

    assert [e.id for e in graph_result["events"]] == [e.id for e in direct_events]
    assert [e.relevance_score for e in graph_result["events"]] == [e.relevance_score for e in direct_events]
    assert [e.confidence_score for e in graph_result["events"]] == [e.confidence_score for e in direct_events]
