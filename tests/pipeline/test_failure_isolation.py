"""Phase 8 Part 7 — dedicated failure-isolation verification: one source
failing, one article's extraction/normalization failing, one agent call
failing, one report generation failing, and one persistence write failing
must each leave the rest of the run's successful output intact, with a
structured diagnostic recorded rather than a silent loss or a full abort."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.evidence.service import NormalizationService
from research_assistant.models.enums import AgentName, SourcePriority, SourceType
from research_assistant.pipeline import PipelineServices, PipelineState, build_graph
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition

from tests.pipeline.conftest import FakeRetrievalService, jpmorgan_source, real_normalization_service
from tests.pipeline.test_graph import FakeLLMClient, _relevant_assessment
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def _jpmorgan_article() -> ExtractedArticle:
    return ExtractedArticle(
        url="https://jpmorganchase.com/a", title="JPMorgan launches AI copilot",
        text="JPMorgan Chase today launched a new generative AI copilot for financial advisors.",
        published_at=D,
    )


def _openai_article() -> ExtractedArticle:
    return ExtractedArticle(
        url="https://openai.com/a", title="OpenAI releases a new model",
        text="OpenAI released a new generative AI foundation model for enterprise customers today.",
        published_at=D,
    )


def _services(*, retrieval_service, normalization_service, agents, report_agent):
    from research_assistant.config import get_config

    config = get_config()
    return PipelineServices(
        config=config, retrieval_service=retrieval_service, normalization_service=normalization_service,
        agents=agents, event_service=EventService(config, clock=FixedClock(D)), report_agent=report_agent,
        digest_service=DigestService(config, clock=FixedClock(D)), clock=FixedClock(D),
    )


def test_one_source_failing_does_not_stop_other_sources_from_being_processed():
    from research_assistant.agents.peer import PeerAgent
    from research_assistant.agents.report import ReportAgent
    from research_assistant.config import get_config

    config = get_config()
    other_source = SourceDefinition(
        name="Broken Source", domain="broken.example.com", urls=["https://broken.example.com/news"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )
    retrieval_service = FakeRetrievalService(
        sources=[jpmorgan_source(), other_source],
        articles_by_source={"JPMorgan Chase": [_jpmorgan_article()]},
        failing_sources={"Broken Source"},
    )
    services = _services(
        retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.PEER: PeerAgent(FakeLLMClient([_relevant_assessment()]), config)},
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
    )

    result = build_graph(services).invoke(PipelineState())

    assert len(result["events"]) == 1
    assert result["events"][0].primary_company == "JPMorgan Chase"
    assert any(d.stage == "retrieve" and "Broken Source" in d.message for d in result["diagnostics"])


def test_one_malformed_feed_or_extraction_failure_does_not_stop_other_articles():
    """A source returning content that fails normalization (e.g. an
    extraction produced no usable text) must not block other articles."""
    from research_assistant.agents.peer import PeerAgent
    from research_assistant.agents.report import ReportAgent
    from research_assistant.config import get_config

    config = get_config()

    class _NormalizationServiceThatFailsOnce:
        def __init__(self, inner):
            self._inner = inner
            self._first_call = True

        def normalize(self, extracted, *, source):
            if self._first_call:
                self._first_call = False
                raise ValueError("simulated malformed extraction")
            return self._inner.normalize(extracted, source=source)

        def deduplicate(self, articles):
            return self._inner.deduplicate(articles)

        def to_evidence_many(self, articles):
            return self._inner.to_evidence_many(articles)

    retrieval_service = FakeRetrievalService(
        sources=[jpmorgan_source()], articles_by_source={"JPMorgan Chase": [_jpmorgan_article(), _jpmorgan_article()]},
    )
    services = _services(
        retrieval_service=retrieval_service,
        normalization_service=_NormalizationServiceThatFailsOnce(real_normalization_service()),
        agents={AgentName.PEER: PeerAgent(FakeLLMClient([_relevant_assessment()]), config)},
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
    )

    result = build_graph(services).invoke(PipelineState())

    assert any(d.stage == "normalize" for d in result["diagnostics"])
    assert len(result["events"]) == 1  # the second (identical) article still made it through


def test_one_agent_call_failing_does_not_stop_other_agents_or_articles():
    from research_assistant.agents.ai_product import AIProductAgent
    from research_assistant.agents.peer import PeerAgent
    from research_assistant.agents.report import ReportAgent
    from research_assistant.config import get_config

    config = get_config()
    ai_provider_source = SourceDefinition(
        name="OpenAI", domain="openai.com", urls=["https://openai.com/news"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.AI_PROVIDER,
    )
    retrieval_service = FakeRetrievalService(
        sources=[jpmorgan_source(), ai_provider_source],
        articles_by_source={"JPMorgan Chase": [_jpmorgan_article()], "OpenAI": [_openai_article()]},
    )
    services = _services(
        retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={
            AgentName.PEER: PeerAgent(FakeLLMClient([_relevant_assessment()]), config),
            AgentName.AI_PRODUCT: AIProductAgent(FakeLLMClient([RuntimeError("agent blew up")]), config),
        },
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
    )

    result = build_graph(services).invoke(PipelineState())

    assert len(result["events"]) == 1
    assert result["events"][0].primary_company == "JPMorgan Chase"
    assert any(d.stage == "agent_analysis" for d in result["diagnostics"])


def test_one_report_generation_failure_does_not_stop_other_reports():
    from research_assistant.agents.ai_product import AIProductAgent
    from research_assistant.agents.peer import PeerAgent
    from research_assistant.agents.report import ReportAgent
    from research_assistant.config import get_config
    from research_assistant.llm.errors import LLMUnavailableError

    config = get_config()
    ai_provider_source = SourceDefinition(
        name="OpenAI", domain="openai.com", urls=["https://openai.com/news"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.AI_PROVIDER,
    )
    retrieval_service = FakeRetrievalService(
        sources=[jpmorgan_source(), ai_provider_source],
        articles_by_source={"JPMorgan Chase": [_jpmorgan_article()], "OpenAI": [_openai_article()]},
    )
    services = _services(
        retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={
            AgentName.PEER: PeerAgent(FakeLLMClient([_relevant_assessment()]), config),
            AgentName.AI_PRODUCT: AIProductAgent(FakeLLMClient([_relevant_assessment()]), config),
        },
        report_agent=ReportAgent(ReportingFakeLLMClient([LLMUnavailableError("Ollama down"), make_narrative()])),
    )

    result = build_graph(services).invoke(PipelineState())

    assert len(result["events"]) == 2
    assert len(result["reports"]) == 1
    assert len(result["report_failures"]) == 1
    # The digest still gets generated even though one report failed.
    assert result["digest"] is not None


def test_persistence_write_failure_rolls_back_without_losing_computed_results(monkeypatch):
    """A SQLite write failure must not lose the Events/Reports/Digest already
    computed in state, and the underlying transaction must roll back rather
    than leaving a half-written database (see storage.persistence_service's
    own rollback test for the transaction-level guarantee; this test
    verifies the pipeline layer's response to that failure)."""
    from research_assistant.agents.peer import PeerAgent
    from research_assistant.agents.report import ReportAgent
    from research_assistant.config import get_config

    config = get_config()
    retrieval_service = FakeRetrievalService(sources=[jpmorgan_source()], articles_by_source={"JPMorgan Chase": [_jpmorgan_article()]})
    services = _services(
        retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.PEER: PeerAgent(FakeLLMClient([_relevant_assessment()]), config)},
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
    )

    class _BrokenPersistenceService:
        def persist_pipeline_result(self, **kwargs):
            raise RuntimeError("simulated disk failure")

    services.persistence_service = _BrokenPersistenceService()

    result = build_graph(services).invoke(PipelineState())

    assert len(result["events"]) == 1  # still returned despite persistence failing
    assert result["reports"][0].title
    assert any(d.stage == "persistence" for d in result["diagnostics"])


def test_a_whole_source_fetch_failure_produces_a_retrieve_diagnostic_end_to_end():
    """Phase 1 fix, proven through the REAL discovery/service layers (not
    the FakeRetrievalService used by the other tests in this file): when
    every one of a source's configured URLs fails to fetch,
    discovery.discover_candidates() raises SourceUnreachableError, and
    pipeline.nodes.retrieve() turns that into a `stage="retrieve"`
    diagnostic with the `"{source_name}: {reason}"` prefix
    dashboard_data.py's `_latest_retrieve_failures()` parses (via
    `partition(": ")`) -- an honest-classification label (Part A1) is
    appended AFTER that prefix, never before it, so the parsed name is
    never polluted."""
    from research_assistant.agents.peer import PeerAgent
    from research_assistant.agents.report import ReportAgent
    from research_assistant.config import get_config
    from research_assistant.retrieval.client import HTTPClient
    from research_assistant.retrieval.service import RetrievalService

    from tests.retrieval.fakes import FakeResponse, FakeSession

    config = get_config()
    config = config.model_copy(deep=True)
    # This test's own purpose is the SourceUnreachableError -> diagnostic
    # path, not the Part A2 browser-fallback path (covered separately in
    # tests/retrieval/test_service.py) -- disabled here so a plain HTTP 500
    # resolves immediately without a real Playwright launch/network call.
    config.settings.retrieval.browser_fallback = False
    broken_source = SourceDefinition(
        name="Morgan Stanley", domain="morganstanley.com", urls=["https://www.morganstanley.com/newsroom"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )

    class _SingleSourceRetrievalService(RetrievalService):
        def list_sources(self, category=None):
            return [broken_source]

    session = FakeSession([FakeResponse(status_code=500, url="https://www.morganstanley.com/newsroom", text="")])
    http_client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    retrieval_service = _SingleSourceRetrievalService(config, http_client=http_client)

    services = _services(
        retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.PEER: PeerAgent(FakeLLMClient([]), config)},
        report_agent=ReportAgent(ReportingFakeLLMClient([])),
    )

    result = build_graph(services).invoke(PipelineState())

    assert result["events"] == []
    retrieve_diagnostics = [d for d in result["diagnostics"] if d.stage == "retrieve"]
    assert len(retrieve_diagnostics) == 1
    # The honest-classification suffix (Part A1) is appended AFTER the
    # "{name}: {reason}" the dashboard parses, never before it -- so the
    # name is still exactly "Morgan Stanley", just with a classification
    # label on the end.
    assert retrieve_diagnostics[0].message == "Morgan Stanley: http_error [HTTP_5XX]"
