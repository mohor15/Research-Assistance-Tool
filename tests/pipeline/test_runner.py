"""PipelineRunner: graph invocation, explicit digest_type/period, dependency
injection, run-id determinism, status determination, and failure/diagnostic
propagation — no live Ollama, no live retrieval, no live SQLite required."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.evidence.service import NormalizationService
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, SourcePriority, SourceType
from research_assistant.pipeline.nodes import PipelineServices
from research_assistant.pipeline.runner import PipelineRunner, RunStatus, generate_run_id
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition

from tests.pipeline.conftest import FakeRetrievalService, jpmorgan_source, real_normalization_service
from tests.pipeline.test_graph import FakeLLMClient, _relevant_assessment
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def _build_runner(config, *, peer_responses, report_responses, retrieval_service=None):
    from research_assistant.agents.peer import PeerAgent
    from research_assistant.agents.report import ReportAgent

    services = PipelineServices(
        config=config,
        retrieval_service=retrieval_service or FakeRetrievalService(
            sources=[jpmorgan_source()], articles_by_source={"JPMorgan Chase": [_jpmorgan_article()]}
        ),
        normalization_service=real_normalization_service(),
        agents={AgentName.PEER: PeerAgent(FakeLLMClient(peer_responses), config)},
        event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient(report_responses)),
        digest_service=DigestService(config, clock=FixedClock(D)),
        clock=FixedClock(D),
    )
    return PipelineRunner(config, services)


def _jpmorgan_article() -> ExtractedArticle:
    return ExtractedArticle(
        url="https://jpmorganchase.com/a", title="JPMorgan launches AI copilot",
        text="JPMorgan Chase today launched a new generative AI copilot for financial advisors.",
        published_at=D,
    )


def test_run_invokes_the_graph_with_explicit_digest_type_and_period(config):
    runner = _build_runner(config, peer_responses=[_relevant_assessment()], report_responses=[make_narrative()])

    result = runner.run(digest_type=DigestType.EVENING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.digest_type == DigestType.EVENING
    assert result.status == RunStatus.SUCCESS
    assert result.counts["events_formed"] == 1


def test_run_id_is_deterministic_for_the_same_digest_type_and_period(config):
    runner = _build_runner(config, peer_responses=[_relevant_assessment(), _relevant_assessment()], report_responses=[make_narrative(), make_narrative()])

    result_a = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)
    result_b = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result_a.run_id == result_b.run_id
    assert result_a.run_id == generate_run_id(DigestType.MORNING, D - timedelta(hours=12), D)


def test_run_id_differs_for_different_periods(config):
    assert generate_run_id(DigestType.MORNING, D, D) != generate_run_id(DigestType.MORNING, D - timedelta(days=1), D)


def test_run_id_differs_for_different_digest_types():
    assert generate_run_id(DigestType.MORNING, D, D) != generate_run_id(DigestType.EVENING, D, D)


def test_dependency_injection_uses_the_supplied_services_not_real_ones(config):
    runner = _build_runner(config, peer_responses=[_relevant_assessment()], report_responses=[make_narrative()])

    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.counts["sources_discovered"] == 1  # from the injected FakeRetrievalService, not a real one


def test_status_is_success_when_no_error_diagnostics(config):
    runner = _build_runner(config, peer_responses=[_relevant_assessment()], report_responses=[make_narrative()])
    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)
    assert result.status == RunStatus.SUCCESS
    assert result.diagnostics == []


def test_status_is_partial_success_when_an_error_occurs_but_events_still_formed(config):
    other_source = SourceDefinition(
        name="Broken Source", domain="broken.example.com", urls=["https://broken.example.com/news"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )
    retrieval_service = FakeRetrievalService(
        sources=[jpmorgan_source(), other_source],
        articles_by_source={"JPMorgan Chase": [_jpmorgan_article()]},
        failing_sources={"Broken Source"},
    )
    runner = _build_runner(
        config, peer_responses=[_relevant_assessment()], report_responses=[make_narrative()],
        retrieval_service=retrieval_service,
    )

    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.status == RunStatus.PARTIAL_SUCCESS
    assert result.counts["events_formed"] == 1
    assert any(d.stage == "retrieve" for d in result.diagnostics)


def test_status_is_failed_when_the_graph_raises_an_unhandled_exception(config):
    class ExplodingRetrievalService:
        def list_sources(self):
            raise RuntimeError("catastrophic failure")

    runner = _build_runner(
        config, peer_responses=[], report_responses=[], retrieval_service=ExplodingRetrievalService(),
    )

    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.status == RunStatus.FAILED
    assert result.counts == {}
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].severity == "error"


def test_run_result_includes_funnel_counts(config):
    runner = _build_runner(config, peer_responses=[_relevant_assessment()], report_responses=[make_narrative()])
    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    for key in ("sources_discovered", "candidates_discovered", "research_items_relevant", "events_formed", "reports_succeeded"):
        assert key in result.counts


def test_build_classmethod_wires_real_services_from_config():
    """Only checks that .build() constructs without error and wires the
    expected component types — does not invoke the real graph (that would
    require live Ollama/network, out of scope for unit tests)."""
    import tempfile
    from pathlib import Path

    from research_assistant.storage import Database

    db = Database(Path(tempfile.mkdtemp()) / "build_test.db")
    from research_assistant.config import get_config

    runner = PipelineRunner.build(get_config(), database=db)

    assert runner.services.persistence_service is not None
    assert len(runner.services.agents) == 5  # Peer, AIProduct, Startup, Analyst, FinancialNews
