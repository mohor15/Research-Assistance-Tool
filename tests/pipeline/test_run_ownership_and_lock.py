"""Phase 17: explicit CURRENT run ownership (Section 4) and the
application-level CURRENT concurrency guard (Section 3). Fixtures/fakes
only -- no live network, no Ollama, no real retrieval."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research_assistant.agents.financial_news import FinancialNewsAgent
from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.report import ReportAgent
from research_assistant.config import get_config
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.evidence.service import NormalizationService
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, SourcePriority, SourceType
from research_assistant.pipeline.nodes import PipelineServices
from research_assistant.pipeline.runner import CurrentRunAlreadyInProgressError, PipelineRunner, RunStatus
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition
from research_assistant.storage.database import Database
from research_assistant.storage.persistence_service import PersistenceService

from tests.pipeline.conftest import FakeRetrievalService, FakeTriageLLMClient, jpmorgan_source, real_normalization_service
from tests.pipeline.test_graph import FakeLLMClient, _relevant_assessment
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative

CONFIG = get_config()
D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def persistence_service(tmp_path: Path) -> PersistenceService:
    database = Database(tmp_path / "test.db")
    database.initialize()
    return PersistenceService(database)


def _jpmorgan_article() -> ExtractedArticle:
    return ExtractedArticle(
        url="https://jpmorganchase.com/a", title="JPMorgan launches AI copilot",
        text="JPMorgan Chase today launched a new generative AI copilot for financial advisors.",
        published_at=D,
    )


def _morning_runner(persistence_service, *, now=D):
    services = PipelineServices(
        config=CONFIG,
        retrieval_service=FakeRetrievalService(
            sources=[jpmorgan_source()], articles_by_source={"JPMorgan Chase": [_jpmorgan_article()]},
        ),
        normalization_service=real_normalization_service(),
        agents={AgentName.PEER: PeerAgent(FakeLLMClient([_relevant_assessment()]), CONFIG)},
        event_service=EventService(CONFIG, clock=FixedClock(now)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
        digest_service=DigestService(CONFIG, clock=FixedClock(now)),
        clock=FixedClock(now),
        persistence_service=persistence_service,
    )
    return PipelineRunner(CONFIG, services)


def _ft_source():
    # A generic, real secondary source with no `research_domain` override
    # (agents/router.py) -- routes to FinancialNewsAgent exactly like
    # "Financial Times" used to before FT gained its own AI/Technology
    # specialist route; this test is about run ownership/locking
    # generically, not FT specifically.
    return SourceDefinition(
        name="Reuters", domain="reuters.com", urls=["https://www.reuters.com/technology/artificial-intelligence/"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
    )


def _current_article(published_at):
    return ExtractedArticle(
        url="https://reuters.com/ai-story", title="Bank announces new enterprise AI partnership",
        text=(
            "A major bank today announced a new enterprise AI partnership and deployed a generative AI "
            "platform across its trading and risk operations, in the latest step of its ongoing AI rollout. "
        ) * 3,
        published_at=published_at,
    )


def _current_runner(persistence_service, *, period_start, period_end):
    agent = FinancialNewsAgent(FakeLLMClient([]), CONFIG)
    services = PipelineServices(
        config=CONFIG,
        retrieval_service=FakeRetrievalService(
            sources=[_ft_source()], articles_by_source={"Reuters": [_current_article(period_start + timedelta(hours=1))]},
        ),
        normalization_service=real_normalization_service(),
        agents={AgentName.FINANCIAL_NEWS: agent},
        event_service=EventService(CONFIG, clock=FixedClock(period_end)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()] * 3)),
        digest_service=DigestService(CONFIG, clock=FixedClock(period_end)),
        clock=FixedClock(period_end),
        persistence_service=persistence_service,
        triage_llm_client=FakeTriageLLMClient(),
        deep_analysis_llm_client=FakeLLMClient([_relevant_assessment(is_dated_development=True)]),
    )
    return PipelineRunner(CONFIG, services)


# ---------------------------------------------------------------------------
# Run ownership (Section 4/20): outputs from run A cannot appear in run B
# ---------------------------------------------------------------------------


def test_events_and_research_items_are_stamped_with_their_own_run_id(persistence_service):
    runner = _current_runner(persistence_service, period_start=D - timedelta(hours=24), period_end=D)

    result = runner.run(digest_type=DigestType.CURRENT, period_start=D - timedelta(hours=24), period_end=D)

    conn = persistence_service.database.connect()
    try:
        owned_events = persistence_service.events.get_by_run_id(conn, result.run_id)
        owned_items = persistence_service.research_items.get_by_run_id(conn, result.run_id)
    finally:
        conn.close()

    assert len(owned_events) == 1
    assert owned_events[0].run_id == result.run_id
    assert len(owned_items) == 1
    assert owned_items[0].run_id == result.run_id


def test_a_different_runs_events_are_never_returned_by_get_by_run_id(persistence_service):
    runner_a = _current_runner(persistence_service, period_start=D - timedelta(hours=24), period_end=D)
    result_a = runner_a.run(digest_type=DigestType.CURRENT, period_start=D - timedelta(hours=24), period_end=D)

    later = D + timedelta(days=1)
    runner_b = _current_runner(persistence_service, period_start=later - timedelta(hours=24), period_end=later)
    result_b = runner_b.run(digest_type=DigestType.CURRENT, period_start=later - timedelta(hours=24), period_end=later)

    conn = persistence_service.database.connect()
    try:
        owned_by_a = {e.id for e in persistence_service.events.get_by_run_id(conn, result_a.run_id)}
        owned_by_b = {e.id for e in persistence_service.events.get_by_run_id(conn, result_b.run_id)}
    finally:
        conn.close()

    assert owned_by_a.isdisjoint(owned_by_b)
    assert result_a.run_id != result_b.run_id


def test_a_run_with_no_persistence_service_leaves_run_id_none(persistence_service):
    """Every pre-existing unit test invokes the graph directly (no
    PipelineRunner, no run_id) -- confirms ResearchItem.run_id/Event.run_id
    default to None rather than crashing, preserving backward
    compatibility with the entire existing test suite."""
    from research_assistant.pipeline import PipelineState, build_graph

    services_without_run_id = PipelineServices(
        config=CONFIG,
        retrieval_service=FakeRetrievalService(
            sources=[jpmorgan_source()], articles_by_source={"JPMorgan Chase": [_jpmorgan_article()]},
        ),
        normalization_service=real_normalization_service(),
        agents={AgentName.PEER: PeerAgent(FakeLLMClient([_relevant_assessment()]), CONFIG)},
        event_service=EventService(CONFIG, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
        digest_service=DigestService(CONFIG, clock=FixedClock(D)),
        clock=FixedClock(D),
    )
    result = build_graph(services_without_run_id).invoke(PipelineState())
    assert result["events"][0].run_id is None
    assert result["research_items"][0].run_id is None


# ---------------------------------------------------------------------------
# Concurrency guard (Section 3/20)
# ---------------------------------------------------------------------------


def test_second_current_run_is_rejected_while_first_is_in_progress(persistence_service):
    conn = persistence_service.database.connect()
    try:
        persistence_service.current_run_lock.try_acquire(conn, run_id="already-running-run", started_at=D)
    finally:
        conn.close()

    runner = _current_runner(persistence_service, period_start=D - timedelta(hours=24), period_end=D)

    with pytest.raises(CurrentRunAlreadyInProgressError) as excinfo:
        runner.run(digest_type=DigestType.CURRENT, period_start=D - timedelta(hours=24), period_end=D)
    assert excinfo.value.run_id == "already-running-run"

    # The rejected run must never have touched the pipeline at all.
    conn = persistence_service.database.connect()
    try:
        assert persistence_service.runs.get(conn, excinfo.value.run_id) is None
    finally:
        conn.close()


def test_lock_is_released_after_a_successful_current_run_so_a_later_one_can_proceed(persistence_service):
    runner = _current_runner(persistence_service, period_start=D - timedelta(hours=24), period_end=D)
    result = runner.run(digest_type=DigestType.CURRENT, period_start=D - timedelta(hours=24), period_end=D)
    assert result.status in (RunStatus.SUCCESS, RunStatus.PARTIAL_SUCCESS)

    conn = persistence_service.database.connect()
    try:
        assert persistence_service.current_run_lock.get(conn) is None
    finally:
        conn.close()

    # A second CURRENT run (different window) is free to proceed.
    later = D + timedelta(days=1)
    runner_2 = _current_runner(persistence_service, period_start=later - timedelta(hours=24), period_end=later)
    result_2 = runner_2.run(digest_type=DigestType.CURRENT, period_start=later - timedelta(hours=24), period_end=later)
    assert result_2.status in (RunStatus.SUCCESS, RunStatus.PARTIAL_SUCCESS)


def test_current_lock_never_blocks_morning_or_evening(persistence_service):
    """The guard is CURRENT-only -- a CURRENT lock held by an in-progress
    scan must never block a scheduled Morning/Evening run."""
    conn = persistence_service.database.connect()
    try:
        persistence_service.current_run_lock.try_acquire(conn, run_id="some-current-run", started_at=D)
    finally:
        conn.close()

    morning_runner = _morning_runner(persistence_service)
    result = morning_runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.status == RunStatus.SUCCESS


class _ExplodingRetrievalService:
    """A genuine unhandled-exception path (mirrors
    `test_runner.py::test_status_is_failed_when_the_graph_raises_an_unhandled_exception`):
    raises before any node can catch/isolate it, so it reaches `run()`'s
    `except Exception` branch rather than being absorbed as a per-source
    diagnostic."""

    def list_sources(self):
        raise RuntimeError("catastrophic failure")


def test_lock_is_released_when_the_run_raises_an_unhandled_exception(persistence_service):
    """A genuine pipeline-stage exception (not just an isolated-source
    diagnostic) must still release the lock -- `run()`'s `finally` must
    fire even on the FAILED path, not only on SUCCESS/PARTIAL_SUCCESS."""
    runner = _current_runner(persistence_service, period_start=D - timedelta(hours=24), period_end=D)
    runner.services.retrieval_service = _ExplodingRetrievalService()

    result = runner.run(digest_type=DigestType.CURRENT, period_start=D - timedelta(hours=24), period_end=D)

    assert result.status == RunStatus.FAILED

    conn = persistence_service.database.connect()
    try:
        assert persistence_service.current_run_lock.get(conn) is None
    finally:
        conn.close()

    # A later CURRENT run (same or different window) must be free to
    # proceed -- the failure must not have left a permanent lock.
    later = D + timedelta(days=1)
    runner_2 = _current_runner(persistence_service, period_start=later - timedelta(hours=24), period_end=later)
    # Must not raise CurrentRunAlreadyInProgressError -- the prior run's
    # failure released the lock, so this one is free to proceed.
    result_2 = runner_2.run(digest_type=DigestType.CURRENT, period_start=later - timedelta(hours=24), period_end=later)
    assert result_2.status in (RunStatus.SUCCESS, RunStatus.PARTIAL_SUCCESS, RunStatus.FAILED)
    conn = persistence_service.database.connect()
    try:
        assert persistence_service.current_run_lock.get(conn) is None
    finally:
        conn.close()


class _RaisingTriageLLMClient:
    """Simulates an unreachable Ollama triage endpoint mid-run. CURRENT's
    triage stage isolates a batch failure as a `warning`-severity
    diagnostic rather than letting it propagate (see
    `make_agent_analysis_node`), so this run still completes -- it must
    still release the lock."""

    def generate_structured(self, *, system: str, prompt: str, schema):
        raise ConnectionError("simulated: could not reach Ollama")


def test_lock_is_released_after_an_isolated_ollama_triage_failure(persistence_service):
    runner = _current_runner(persistence_service, period_start=D - timedelta(hours=24), period_end=D)
    runner.services.triage_llm_client = _RaisingTriageLLMClient()

    result = runner.run(digest_type=DigestType.CURRENT, period_start=D - timedelta(hours=24), period_end=D)
    assert result.status != RunStatus.FAILED

    conn = persistence_service.database.connect()
    try:
        assert persistence_service.current_run_lock.get(conn) is None
    finally:
        conn.close()
