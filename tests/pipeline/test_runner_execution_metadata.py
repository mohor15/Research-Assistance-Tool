"""Phase 14: `PipelineRunResult`'s explicit execution/digest-persistence
metadata -- closing the forensic-diagnosis finding that neither
"scheduled vs manual" nor "why is there no digest" could previously be
answered without guessing (see `models.enums.ExecutionMode` /
`DigestPersistenceStatus`)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, DigestPersistenceStatus, ExecutionMode
from research_assistant.pipeline.nodes import PipelineServices
from research_assistant.pipeline.runner import PipelineRunner, RunStatus
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import ExtractedArticle
from research_assistant.storage.database import Database
from research_assistant.storage.persistence_service import PersistenceService

from tests.pipeline.conftest import FakeRetrievalService, jpmorgan_source, real_normalization_service
from tests.pipeline.test_graph import FakeLLMClient, _relevant_assessment
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def _jpmorgan_article() -> ExtractedArticle:
    return ExtractedArticle(
        url="https://jpmorganchase.com/a", title="JPMorgan launches AI copilot",
        text="JPMorgan Chase today launched a new generative AI copilot for financial advisors.",
        published_at=D - timedelta(hours=1),
    )


def _build_runner(config, tmp_path: Path, *, peer_responses, report_responses, persistence_service=None):
    from research_assistant.agents.peer import PeerAgent
    from research_assistant.agents.report import ReportAgent

    if persistence_service is None:
        database = Database(tmp_path / "run_metadata.db")
        database.initialize()
        persistence_service = PersistenceService(database)

    services = PipelineServices(
        config=config,
        retrieval_service=FakeRetrievalService(
            sources=[jpmorgan_source()], articles_by_source={"JPMorgan Chase": [_jpmorgan_article()]},
        ),
        normalization_service=real_normalization_service(),
        agents={AgentName.PEER: PeerAgent(FakeLLMClient(peer_responses), config)},
        event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient(report_responses)),
        digest_service=DigestService(config, clock=FixedClock(D)),
        clock=FixedClock(D),
        persistence_service=persistence_service,
    )
    return PipelineRunner(config, services), persistence_service


def test_execution_mode_defaults_to_manual(config, tmp_path):
    runner, _ = _build_runner(config, tmp_path, peer_responses=[_relevant_assessment()], report_responses=[make_narrative()])
    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)
    assert result.execution_mode == ExecutionMode.MANUAL


def test_execution_mode_scheduled_is_recorded_when_passed(config, tmp_path):
    runner, _ = _build_runner(config, tmp_path, peer_responses=[_relevant_assessment()], report_responses=[make_narrative()])
    result = runner.run(
        digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D,
        execution_mode=ExecutionMode.SCHEDULED,
    )
    assert result.execution_mode == ExecutionMode.SCHEDULED


def test_period_start_and_end_are_recorded_on_the_result(config, tmp_path):
    runner, _ = _build_runner(config, tmp_path, peer_responses=[_relevant_assessment()], report_responses=[make_narrative()])
    start, end = D - timedelta(hours=12), D
    result = runner.run(digest_type=DigestType.MORNING, period_start=start, period_end=end)
    assert result.period_start == start
    assert result.period_end == end


def test_digest_persisted_with_events_is_status_persisted(config, tmp_path):
    runner, _ = _build_runner(config, tmp_path, peer_responses=[_relevant_assessment()], report_responses=[make_narrative()])
    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.digest_attempted is True
    assert result.digest_persisted is True
    assert result.digest_persistence_status == DigestPersistenceStatus.PERSISTED
    assert result.digest_id is not None


def test_digest_persisted_with_zero_events_is_status_no_qualifying_events(config, tmp_path):
    from research_assistant.models.research_item import RelevanceDecision, RelevanceLabel

    irrelevant = _relevant_assessment(relevance=RelevanceDecision(label=RelevanceLabel.IRRELEVANT, reason="no AI angle"))
    runner, _ = _build_runner(config, tmp_path, peer_responses=[irrelevant], report_responses=[])

    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.counts["events_formed"] == 0
    assert result.digest_attempted is True
    assert result.digest_persisted is True  # an empty digest IS still persisted, per existing architecture
    assert result.digest_persistence_status == DigestPersistenceStatus.NO_QUALIFYING_EVENTS


def test_digest_persistence_failure_is_recorded_as_persistence_failed(config, tmp_path):
    class _ExplodingPersistenceService:
        def persist_pipeline_result(self, **kwargs):
            raise RuntimeError("disk full")

    runner, _ = _build_runner(
        config, tmp_path, peer_responses=[_relevant_assessment()], report_responses=[make_narrative()],
        persistence_service=_ExplodingPersistenceService(),
    )

    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.digest_attempted is True  # the digest WAS built in-memory
    assert result.digest_persisted is False
    assert result.digest_persistence_status == DigestPersistenceStatus.PERSISTENCE_FAILED
    assert any(d.stage == "persistence" for d in result.diagnostics)


def test_graph_level_failure_before_digest_generation_is_not_attempted(config, tmp_path):
    class ExplodingRetrievalService:
        def list_sources(self):
            raise RuntimeError("catastrophic failure")

    runner, _ = _build_runner(config, tmp_path, peer_responses=[], report_responses=[])
    runner.services.retrieval_service = ExplodingRetrievalService()  # swap in a broken service post-construction

    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.status == RunStatus.FAILED
    assert result.digest_attempted is False
    assert result.digest_persisted is False
    assert result.digest_persistence_status == DigestPersistenceStatus.NOT_ATTEMPTED


def test_no_persistence_service_configured_is_not_attempted(config, tmp_path):
    from research_assistant.agents.peer import PeerAgent
    from research_assistant.agents.report import ReportAgent

    services = PipelineServices(
        config=config,
        retrieval_service=FakeRetrievalService(
            sources=[jpmorgan_source()], articles_by_source={"JPMorgan Chase": [_jpmorgan_article()]},
        ),
        normalization_service=real_normalization_service(),
        agents={AgentName.PEER: PeerAgent(FakeLLMClient([_relevant_assessment()]), config)},
        event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
        digest_service=DigestService(config, clock=FixedClock(D)),
        clock=FixedClock(D),
        # persistence_service intentionally omitted -- defaults to None
    )
    runner = PipelineRunner(config, services)

    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.digest_persisted is False
    assert result.digest_persistence_status == DigestPersistenceStatus.NOT_ATTEMPTED
