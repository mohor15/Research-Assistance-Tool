"""Status-semantics fix, confirmed via the read-only forensic investigation
of production run `5e317bf0b132dd45`: `_determine_status` must not report
FAILED for a run that completed all stages and correctly persisted a
digest merely because some isolated source(s) had connection errors and
zero events happened to qualify. The only real-failure signal remaining
once the graph has already completed is an actual digest persistence
failure (an actual pipeline-stage failure is a separate code path, the
`except` branch in `PipelineRunner.run`, and never reaches
`_determine_status` at all)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, DigestPersistenceStatus
from research_assistant.models.research_item import RelevanceDecision, RelevanceLabel
from research_assistant.pipeline.nodes import PipelineServices
from research_assistant.pipeline.runner import PipelineRunner, RunStatus, _determine_status
from research_assistant.pipeline.state import PipelineDiagnostic
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition, SourcePriority, SourceType
from research_assistant.storage.database import Database
from research_assistant.storage.persistence_service import PersistenceService

from tests.pipeline.conftest import FakeRetrievalService, jpmorgan_source, real_normalization_service
from tests.pipeline.test_graph import FakeLLMClient, _relevant_assessment
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)

_ERROR_DIAGNOSTIC = PipelineDiagnostic(stage="retrieve", message="Some Source: connection_error", severity="error")
_WARNING_DIAGNOSTIC = PipelineDiagnostic(stage="agent_analysis", message="peer: top_k=5 applied", severity="warning")

# ---------------------------------------------------------------------------
# Unit-level tests directly against `_determine_status` (TESTS 1-5's core
# logic, isolated from the full graph).
# ---------------------------------------------------------------------------


def test_1_zero_errors_zero_events_is_success():
    status = _determine_status([], digest_persistence_status=DigestPersistenceStatus.NO_QUALIFYING_EVENTS)
    assert status == RunStatus.SUCCESS


def test_2_zero_errors_with_events_is_success():
    status = _determine_status([], digest_persistence_status=DigestPersistenceStatus.PERSISTED)
    assert status == RunStatus.SUCCESS


def test_3_source_errors_zero_events_digest_persisted_is_partial_success():
    """The exact scenario from production run 5e317bf0b132dd45."""
    status = _determine_status(
        [_ERROR_DIAGNOSTIC], digest_persistence_status=DigestPersistenceStatus.NO_QUALIFYING_EVENTS,
    )
    assert status == RunStatus.PARTIAL_SUCCESS


def test_4_source_errors_with_events_digest_persisted_is_partial_success():
    status = _determine_status([_ERROR_DIAGNOSTIC], digest_persistence_status=DigestPersistenceStatus.PERSISTED)
    assert status == RunStatus.PARTIAL_SUCCESS


def test_5_actual_digest_persistence_failure_is_failed():
    """A real persistence failure is FAILED regardless of whether there
    were also source-level errors."""
    status = _determine_status([], digest_persistence_status=DigestPersistenceStatus.PERSISTENCE_FAILED)
    assert status == RunStatus.FAILED

    status_with_errors_too = _determine_status(
        [_ERROR_DIAGNOSTIC], digest_persistence_status=DigestPersistenceStatus.PERSISTENCE_FAILED,
    )
    assert status_with_errors_too == RunStatus.FAILED


def test_a_pure_warning_never_downgrades_below_success():
    status = _determine_status([_WARNING_DIAGNOSTIC], digest_persistence_status=DigestPersistenceStatus.PERSISTED)
    assert status == RunStatus.SUCCESS


def test_not_attempted_persistence_status_with_errors_is_still_partial_success():
    """Covers the pre-existing "no persistence_service configured" path
    (common in unit tests) -- must not regress to FAILED just because no
    persistence service happens to be wired up."""
    status = _determine_status([_ERROR_DIAGNOSTIC], digest_persistence_status=DigestPersistenceStatus.NOT_ATTEMPTED)
    assert status == RunStatus.PARTIAL_SUCCESS


# ---------------------------------------------------------------------------
# TEST 6: an actual pipeline-stage failure (the graph itself raises) is a
# SEPARATE code path -- PipelineRunner.run()'s `except` branch -- and must
# remain FAILED unconditionally. This exercises the full PipelineRunner,
# not the bare function, to prove that code path is untouched by this fix.
# ---------------------------------------------------------------------------


def _jpmorgan_article() -> ExtractedArticle:
    return ExtractedArticle(
        url="https://jpmorganchase.com/a", title="JPMorgan launches AI copilot",
        text="JPMorgan Chase today launched a new generative AI copilot for financial advisors.",
        published_at=D - timedelta(hours=1),
    )


def _build_runner(config, tmp_path: Path, *, peer_responses, report_responses, retrieval_service=None, persistence_service=None):
    from research_assistant.agents.peer import PeerAgent
    from research_assistant.agents.report import ReportAgent

    if persistence_service is None:
        database = Database(tmp_path / "status_semantics.db")
        database.initialize()
        persistence_service = PersistenceService(database)

    services = PipelineServices(
        config=config,
        retrieval_service=retrieval_service or FakeRetrievalService(
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
    return PipelineRunner(config, services)


def test_6_an_actual_pipeline_stage_failure_is_failed(config, tmp_path):
    class ExplodingRetrievalService:
        def list_sources(self):
            raise RuntimeError("catastrophic failure")

    runner = _build_runner(
        config, tmp_path, peer_responses=[], report_responses=[], retrieval_service=ExplodingRetrievalService(),
    )

    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.status == RunStatus.FAILED
    assert result.digest_attempted is False


def test_6b_a_real_digest_persistence_failure_through_the_full_runner_is_failed(config, tmp_path):
    class _ExplodingPersistenceService:
        def persist_pipeline_result(self, **kwargs):
            raise RuntimeError("disk full")

    runner = _build_runner(
        config, tmp_path, peer_responses=[_relevant_assessment()], report_responses=[make_narrative()],
        persistence_service=_ExplodingPersistenceService(),
    )

    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.status == RunStatus.FAILED
    assert result.digest_persistence_status == DigestPersistenceStatus.PERSISTENCE_FAILED


# ---------------------------------------------------------------------------
# Regression test for the exact production incident: source errors, zero
# events, digest genuinely persisted (no_qualifying_events) -- must be
# PARTIAL_SUCCESS, never FAILED. Built from a real, full PipelineRunner run
# (not just the bare `_determine_status` function) against an isolated
# temp DB -- the production database is never touched.
# ---------------------------------------------------------------------------


def test_production_run_5e317bf0b132dd45_regression_isolated_sources_zero_events_is_partial_success(config, tmp_path):
    other_source = SourceDefinition(
        name="Broken Source", domain="broken.example.com", urls=["https://broken.example.com/news"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )
    irrelevant_article = ExtractedArticle(
        url="https://jpmorganchase.com/a", title="JPMorgan community grant program",
        text="JPMorgan Chase today announced a new community grant program supporting small businesses.",
        published_at=D - timedelta(hours=1),
    )
    retrieval_service = FakeRetrievalService(
        sources=[jpmorgan_source(), other_source],
        articles_by_source={"JPMorgan Chase": [irrelevant_article]},
        failing_sources={"Broken Source"},  # mirrors the 7 real connection_error sources
    )
    irrelevant_assessment = _relevant_assessment(
        relevance=RelevanceDecision(label=RelevanceLabel.IRRELEVANT, reason="no AI/GenAI angle"),
    )
    runner = _build_runner(
        config, tmp_path, peer_responses=[irrelevant_assessment], report_responses=[],
        retrieval_service=retrieval_service,
    )

    result = runner.run(digest_type=DigestType.EVENING, period_start=D - timedelta(hours=12), period_end=D)

    assert any(d.stage == "retrieve" and d.severity == "error" for d in result.diagnostics)  # source errors occurred
    assert result.counts["events_formed"] == 0  # zero qualifying events, exactly like the real run
    assert result.digest_attempted is True
    assert result.digest_persisted is True
    assert result.digest_persistence_status == DigestPersistenceStatus.NO_QUALIFYING_EVENTS
    assert result.status == RunStatus.PARTIAL_SUCCESS  # the fix: NOT FAILED
