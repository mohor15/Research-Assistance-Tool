"""`PipelineRunner`'s one automatic-email-delivery hook: fires exactly when
a run's digest was persisted WITH qualifying events (`DigestPersistenceStatus
.PERSISTED`), never for `NO_QUALIFYING_EVENTS`/`FAILED`, and never affects
the run's own reported status even when delivery itself fails (no SMTP
configured in these tests)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.report import ReportAgent
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.evidence.service import NormalizationService
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, DigestPersistenceStatus
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


def _build_runner_with_persistence(config, database, *, peer_responses, report_responses, retrieval_service=None):
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
        persistence_service=PersistenceService(database),
    )
    return PipelineRunner(config, services)


@pytest.fixture
def database(tmp_path: Path) -> Database:
    db = Database(tmp_path / "email_hook_test.db")
    db.initialize()
    return db


def test_automatic_delivery_is_triggered_for_a_persisted_digest_with_events(config, database, monkeypatch):
    calls = []

    def fake_deliver(service, cfg, *, digest_id, digest_type, now=None, mailer=None):
        calls.append({"digest_id": digest_id, "digest_type": digest_type})
        return None

    monkeypatch.setattr("research_assistant.delivery.service.deliver_digest_if_eligible", fake_deliver)

    runner = _build_runner_with_persistence(
        config, database, peer_responses=[_relevant_assessment()], report_responses=[make_narrative()],
    )
    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.digest_persistence_status == DigestPersistenceStatus.PERSISTED
    assert len(calls) == 1
    assert calls[0]["digest_id"] == result.digest_id
    assert calls[0]["digest_type"] == DigestType.MORNING


def test_automatic_delivery_is_not_triggered_when_no_qualifying_events(config, database, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "research_assistant.delivery.service.deliver_digest_if_eligible", lambda *a, **k: calls.append(1),
    )

    class _NoSourcesRetrievalService:
        def list_sources(self):
            return []

    runner = _build_runner_with_persistence(
        config, database, peer_responses=[], report_responses=[], retrieval_service=_NoSourcesRetrievalService(),
    )
    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.digest_persistence_status == DigestPersistenceStatus.NO_QUALIFYING_EVENTS
    assert calls == []


def test_automatic_delivery_is_not_triggered_when_the_run_fails(config, database, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "research_assistant.delivery.service.deliver_digest_if_eligible", lambda *a, **k: calls.append(1),
    )

    class _ExplodingRetrievalService:
        def list_sources(self):
            raise RuntimeError("boom")

    runner = _build_runner_with_persistence(
        config, database, peer_responses=[], report_responses=[], retrieval_service=_ExplodingRetrievalService(),
    )
    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.status == RunStatus.FAILED
    assert calls == []


def test_a_real_delivery_attempt_failure_does_not_affect_the_run_result(config, database):
    """Integration-level (not monkeypatched): with no SMTP configured in
    settings.yaml, the real delivery path records a failed delivery
    attempt, but the pipeline run itself is still reported by its own
    (digest-persistence-based) status -- email delivery is isolated from
    pipeline/run success semantics (Section 3/8)."""
    runner = _build_runner_with_persistence(
        config, database, peer_responses=[_relevant_assessment()], report_responses=[make_narrative()],
    )
    result = runner.run(digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert result.status in (RunStatus.SUCCESS, RunStatus.PARTIAL_SUCCESS)
    assert result.digest_persistence_status == DigestPersistenceStatus.PERSISTED

    conn = database.connect()
    try:
        rows = conn.execute("SELECT status, failure_reason FROM email_deliveries WHERE digest_id = ?", (result.digest_id,)).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert "not configured" in rows[0]["failure_reason"]
