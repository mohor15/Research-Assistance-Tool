"""Shared fixtures for scheduling-package tests. `run_morning_digest`/
`run_evening_digest` are tested against a scripted `FakePipelineRunner` —
no live Ollama, retrieval, or scheduler process required."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest

from research_assistant.config import get_config
from research_assistant.events.clock import FixedClock
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import DigestPersistenceStatus, ExecutionMode
from research_assistant.pipeline.runner import PipelineRunResult, RunStatus

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


class FakePipelineRunner:
    """Duck-types the one method/attribute `scheduling.jobs` actually uses."""

    def __init__(self, *, now: datetime = D, config=None):
        self.services = _FakeServices(now, config=config)
        self.calls: List[dict] = []

    def run(
        self, *, digest_type: DigestType, period_start: datetime, period_end: datetime,
        execution_mode: ExecutionMode = ExecutionMode.MANUAL,
    ) -> PipelineRunResult:
        self.calls.append({
            "digest_type": digest_type, "period_start": period_start, "period_end": period_end,
            "execution_mode": execution_mode,
        })
        return PipelineRunResult(
            run_id=f"run-{digest_type.value}", digest_type=digest_type, started_at=period_end,
            finished_at=period_end, status=RunStatus.SUCCESS, counts={"events_formed": 1},
            execution_mode=execution_mode, period_start=period_start, period_end=period_end,
            digest_attempted=True, digest_persisted=True, digest_persistence_status=DigestPersistenceStatus.PERSISTED,
        )


class _FakeServices:
    def __init__(self, now: datetime, config=None):
        self.clock = FixedClock(now)
        self.config = config if config is not None else get_config()
        self.persistence_service = None


@pytest.fixture
def config():
    return get_config()


@pytest.fixture
def fake_runner():
    return FakePipelineRunner()
