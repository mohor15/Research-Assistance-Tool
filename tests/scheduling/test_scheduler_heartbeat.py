"""ResearchScheduler's heartbeat/state-recording behavior against a REAL
(temp-file) database -- closes the forensic-diagnosis finding that the
system could not answer "is the scheduler alive?" independently of
`pipeline_runs`. No live network, no live Ollama: `FakePipelineRunner` from
`tests.scheduling.conftest` stands in for the real `PipelineRunner`."""

from __future__ import annotations

from pathlib import Path

from research_assistant.scheduling.scheduler import EVENING_JOB_ID, MORNING_JOB_ID, ResearchScheduler
from research_assistant.storage.database import Database
from research_assistant.storage.persistence_service import PersistenceService

from tests.scheduling.conftest import FakePipelineRunner


def _service(tmp_path: Path) -> PersistenceService:
    database = Database(tmp_path / "heartbeat.db")
    database.initialize()
    return PersistenceService(database)


def test_start_writes_an_immediate_heartbeat_without_waiting_for_the_interval(tmp_path):
    service = _service(tmp_path)
    scheduler = ResearchScheduler(config=FakePipelineRunner().services.config, runner=FakePipelineRunner(), persistence_service=service)

    scheduler.start()
    try:
        conn = service.database.connect()
        record = service.scheduler_state.get(conn)
        conn.close()

        assert record is not None
        assert record.started_at is not None
        assert record.last_heartbeat_at is not None
    finally:
        scheduler.shutdown()


def test_start_records_next_morning_and_evening_run_times(tmp_path):
    service = _service(tmp_path)
    runner = FakePipelineRunner()
    scheduler = ResearchScheduler(config=runner.services.config, runner=runner, persistence_service=service)

    scheduler.start()
    try:
        conn = service.database.connect()
        record = service.scheduler_state.get(conn)
        conn.close()

        assert record.next_morning_run_at is not None
        assert record.next_evening_run_at is not None
    finally:
        scheduler.shutdown()


def test_a_fired_scheduled_run_updates_last_scheduled_fields(tmp_path):
    service = _service(tmp_path)
    runner = FakePipelineRunner()
    scheduler = ResearchScheduler(config=runner.services.config, runner=runner, persistence_service=service)
    scheduler.register_jobs()

    jobs = {job.id: job for job in scheduler.get_jobs()}
    jobs[MORNING_JOB_ID].func()  # directly invoke, as a real APScheduler fire would

    conn = service.database.connect()
    record = service.scheduler_state.get(conn)
    conn.close()

    assert record.last_scheduled_run_id == "run-morning"
    assert record.last_scheduled_execution_mode == "scheduled"
    assert record.last_scheduled_digest_type == "morning"
    assert record.last_scheduled_status == "success"


def test_heartbeat_with_no_persistence_service_is_a_safe_no_op(tmp_path):
    runner = FakePipelineRunner()
    scheduler = ResearchScheduler(config=runner.services.config, runner=runner, persistence_service=None)

    scheduler.start()  # must not raise even though nothing is configured to persist to
    scheduler.shutdown()
