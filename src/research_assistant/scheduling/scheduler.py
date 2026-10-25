"""APScheduler wrapper — timing/orchestration only, never business logic.

Registers two recurring jobs (morning digest, evening digest) in an
explicit IANA timezone (`config.settings.system.timezone`, which is always
`Asia/Kolkata` per config/settings.yaml — never the OS's local timezone)
and delegates each firing to `PipelineRunner` via `scheduling.jobs`.
Morning/Evening fire daily via `CronTrigger` at
`config.settings.research.morning_digest_hour`/`.evening_digest_hour`
(both already existed since Phase 0). CURRENT has no scheduled variant at
all -- it is only ever triggered manually (e.g. the UI's "Run Live
Current Scan" button) and never sends email automatically; see
`PipelineRunner._deliver_digest_if_eligible`.

    ResearchScheduler(config, runner)
              |
    .start() -> registers jobs, starts APScheduler
              |
    08:00 / 20:00 Asia/Kolkata (daily)
              |
    run_morning_digest / run_evening_digest   (execution_mode=SCHEDULED)
              |
              v
    PipelineRunner.run(...)   -- the existing LangGraph pipeline, unchanged

Phase 14: this class is now the ONLY place `execution_mode=SCHEDULED` is
ever passed to `PipelineRunner.run` (closing the forensic-diagnosis finding
that this class existed, fully tested, with zero real call sites). It also
now owns a lightweight heartbeat (`scheduler_state` table, via
`SchedulerStateRepository`) so the system's operational state is
observable independently of `pipeline_runs` -- a stale/dead scheduler and
an old-but-real run are not the same fact, and the UI must be able to tell
them apart (see `ui.dashboard_data.get_scheduler_status`). The actual
persistent process that keeps this instance alive is
`scheduling.daemon` -- NOT Streamlit, which only ever reads the persisted
state.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from apscheduler.job import Job
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from research_assistant.config.schema import AppConfig
from research_assistant.events.clock import Clock, SystemClock
from research_assistant.models.enums import ExecutionMode
from research_assistant.pipeline.runner import PipelineRunner
from research_assistant.scheduling.jobs import run_evening_digest, run_morning_digest

logger = logging.getLogger(__name__)

MORNING_JOB_ID = "morning_digest"
EVENING_JOB_ID = "evening_digest"
HEARTBEAT_JOB_ID = "scheduler_heartbeat"

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60

# Forensic finding (2026-10-08): the 2026-10-07 08:00 IST morning run never
# fired at all -- APScheduler logged "was missed by 1:50:41" and silently
# dropped it. APScheduler's own default `misfire_grace_time` is 1 second,
# so any delay larger than that (the process being suspended/asleep, or
# simply busy running the other cron job) causes the run to be skipped
# rather than executed late. A generous grace window ensures a legitimately
# due Morning/Evening run still executes even if the process was briefly
# unavailable, instead of silently vanishing for the rest of that cycle.
MISFIRE_GRACE_TIME_SECONDS = 4 * 60 * 60  # 4 hours


class ResearchScheduler:
    def __init__(
        self,
        config: AppConfig,
        runner: PipelineRunner,
        *,
        persistence_service=None,
        clock: Optional[Clock] = None,
        heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self._config = config
        self._runner = runner
        # Defaults to the runner's own persistence_service (the normal
        # production wiring) so a caller doesn't have to pass it twice --
        # explicitly overridable (e.g. tests using an isolated DB).
        self._persistence_service = (
            persistence_service if persistence_service is not None else getattr(runner.services, "persistence_service", None)
        )
        self._clock = clock or SystemClock()
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._scheduler = BackgroundScheduler(timezone=config.settings.system.timezone)
        self._started = False

    @property
    def timezone(self) -> str:
        return self._config.settings.system.timezone

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def persistence_service(self):
        """Exposed read-only so `scheduling.daemon` can run its one-time,
        bounded email-delivery backfill discovery at startup (see
        `delivery.service.discover_and_deliver_pending`) without reaching
        into a private attribute. Never used for anything CURRENT-related
        -- this remains exactly the same object `_write_heartbeat`/
        `_record_last_scheduled_run` already use."""
        return self._persistence_service

    def _run_morning(self) -> None:
        self._run_and_record(run_morning_digest, digest_type_label="morning")

    def _run_evening(self) -> None:
        self._run_and_record(run_evening_digest, digest_type_label="evening")

    def _run_and_record(self, job_fn, *, digest_type_label: str) -> None:
        """A failed scheduled run must never affect the scheduler's own
        lifecycle: the exception is caught and recorded here, not
        re-raised. APScheduler's own executor would catch an unhandled
        exception too (so re-raising would not, by itself, have crashed the
        process) -- but re-raising skipped recording the failure and skipped
        the post-run heartbeat write below, and relied entirely on
        APScheduler's internal behavior rather than this class explicitly
        isolating scheduled-run failures from scheduler health, as a
        resilient scheduler must."""
        logger.info("scheduler: firing scheduled %s run", digest_type_label)
        try:
            result = job_fn(self._runner, execution_mode=ExecutionMode.SCHEDULED)
        except Exception as exc:
            logger.exception("scheduler: scheduled %s run raised an unhandled exception", digest_type_label)
            self._record_scheduled_run_exception(digest_type_label, exc)
            self._write_heartbeat()
            return
        logger.info(
            "scheduler: scheduled %s run finished: run_id=%s status=%s digest_persistence_status=%s",
            digest_type_label, result.run_id, result.status.value, result.digest_persistence_status.value,
        )
        self._record_last_scheduled_run(result)
        self._write_heartbeat()

    def _record_last_scheduled_run(self, result) -> None:
        if self._persistence_service is None:
            return
        try:
            conn = self._persistence_service.database.connect()
            try:
                self._persistence_service.scheduler_state.upsert(
                    conn,
                    last_scheduled_run_id=result.run_id,
                    last_scheduled_execution_mode=result.execution_mode.value,
                    last_scheduled_digest_type=result.digest_type.value,
                    last_scheduled_started_at=result.started_at,
                    last_scheduled_finished_at=result.finished_at,
                    last_scheduled_status=result.status.value,
                )
            finally:
                conn.close()
        except Exception:
            # A persistence hiccup recording the outcome must never take the
            # scheduler lifecycle down with it -- the next scheduled run
            # must still be possible.
            logger.exception("scheduler: failed to persist last-scheduled-run state (scheduler remains alive)")

    def _record_scheduled_run_exception(self, digest_type_label: str, exc: Exception) -> None:
        if self._persistence_service is None:
            return
        try:
            conn = self._persistence_service.database.connect()
            try:
                self._persistence_service.scheduler_state.upsert(
                    conn,
                    last_scheduled_digest_type=digest_type_label,
                    last_scheduled_finished_at=self._clock.now(),
                    last_scheduled_status=f"exception: {exc!r}",
                )
            finally:
                conn.close()
        except Exception:
            logger.exception("scheduler: failed to persist scheduled-run exception state (scheduler remains alive)")

    def _write_heartbeat(self) -> None:
        """Called at `start()` and on every heartbeat-interval tick (and
        additionally right after a scheduled run completes, success or
        failure) -- cheap, read-nothing-from-the-network, pure bookkeeping.
        A no-op if no persistence service is configured (mirrors the rest of
        this project's "persistence is entirely optional" convention). Any
        persistence failure here is swallowed (logged only): a transient DB
        error must never stop future heartbeat ticks or scheduled runs."""
        if self._persistence_service is None:
            return
        try:
            morning_job = self._scheduler.get_job(MORNING_JOB_ID)
            evening_job = self._scheduler.get_job(EVENING_JOB_ID)
            conn = self._persistence_service.database.connect()
            try:
                self._persistence_service.scheduler_state.upsert(
                    conn,
                    last_heartbeat_at=self._clock.now(),
                    next_morning_run_at=getattr(morning_job, "next_run_time", None),
                    next_evening_run_at=getattr(evening_job, "next_run_time", None),
                )
            finally:
                conn.close()
        except Exception:
            logger.exception("scheduler: failed to write heartbeat (scheduler remains alive)")

    def register_jobs(self) -> None:
        research_settings = self._config.settings.research
        tz = self._config.settings.system.timezone

        self._scheduler.add_job(
            self._run_morning,
            trigger=CronTrigger(hour=research_settings.morning_digest_hour, minute=0, timezone=tz),
            id=MORNING_JOB_ID,
            replace_existing=True,
            misfire_grace_time=MISFIRE_GRACE_TIME_SECONDS,
        )
        self._scheduler.add_job(
            self._run_evening,
            trigger=CronTrigger(hour=research_settings.evening_digest_hour, minute=0, timezone=tz),
            id=EVENING_JOB_ID,
            replace_existing=True,
            misfire_grace_time=MISFIRE_GRACE_TIME_SECONDS,
        )
        self._scheduler.add_job(
            self._write_heartbeat,
            trigger=IntervalTrigger(seconds=self._heartbeat_interval_seconds),
            id=HEARTBEAT_JOB_ID,
            replace_existing=True,
        )
        logger.info(
            "scheduler: registered morning_digest at %02d:00, evening_digest at %02d:00, heartbeat every %ds",
            research_settings.morning_digest_hour, research_settings.evening_digest_hour,
            self._heartbeat_interval_seconds,
        )

    def get_jobs(self) -> List[Job]:
        return self._scheduler.get_jobs()

    def start(self) -> None:
        if self._started:
            logger.warning("scheduler: start() called while already running — ignoring (no duplicate instance)")
            return
        self.register_jobs()
        self._scheduler.start()
        self._started = True
        if self._persistence_service is not None:
            conn = self._persistence_service.database.connect()
            try:
                self._persistence_service.scheduler_state.upsert(conn, started_at=self._clock.now())
            finally:
                conn.close()
        self._write_heartbeat()  # immediately observable, not only after the first interval tick
        logger.info("scheduler: started")

    def shutdown(self, *, wait: bool = True) -> None:
        if not self._started:
            logger.info("scheduler: shutdown() called while not running — nothing to do")
            return
        self._scheduler.shutdown(wait=wait)
        self._started = False
        logger.info("scheduler: shut down cleanly")
