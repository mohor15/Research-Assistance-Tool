"""PipelineRunner — the single entry point scheduled jobs and manual runs
should use to actually execute a digest run.

    PipelineRunner.build(config)   -- constructs real services from config
              |
    runner.run(digest_type, period_start, period_end)
              |
    build_graph(services).invoke(PipelineState(...))   -- unchanged Phase 6 graph
              |
    PipelineRunResult (run_id, status, diagnostics, counts)

This is orchestration only: it never re-implements a pipeline stage, and
`.build()` is the *only* place that wires up the real Ollama-backed agents/
report agent and the real SQLite-backed persistence service — everywhere
else in the codebase (including all tests) injects `PipelineServices`
directly, exactly as in Phase 6/7.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from research_assistant.agents.ai_product import AIProductAgent
from research_assistant.agents.analyst import AnalystAgent
from research_assistant.agents.financial_news import FinancialNewsAgent
from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.report import ReportAgent
from research_assistant.agents.startup import StartupAgent
from research_assistant.config import get_config
from research_assistant.config.schema import AppConfig
from research_assistant.events.clock import Clock, SystemClock
from research_assistant.events.service import EventService
from research_assistant.evidence.service import NormalizationService
from research_assistant.llm.client import create_ollama_client
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, DigestPersistenceStatus, ExecutionMode
from research_assistant.pipeline.graph import build_graph
from research_assistant.pipeline.nodes import PipelineServices
from research_assistant.pipeline.state import PipelineDiagnostic, PipelineState
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.service import RetrievalService
from research_assistant.storage.database import Database
from research_assistant.storage.persistence_service import PersistenceService
from research_assistant.utils.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

_ID_LENGTH = 16


class RunStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class PipelineRunResult(BaseModel):
    run_id: str
    digest_type: DigestType
    started_at: datetime
    finished_at: datetime
    status: RunStatus
    diagnostics: List[PipelineDiagnostic] = Field(default_factory=list)
    counts: Dict[str, int] = Field(default_factory=dict)
    # Phase 14: explicit execution/digest-persistence metadata, closing the
    # forensic-diagnosis finding that neither "was this scheduled or
    # manual" nor "why is there no digest" could previously be answered
    # without guessing. Never inferred by a caller after the fact -- set
    # once, here, at the moment each fact becomes known.
    execution_mode: ExecutionMode = ExecutionMode.MANUAL
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    digest_attempted: bool = False
    digest_persisted: bool = False
    digest_persistence_status: DigestPersistenceStatus = DigestPersistenceStatus.NOT_ATTEMPTED
    digest_id: Optional[str] = None


class CurrentRunAlreadyInProgressError(RuntimeError):
    """Raised instead of starting a second CURRENT pipeline run while one
    is already in flight (Section 3's required application-level guard).
    Never raised for Morning/Evening -- CURRENT is the only digest type
    with no scheduled variant and therefore the only one an operator can
    accidentally double-trigger from the admin UI."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"CURRENT scan already running (run_id={run_id})")


def generate_run_id(digest_type: DigestType, period_start: datetime, period_end: datetime) -> str:
    """Deterministic — derived from the run's identity (digest type + period
    bounds), never a random UUID or a wall-clock timestamp. A scheduler
    retry, a manual re-run, or a process restart that re-executes "the same
    conceptual run" produces the same run_id, so persistence (Phase 7's
    upsert-by-id) makes it naturally idempotent rather than duplicating a
    run record."""
    basis = f"{digest_type.value}:{period_start.isoformat()}:{period_end.isoformat()}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:_ID_LENGTH]


class PipelineRunner:
    def __init__(self, config: AppConfig, services: PipelineServices) -> None:
        self._config = config
        self._services = services
        self._graph = build_graph(services)

    @property
    def services(self) -> PipelineServices:
        return self._services

    @classmethod
    def build(
        cls,
        config: Optional[AppConfig] = None,
        *,
        database: Optional[Database] = None,
        clock: Optional[Clock] = None,
    ) -> "PipelineRunner":
        """Constructs real (Ollama-backed, SQLite-backed) services from
        configuration — this is the path scheduled jobs and the manual CLI
        use. Every unit test instead constructs `PipelineServices` directly
        with fakes and calls `PipelineRunner(config, services)`."""
        config = config or get_config()
        clock = clock or SystemClock()

        llm_client = create_ollama_client(config)
        current_llm_settings = config.settings.current_llm
        triage_llm_client = create_ollama_client(config, timeout_seconds=current_llm_settings.triage_timeout_seconds)
        deep_analysis_llm_client = create_ollama_client(
            config, timeout_seconds=current_llm_settings.deep_analysis_timeout_seconds
        )
        agents = {
            AgentName.PEER: PeerAgent(llm_client, config),
            AgentName.AI_PRODUCT: AIProductAgent(llm_client, config),
            AgentName.STARTUP: StartupAgent(llm_client, config),
            AgentName.ANALYST: AnalystAgent(llm_client, config),
            AgentName.FINANCIAL_NEWS: FinancialNewsAgent(llm_client, config),
        }

        if database is None:
            database = Database(PROJECT_ROOT / config.settings.storage.database_path)
        database.initialize()

        services = PipelineServices(
            config=config,
            retrieval_service=RetrievalService(config),
            normalization_service=NormalizationService(),
            agents=agents,
            event_service=EventService(config, clock=clock),
            report_agent=ReportAgent(llm_client),
            digest_service=DigestService(config, clock=clock),
            clock=clock,
            persistence_service=PersistenceService(database),
            triage_llm_client=triage_llm_client,
            deep_analysis_llm_client=deep_analysis_llm_client,
        )
        return cls(config, services)

    def run(
        self, *, digest_type: DigestType, period_start: datetime, period_end: datetime,
        execution_mode: ExecutionMode = ExecutionMode.MANUAL,
    ) -> PipelineRunResult:
        run_id = generate_run_id(digest_type, period_start, period_end)

        # Section 3's required application-level guard: CURRENT has no
        # scheduled variant, so it is the one digest type an operator can
        # accidentally trigger twice (e.g. a double click on "Run Live
        # Current Scan now"). Checked/acquired BEFORE anything else --
        # no retrieval, no Ollama call, no state is touched for the
        # losing caller. Deliberately never applied to Morning/Evening
        # (both are scheduler-owned and already serialized by the
        # scheduler itself).
        persistence_service = getattr(self._services, "persistence_service", None)
        current_lock_held = False
        if digest_type == DigestType.CURRENT and persistence_service is not None:
            conn = persistence_service.database.connect()
            try:
                acquired = persistence_service.current_run_lock.try_acquire(
                    conn, run_id=run_id, started_at=self._services.clock.now(),
                )
            finally:
                conn.close()
            if not acquired:
                conn = persistence_service.database.connect()
                try:
                    holder = persistence_service.current_run_lock.get(conn)
                finally:
                    conn.close()
                raise CurrentRunAlreadyInProgressError(holder.run_id if holder else "unknown")
            current_lock_held = True

        try:
            return self._run_locked(
                run_id=run_id, digest_type=digest_type, period_start=period_start, period_end=period_end,
                execution_mode=execution_mode,
            )
        finally:
            if current_lock_held:
                conn = persistence_service.database.connect()
                try:
                    persistence_service.current_run_lock.release(conn, run_id=run_id)
                finally:
                    conn.close()

    def _run_locked(
        self, *, run_id: str, digest_type: DigestType, period_start: datetime, period_end: datetime,
        execution_mode: ExecutionMode,
    ) -> PipelineRunResult:
        started_at = self._services.clock.now()
        logger.info(
            "pipeline_run: run_id=%s digest_type=%s execution_mode=%s period=%s..%s starting",
            run_id, digest_type.value, execution_mode.value, period_start.isoformat(), period_end.isoformat(),
        )

        initial_state = PipelineState(
            digest_type=digest_type, period_start=period_start, period_end=period_end, run_id=run_id,
        )

        try:
            result = self._graph.invoke(initial_state)
        except Exception as exc:
            finished_at = self._services.clock.now()
            logger.error("pipeline_run: run_id=%s failed with an unhandled exception: %s", run_id, exc)
            run_result = PipelineRunResult(
                run_id=run_id, digest_type=digest_type, started_at=started_at, finished_at=finished_at,
                status=RunStatus.FAILED,
                diagnostics=[PipelineDiagnostic(stage="pipeline_run", message=str(exc), severity="error")],
                counts={},
                execution_mode=execution_mode, period_start=period_start, period_end=period_end,
                # The graph never reached digest_generation -- a digest was
                # never attempted, not merely "not persisted".
                digest_attempted=False, digest_persisted=False,
                digest_persistence_status=DigestPersistenceStatus.NOT_ATTEMPTED,
            )
            self._persist_run(run_result)
            return run_result

        finished_at = self._services.clock.now()
        diagnostics = list(result.get("diagnostics", []))
        counts = dict(result.get("metrics", {}))

        digest_obj = result.get("digest")
        digest_attempted = digest_obj is not None
        persistence_failed = any(d.stage == "persistence" for d in diagnostics)
        if not digest_attempted or self._services.persistence_service is None:
            digest_persisted = False
            digest_persistence_status = DigestPersistenceStatus.NOT_ATTEMPTED
        elif persistence_failed:
            digest_persisted = False
            digest_persistence_status = DigestPersistenceStatus.PERSISTENCE_FAILED
        else:
            digest_persisted = True
            digest_persistence_status = (
                DigestPersistenceStatus.PERSISTED if digest_obj.events else DigestPersistenceStatus.NO_QUALIFYING_EVENTS
            )

        # digest_persistence_status is computed BEFORE status so that a real
        # persistence failure can override an otherwise-isolated-errors
        # PARTIAL_SUCCESS verdict (see _determine_status).
        status = _determine_status(diagnostics, digest_persistence_status=digest_persistence_status)

        logger.info(
            "pipeline_run: run_id=%s status=%s execution_mode=%s digest_persistence_status=%s "
            "finished in %.2fs diagnostics=%d",
            run_id, status.value, execution_mode.value, digest_persistence_status.value,
            (finished_at - started_at).total_seconds(), len(diagnostics),
        )

        run_result = PipelineRunResult(
            run_id=run_id, digest_type=digest_type, started_at=started_at, finished_at=finished_at,
            status=status, diagnostics=diagnostics, counts=counts,
            execution_mode=execution_mode, period_start=period_start, period_end=period_end,
            digest_attempted=digest_attempted, digest_persisted=digest_persisted,
            digest_persistence_status=digest_persistence_status,
            digest_id=digest_obj.id if digest_obj else None,
        )
        self._persist_run(run_result)
        self._deliver_digest_if_eligible(run_result)
        return run_result

    def _deliver_digest_if_eligible(self, run_result: "PipelineRunResult") -> None:
        """The single place every digest type (CURRENT, Morning, Evening --
        whichever process actually called `.run()`) becomes eligible for
        automatic email delivery, immediately after persistence succeeds.
        Only `DigestPersistenceStatus.PERSISTED` (a digest WAS persisted
        AND has >=1 qualifying event) triggers delivery -- never
        `NO_QUALIFYING_EVENTS` (nothing to email), `PERSISTENCE_FAILED`, or
        `NOT_ATTEMPTED`. Any failure here (including email being disabled
        or unconfigured) is caught and logged, never re-raised -- email
        delivery must never affect pipeline/run success semantics.

        CURRENT has no automatic email delivery at all -- it is manual-only
        (the UI's "Run Live Current Scan" button / the manual CLI), and a
        manually triggered scan finishing is surfaced to the user in the
        UI itself (a completion notice), never via email. Only Morning/
        Evening ever reach `deliver_digest_if_eligible` below."""
        if run_result.digest_persistence_status != DigestPersistenceStatus.PERSISTED or run_result.digest_id is None:
            return
        if run_result.digest_type == DigestType.CURRENT:
            return
        persistence_service = getattr(self._services, "persistence_service", None)
        if persistence_service is None:
            return
        try:
            from research_assistant.delivery.service import deliver_digest_if_eligible

            deliver_digest_if_eligible(
                persistence_service, self._config, digest_id=run_result.digest_id,
                digest_type=run_result.digest_type, now=self._services.clock.now(),
            )
        except Exception:
            logger.exception(
                "pipeline_run: run_id=%s automatic email delivery raised unexpectedly (run result unaffected)",
                run_result.run_id,
            )

    def _persist_run(self, run_result: "PipelineRunResult") -> None:
        """Best-effort: a failure persisting the *run record itself* must not
        mask or replace the real run result already computed above."""
        persistence_service = getattr(self._services, "persistence_service", None)
        if persistence_service is None:
            return
        try:
            persistence_service.persist_run(run_result)
        except Exception as exc:
            logger.error("pipeline_run: run_id=%s failed to persist run record: %s", run_result.run_id, exc)


def _determine_status(
    diagnostics: List[PipelineDiagnostic], *, digest_persistence_status: DigestPersistenceStatus,
) -> RunStatus:
    """Status-semantics fix (confirmed via the forensic investigation of
    production run `5e317bf0b132dd45`): the pipeline graph having already
    completed (this function is only ever called on the success path of
    `run()` -- an unhandled exception is handled entirely in the `except`
    branch above and never reaches here, so `digest_persistence_status`
    being anything other than `PERSISTENCE_FAILED` here means every stage
    through `digest_generation`/`persistence` genuinely ran) means the only
    remaining real-failure signal is an ACTUAL digest persistence failure --
    never merely "some source(s) had connection errors" or "zero events
    qualified this run". Those are normal, expected, isolated conditions a
    fully-completed run can have:

    - FAILED: the digest persistence transaction itself raised
      (`digest_persistence_status == PERSISTENCE_FAILED`) -- an actual
      pipeline-stage failure is handled separately in `run()`'s `except`
      branch and never reaches this function at all.
    - SUCCESS: no error-severity diagnostics at all.
    - PARTIAL_SUCCESS: one or more error-severity diagnostics (e.g.
      isolated source retrieval failures) occurred, but the run still
      completed through persistence -- regardless of whether any events
      were formed. A run with 0 events because nothing relevant happened
      this run is not a failure merely because some other source also
      happened to be unreachable.

    A pure `warning` (e.g. a top_k truncation notice, or the Phase 13E
    zero-candidate-source notice) never downgrades a run below SUCCESS --
    only `severity="error"` diagnostics are considered here, unchanged from
    the original semantics.
    """
    if digest_persistence_status == DigestPersistenceStatus.PERSISTENCE_FAILED:
        return RunStatus.FAILED
    has_errors = any(d.severity == "error" for d in diagnostics)
    if not has_errors:
        return RunStatus.SUCCESS
    return RunStatus.PARTIAL_SUCCESS
