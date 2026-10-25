"""The single entry point later phases (and the LangGraph persistence node)
should use to persist one pipeline run's output.

Coordinates the six repositories in FK-safe order (articles before
evidence/research_items, research_items/articles before events, events
before reports/digests) inside one transaction per call. Every upsert is
keyed by an id that already existed before persistence (content-derived
article/evidence/event ids, or the LLM-produced-but-deterministically-
assembled research_item/report/digest ids) — nothing here regenerates an
id, and running the same call twice updates rather than duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

from research_assistant.events.models import Event
from research_assistant.models.article import Article
from research_assistant.models.digest import Digest
from research_assistant.models.evidence import Evidence
from research_assistant.models.report import Report, ReportFailure
from research_assistant.models.research_item import ResearchItem
from research_assistant.storage.database import Database
from research_assistant.storage.repositories.article import ArticleRepository
from research_assistant.storage.repositories.backlog import BacklogRepository, BacklogRow
from research_assistant.storage.repositories.current_run_lock import CurrentRunLockRepository
from research_assistant.storage.repositories.digest import DigestRepository
from research_assistant.storage.repositories.email_delivery import EmailDeliveryRepository
from research_assistant.storage.repositories.email_settings import EmailSettingsRepository
from research_assistant.storage.repositories.event import EventRepository
from research_assistant.storage.repositories.evidence import EvidenceRepository
from research_assistant.storage.repositories.newsletter import NewsletterSelectionRepository
from research_assistant.storage.repositories.report import ReportRepository
from research_assistant.storage.repositories.research_item import ResearchItemRepository
from research_assistant.storage.repositories.run import RunRepository
from research_assistant.storage.repositories.scheduler_state import SchedulerStateRepository
from research_assistant.storage.repositories.tier_b_report import TierBReportRepository


@dataclass(frozen=True)
class BacklogCandidateRecord:
    """A pending backlog row reconstructed alongside its real, already-
    persisted Article/Evidence -- never a copy of their content."""

    backlog: BacklogRow
    article: Article
    evidence: Evidence


class PersistenceService:
    def __init__(self, database: Database) -> None:
        self.database = database  # public: the dashboard query layer opens its own read-only connections through this
        self.articles = ArticleRepository()
        self.evidence = EvidenceRepository()
        self.research_items = ResearchItemRepository()
        self.events = EventRepository()
        self.reports = ReportRepository()
        self.digests = DigestRepository()
        self.runs = RunRepository()
        self.scheduler_state = SchedulerStateRepository()
        self.newsletter = NewsletterSelectionRepository()
        self.current_run_lock = CurrentRunLockRepository()
        self.email_deliveries = EmailDeliveryRepository()
        self.email_settings = EmailSettingsRepository()
        self.backlog = BacklogRepository()
        self.tier_b_reports = TierBReportRepository()

    def persist_run(self, run) -> None:  # run: PipelineRunResult
        """Persists a run record only (Phase 8) — kept separate from
        `persist_pipeline_result` since `PipelineRunner` builds the run
        result *after* the graph (including the persistence node) has
        already finished, so it is saved in its own short transaction."""
        conn = self.database.connect()
        try:
            self.runs.upsert(conn, run)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def persist_pipeline_result(
        self,
        *,
        articles: List[Article] = (),
        evidence: List[Evidence] = (),
        research_items: List[ResearchItem] = (),
        events: List[Event] = (),
        reports: List[Report] = (),
        report_failures: List[ReportFailure] = (),
        digest: Optional[Digest] = None,
        backlog_updates: List[Any] = (),
        run_id: Optional[str] = None,
    ) -> None:
        """`backlog_updates` are `pipeline.state.BacklogUpdate` instances,
        accepted here by duck typing only (never imported -- mirrors
        `PipelineServices.persistence_service: Any`'s existing, deliberate
        decoupling in the other direction) so `storage` never depends on
        `pipeline`. `run_id` stamps a BRAND-NEW backlog row's
        `first_seen_run_id`; a RESUMED row (one whose `BacklogUpdate`
        already carries its own `first_seen_run_id`/`first_seen_at`) keeps
        those original values instead, so its fairness-ordering position
        is never reset merely by being deferred again."""
        conn = self.database.connect()
        try:
            for article in articles:
                self.articles.upsert(conn, article)
            for item in evidence:
                self.evidence.upsert(conn, item)
            for item in research_items:
                self.research_items.upsert(conn, item)
            for event in events:
                self.events.upsert(conn, event)
            for report in reports:
                self.reports.upsert_report(conn, report)
            for failure in report_failures:
                self.reports.upsert_failure(conn, failure)
            if digest is not None:
                self.digests.upsert(conn, digest)
            # Backlog rows are upserted in this SAME transaction as
            # everything else -- a candidate's disposition is never
            # recorded separately from (or ahead of) the rest of this run's
            # output, so a crash/exception here rolls back atomically with
            # the ResearchItems/Events it describes, never leaving the
            # backlog referring to a run that didn't actually commit.
            now = datetime.now(timezone.utc)
            for update in backlog_updates:
                existing = self.backlog.get(conn, update.article_id) if update.first_seen_run_id is None else None
                first_seen_run_id = update.first_seen_run_id or (existing.first_seen_run_id if existing else run_id) or "unknown"
                first_seen_at = update.first_seen_at or (existing.first_seen_at if existing else now)
                self.backlog.upsert(conn, BacklogRow(
                    article_id=update.article_id, agent=update.agent, status=update.status, reason=update.reason,
                    attempt_count=update.attempt_count, defer_count=update.defer_count,
                    first_seen_run_id=first_seen_run_id,
                    first_seen_at=first_seen_at, last_attempt_run_id=run_id, last_attempt_at=now,
                    triage_relevant=update.triage_relevant, triage_confidence=update.triage_confidence,
                    research_item_id=update.research_item_id, updated_at=now,
                ))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_pending_backlog_candidates(
        self, *, limit: int, agents: Optional[List[str]] = None,
    ) -> List[BacklogCandidateRecord]:
        """Read-only: reconstructs each pending backlog candidate's real,
        already-persisted Article and (first) Evidence record -- never a
        stored copy. Safe to call mid-run (a read, not a write); the
        actual backlog state transition for these candidates is still only
        ever written once, atomically, via `persist_pipeline_result`."""
        conn = self.database.connect()
        try:
            rows = self.backlog.list_pending(conn, limit=limit, agents=agents)
            records: List[BacklogCandidateRecord] = []
            for row in rows:
                article = self.articles.get(conn, row.article_id)
                if article is None:
                    continue  # should not happen (FK), but never crash a run over it
                evidence_list = self.evidence.get_by_article_id(conn, row.article_id)
                if not evidence_list:
                    continue  # no retrieved content survives -- nothing to resume
                records.append(BacklogCandidateRecord(backlog=row, article=article, evidence=evidence_list[0]))
            return records
        finally:
            conn.close()

    def requeue_exhausted_backlog_candidate(
        self, article_id: str, *, reset_attempts: bool = True,
    ) -> str:
        """Explicit, single-candidate recovery for a `rejected_retry_
        exhausted` row -- see `BacklogRepository.requeue_exhausted` for the
        full contract (status validation, idempotency, `defer_count`
        preservation). Never automatic, never bulk; no scheduler or UI
        control calls this -- it is a deliberately manual, narrowly-scoped
        operation. Returns "requeued", "not_found", or "invalid_status"."""
        conn = self.database.connect()
        try:
            result = self.backlog.requeue_exhausted(
                conn, article_id, now=datetime.now(timezone.utc), reset_attempts=reset_attempts,
            )
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def backlog_summary(self) -> dict:
        """Read-only observability: pending backlog size and the oldest
        pending candidate's age, for admin/diagnostic visibility (Section
        8's "clear visibility into remaining backlog size and oldest
        pending age") -- never used to gate or alter processing itself."""
        conn = self.database.connect()
        try:
            return {
                "pending_count": self.backlog.count_pending(conn),
                "oldest_pending_first_seen_at": self.backlog.oldest_pending_first_seen_at(conn),
            }
        finally:
            conn.close()
