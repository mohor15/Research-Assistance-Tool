"""CURRENT candidate backlog persistence.

A candidate that passes every deterministic gate and has not been
conclusively rejected, but whose own run ran out of triage/deep-analysis
budget before reaching it, is upserted here (never duplicated -- one row
per `article_id`) so a later CURRENT run can resume it instead of it being
silently lost. Never stores a copy of the Article/Evidence content itself:
`article_id` is a stable reference into the already-persisted `articles`/
`evidence` tables (every retrieved article+evidence is persisted regardless
of whether a ResearchItem resulted -- see `persistence_service.
persist_pipeline_result`), so resuming a backlog candidate re-reads the
real, original content rather than any copy.

Claim safety needs no separate lease/claim protocol of its own: CURRENT
already serializes itself through the existing `current_run_lock` singleton
(only one CURRENT run is ever in flight), so two runs can never read-modify-
write the same backlog row concurrently.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from research_assistant.models.enums import BacklogStatus
from research_assistant.storage.serde import dt_from_str, dt_to_str

# Re-exported for convenience/backward-compat within this module's own
# callers -- the single source of truth is `models.enums.BacklogStatus`.
PENDING_TRIAGE_BUDGET = BacklogStatus.PENDING_TRIAGE_BUDGET.value
PENDING_DEEP_ANALYSIS_BUDGET = BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value
PENDING_RETRY = BacklogStatus.PENDING_RETRY.value
PENDING_STATUSES = (PENDING_TRIAGE_BUDGET, PENDING_DEEP_ANALYSIS_BUDGET, PENDING_RETRY)
# Two-tier CURRENT digest (Tier B): the subset of PENDING_STATUSES that
# represents an EXPLICIT triage pass still awaiting/blocked on specialist
# analysis -- excludes PENDING_TRIAGE_BUDGET (never triaged at all).
PENDING_STATUSES_FOR_TIER_B = (PENDING_DEEP_ANALYSIS_BUDGET, PENDING_RETRY)

COMPLETED = BacklogStatus.COMPLETED.value
REJECTED_TRIAGE_IRRELEVANT = BacklogStatus.REJECTED_TRIAGE_IRRELEVANT.value
REJECTED_DEEP_ANALYSIS_IRRELEVANT = BacklogStatus.REJECTED_DEEP_ANALYSIS_IRRELEVANT.value
REJECTED_DATE_GATE = BacklogStatus.REJECTED_DATE_GATE.value
REJECTED_RETRY_EXHAUSTED = BacklogStatus.REJECTED_RETRY_EXHAUSTED.value
QUARANTINED = BacklogStatus.QUARANTINED.value
# QUARANTINED is terminal in the sense that it is excluded from automatic
# `list_pending` reconsideration (retrying a structurally malformed row
# would just fail identically every run) -- but it is explicitly NOT
# included in `purge_expired_terminal`'s cleanup set below, since a
# quarantined row is evidence of a defect worth a human looking at, never
# something to silently age out alongside an ordinary completed/rejected
# outcome.
TERMINAL_STATUSES = (
    COMPLETED, REJECTED_TRIAGE_IRRELEVANT, REJECTED_DEEP_ANALYSIS_IRRELEVANT, REJECTED_DATE_GATE,
    REJECTED_RETRY_EXHAUSTED,
)
_EXCLUDED_FROM_AUTOMATIC_RETRY = (*TERMINAL_STATUSES, QUARANTINED)


@dataclass(frozen=True)
class BacklogRow:
    article_id: str
    agent: str
    status: str
    reason: Optional[str]
    attempt_count: int
    defer_count: int
    first_seen_run_id: str
    first_seen_at: datetime
    last_attempt_run_id: Optional[str]
    last_attempt_at: Optional[datetime]
    triage_relevant: Optional[bool]
    triage_confidence: Optional[float]
    research_item_id: Optional[str]
    updated_at: datetime


class BacklogRepository:
    def upsert(self, conn: sqlite3.Connection, row: BacklogRow) -> None:
        conn.execute(
            """
            INSERT INTO candidate_backlog (
                article_id, agent, status, reason, attempt_count, defer_count, first_seen_run_id, first_seen_at,
                last_attempt_run_id, last_attempt_at, triage_relevant, triage_confidence, research_item_id,
                updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(article_id) DO UPDATE SET
                agent=excluded.agent, status=excluded.status, reason=excluded.reason,
                attempt_count=excluded.attempt_count, defer_count=excluded.defer_count,
                last_attempt_run_id=excluded.last_attempt_run_id,
                last_attempt_at=excluded.last_attempt_at, triage_relevant=excluded.triage_relevant,
                triage_confidence=excluded.triage_confidence, research_item_id=excluded.research_item_id,
                updated_at=excluded.updated_at
            """,
            (
                row.article_id, row.agent, row.status, row.reason, row.attempt_count, row.defer_count,
                row.first_seen_run_id, dt_to_str(row.first_seen_at),
                row.last_attempt_run_id, dt_to_str(row.last_attempt_at),
                None if row.triage_relevant is None else int(row.triage_relevant),
                row.triage_confidence, row.research_item_id, dt_to_str(row.updated_at),
            ),
        )

    def get(self, conn: sqlite3.Connection, article_id: str) -> Optional[BacklogRow]:
        row = conn.execute("SELECT * FROM candidate_backlog WHERE article_id = ?", (article_id,)).fetchone()
        return _row_to_backlog(row) if row else None

    def list_pending(
        self, conn: sqlite3.Connection, *, limit: int, agents: Optional[List[str]] = None,
    ) -> List[BacklogRow]:
        """Oldest-`first_seen_at`-first (fairness: longest-waiting
        candidates are claimed before more recently deferred ones), bounded
        to `limit` -- a run never claims the entire backlog at once."""
        placeholders = ",".join("?" * len(PENDING_STATUSES))
        params: list = list(PENDING_STATUSES)
        agent_clause = ""
        if agents:
            agent_clause = f" AND agent IN ({','.join('?' * len(agents))})"
            params.extend(agents)
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM candidate_backlog WHERE status IN ({placeholders}){agent_clause} "
            f"ORDER BY first_seen_at ASC LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_backlog(row) for row in rows]

    def list_pending_for_run(self, conn: sqlite3.Connection, *, run_id: str) -> List[BacklogRow]:
        """Two-tier CURRENT digest (Tier B): every backlog row this
        SPECIFIC run touched (`last_attempt_run_id = run_id` -- set
        unconditionally on every upsert by `persistence_service.
        persist_pipeline_result`, whether the row is a real deep-analysis
        attempt or a pure budget deferral) that is still `pending_deep_
        analysis_budget` or `pending_retry` AND has an explicit passing
        triage decision (`triage_relevant = 1`).

        Deliberately excludes `pending_triage_budget` (never triaged at
        all -- must never be presented as "relevant", per the task's own
        "never-triaged budget-limited candidates are not falsely labelled
        relevant" rule) and `rejected_retry_exhausted` (no longer
        "awaiting" anything -- it is now a terminal disposition and will
        never be retried again, so presenting it as "awaiting specialist
        analysis" would be misleading)."""
        placeholders = ",".join("?" * len(PENDING_STATUSES_FOR_TIER_B))
        rows = conn.execute(
            f"SELECT * FROM candidate_backlog WHERE last_attempt_run_id = ? "
            f"AND status IN ({placeholders}) AND triage_relevant = 1 "
            f"ORDER BY first_seen_at ASC",
            [run_id, *PENDING_STATUSES_FOR_TIER_B],
        ).fetchall()
        return [_row_to_backlog(row) for row in rows]

    def count_pending(self, conn: sqlite3.Connection) -> int:
        placeholders = ",".join("?" * len(PENDING_STATUSES))
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM candidate_backlog WHERE status IN ({placeholders})",
            list(PENDING_STATUSES),
        ).fetchone()
        return row["n"]

    def oldest_pending_first_seen_at(self, conn: sqlite3.Connection) -> Optional[datetime]:
        placeholders = ",".join("?" * len(PENDING_STATUSES))
        row = conn.execute(
            f"SELECT MIN(first_seen_at) AS oldest FROM candidate_backlog WHERE status IN ({placeholders})",
            list(PENDING_STATUSES),
        ).fetchone()
        return dt_from_str(row["oldest"]) if row and row["oldest"] else None

    def requeue_exhausted(
        self, conn: sqlite3.Connection, article_id: str, *, now: datetime, reset_attempts: bool = True,
    ) -> str:
        """Explicit, narrowly-scoped recovery for exactly one candidate
        currently `rejected_retry_exhausted` -- never a bulk/automatic
        operation (there is no "requeue everything" call anywhere in this
        codebase). Validates the CURRENT status itself (never trusts a
        caller's assumption), so this can never accidentally move a
        `completed`, `rejected_triage_irrelevant`, `rejected_date_gate`, or
        `quarantined` row back to pending.

        `reset_attempts` (default `True`): a deliberate recovery action
        gives the candidate a genuinely fresh shot at `backlog_max_attempts`
        rather than being one failure away from immediately re-exhausting;
        pass `False` to preserve the exhausted attempt count instead (the
        new status is `pending_retry` either way, so a caller that wants
        the next failure to re-exhaust immediately can choose that). The
        bounded-aging `defer_count` is always preserved, never reset --
        it is a record of real history, not a penalty to erase.

        Returns "requeued" on success, "not_found" if no row exists for
        `article_id`, or "invalid_status" (a no-op, nothing written) if the
        row exists but is not currently `rejected_retry_exhausted` --
        idempotent: calling this twice in a row returns "requeued" once and
        "invalid_status" every time after, never an exception, never a
        second silent mutation."""
        row = self.get(conn, article_id)
        if row is None:
            return "not_found"
        if row.status != REJECTED_RETRY_EXHAUSTED:
            return "invalid_status"
        conn.execute(
            """
            UPDATE candidate_backlog
            SET status = ?, reason = ?, attempt_count = ?, last_attempt_run_id = ?, last_attempt_at = ?,
                updated_at = ?
            WHERE article_id = ? AND status = ?
            """,
            (
                PENDING_RETRY,
                f"manually requeued from rejected_retry_exhausted"
                + (" (attempt_count reset to 0)" if reset_attempts else " (attempt_count preserved)"),
                0 if reset_attempts else row.attempt_count,
                "manual_requeue", dt_to_str(now), dt_to_str(now),
                article_id, REJECTED_RETRY_EXHAUSTED,
            ),
        )
        return "requeued"

    def purge_expired_terminal(self, conn: sqlite3.Connection, *, before: datetime) -> int:
        """Deletes only TERMINAL rows (`completed`/`rejected_*`) whose
        `updated_at` is older than `before` -- a pending row is NEVER
        touched by this, regardless of age, so bounded storage never comes
        at the cost of silently dropping unprocessed work. Must be called
        explicitly (e.g. a maintenance script); nothing in the pipeline
        invokes this automatically."""
        placeholders = ",".join("?" * len(TERMINAL_STATUSES))
        cur = conn.execute(
            f"DELETE FROM candidate_backlog WHERE status IN ({placeholders}) AND updated_at < ?",
            [*TERMINAL_STATUSES, dt_to_str(before)],
        )
        return cur.rowcount


def _row_to_backlog(row: sqlite3.Row) -> BacklogRow:
    return BacklogRow(
        article_id=row["article_id"], agent=row["agent"], status=row["status"], reason=row["reason"],
        attempt_count=row["attempt_count"], defer_count=row["defer_count"], first_seen_run_id=row["first_seen_run_id"],
        first_seen_at=dt_from_str(row["first_seen_at"]), last_attempt_run_id=row["last_attempt_run_id"],
        last_attempt_at=dt_from_str(row["last_attempt_at"]),
        triage_relevant=None if row["triage_relevant"] is None else bool(row["triage_relevant"]),
        triage_confidence=row["triage_confidence"], research_item_id=row["research_item_id"],
        updated_at=dt_from_str(row["updated_at"]),
    )
