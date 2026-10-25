"""Phase 14: the persistent scheduler process's own observable state --
a single row (`id='singleton'`), never inferred from `pipeline_runs` alone.

`upsert` is a PARTIAL update: any field left `None` keeps its existing
stored value (via `COALESCE(excluded.x, scheduler_state.x)`), since
different callers update different subsets at different times -- a
heartbeat tick only touches `last_heartbeat_at`/`next_*_run_at`; a
completed scheduled run only touches the `last_scheduled_*` fields.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from research_assistant.storage.serde import dt_from_str, dt_to_str

_SINGLETON_ID = "singleton"


@dataclass
class SchedulerStateRecord:
    started_at: Optional[datetime]
    last_heartbeat_at: Optional[datetime]
    next_morning_run_at: Optional[datetime]
    next_evening_run_at: Optional[datetime]
    last_scheduled_run_id: Optional[str]
    last_scheduled_execution_mode: Optional[str]
    last_scheduled_digest_type: Optional[str]
    last_scheduled_started_at: Optional[datetime]
    last_scheduled_finished_at: Optional[datetime]
    last_scheduled_status: Optional[str]


class SchedulerStateRepository:
    def upsert(
        self,
        conn: sqlite3.Connection,
        *,
        started_at: Optional[datetime] = None,
        last_heartbeat_at: Optional[datetime] = None,
        next_morning_run_at: Optional[datetime] = None,
        next_evening_run_at: Optional[datetime] = None,
        last_scheduled_run_id: Optional[str] = None,
        last_scheduled_execution_mode: Optional[str] = None,
        last_scheduled_digest_type: Optional[str] = None,
        last_scheduled_started_at: Optional[datetime] = None,
        last_scheduled_finished_at: Optional[datetime] = None,
        last_scheduled_status: Optional[str] = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO scheduler_state (
                id, started_at, last_heartbeat_at, next_morning_run_at, next_evening_run_at,
                last_scheduled_run_id, last_scheduled_execution_mode,
                last_scheduled_digest_type, last_scheduled_started_at, last_scheduled_finished_at,
                last_scheduled_status
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                started_at = COALESCE(excluded.started_at, scheduler_state.started_at),
                last_heartbeat_at = COALESCE(excluded.last_heartbeat_at, scheduler_state.last_heartbeat_at),
                next_morning_run_at = COALESCE(excluded.next_morning_run_at, scheduler_state.next_morning_run_at),
                next_evening_run_at = COALESCE(excluded.next_evening_run_at, scheduler_state.next_evening_run_at),
                last_scheduled_run_id = COALESCE(excluded.last_scheduled_run_id, scheduler_state.last_scheduled_run_id),
                last_scheduled_execution_mode = COALESCE(excluded.last_scheduled_execution_mode, scheduler_state.last_scheduled_execution_mode),
                last_scheduled_digest_type = COALESCE(excluded.last_scheduled_digest_type, scheduler_state.last_scheduled_digest_type),
                last_scheduled_started_at = COALESCE(excluded.last_scheduled_started_at, scheduler_state.last_scheduled_started_at),
                last_scheduled_finished_at = COALESCE(excluded.last_scheduled_finished_at, scheduler_state.last_scheduled_finished_at),
                last_scheduled_status = COALESCE(excluded.last_scheduled_status, scheduler_state.last_scheduled_status)
            """,
            (
                _SINGLETON_ID,
                dt_to_str(started_at) if started_at else None,
                dt_to_str(last_heartbeat_at) if last_heartbeat_at else None,
                dt_to_str(next_morning_run_at) if next_morning_run_at else None,
                dt_to_str(next_evening_run_at) if next_evening_run_at else None,
                last_scheduled_run_id, last_scheduled_execution_mode, last_scheduled_digest_type,
                dt_to_str(last_scheduled_started_at) if last_scheduled_started_at else None,
                dt_to_str(last_scheduled_finished_at) if last_scheduled_finished_at else None,
                last_scheduled_status,
            ),
        )
        conn.commit()

    def get(self, conn: sqlite3.Connection) -> Optional[SchedulerStateRecord]:
        row = conn.execute("SELECT * FROM scheduler_state WHERE id = ?", (_SINGLETON_ID,)).fetchone()
        if row is None:
            return None
        return SchedulerStateRecord(
            started_at=dt_from_str(row["started_at"]) if row["started_at"] else None,
            last_heartbeat_at=dt_from_str(row["last_heartbeat_at"]) if row["last_heartbeat_at"] else None,
            next_morning_run_at=dt_from_str(row["next_morning_run_at"]) if row["next_morning_run_at"] else None,
            next_evening_run_at=dt_from_str(row["next_evening_run_at"]) if row["next_evening_run_at"] else None,
            last_scheduled_run_id=row["last_scheduled_run_id"],
            last_scheduled_execution_mode=row["last_scheduled_execution_mode"],
            last_scheduled_digest_type=row["last_scheduled_digest_type"],
            last_scheduled_started_at=dt_from_str(row["last_scheduled_started_at"]) if row["last_scheduled_started_at"] else None,
            last_scheduled_finished_at=dt_from_str(row["last_scheduled_finished_at"]) if row["last_scheduled_finished_at"] else None,
            last_scheduled_status=row["last_scheduled_status"],
        )
