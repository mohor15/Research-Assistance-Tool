"""PipelineRun persistence (Phase 8) — deterministic run_id, upsert-safe
(a scheduler retry/restart never duplicates a run record).

Phase 14: extended with explicit execution/digest-persistence metadata
(`execution_mode`, `period_start`/`period_end`, `digest_attempted`/
`digest_persisted`/`digest_persistence_status`/`digest_id`) so a reader can
distinguish "scheduled vs manual" and "no digest because nothing qualified"
vs "no digest because persistence failed" vs "no digest because the run
never reached that stage" without inference -- see
`research_assistant.models.enums.DigestPersistenceStatus`.
"""

from __future__ import annotations

import sqlite3
from typing import List, Optional

from research_assistant.models.digest import DigestType
from research_assistant.models.enums import DigestPersistenceStatus, ExecutionMode
from research_assistant.pipeline.state import PipelineDiagnostic
from research_assistant.storage.serde import dt_from_str, dt_to_str, json_dump, json_load


class RunRepository:
    def upsert(self, conn: sqlite3.Connection, run) -> None:  # run: PipelineRunResult (typed at call site to avoid an import cycle)
        conn.execute(
            """
            INSERT INTO pipeline_runs (
                run_id, digest_type, started_at, finished_at, status, diagnostics_json, counts_json,
                execution_mode, period_start, period_end,
                digest_attempted, digest_persisted, digest_persistence_status, digest_id
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
                digest_type=excluded.digest_type, started_at=excluded.started_at, finished_at=excluded.finished_at,
                status=excluded.status, diagnostics_json=excluded.diagnostics_json, counts_json=excluded.counts_json,
                execution_mode=excluded.execution_mode, period_start=excluded.period_start, period_end=excluded.period_end,
                digest_attempted=excluded.digest_attempted, digest_persisted=excluded.digest_persisted,
                digest_persistence_status=excluded.digest_persistence_status, digest_id=excluded.digest_id
            """,
            (
                run.run_id, run.digest_type.value, dt_to_str(run.started_at), dt_to_str(run.finished_at),
                run.status.value, json_dump([d.model_dump(mode="json") for d in run.diagnostics]), json_dump(run.counts),
                run.execution_mode.value, dt_to_str(run.period_start) if run.period_start else None,
                dt_to_str(run.period_end) if run.period_end else None,
                1 if run.digest_attempted else 0, 1 if run.digest_persisted else 0,
                run.digest_persistence_status.value, run.digest_id,
            ),
        )

    def get(self, conn: sqlite3.Connection, run_id: str):
        row = conn.execute("SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)).fetchone()
        return _row_to_run(row) if row else None

    def get_recent(self, conn: sqlite3.Connection, *, digest_type: Optional[DigestType] = None, limit: int = 20) -> List:
        if digest_type is not None:
            rows = conn.execute(
                "SELECT * FROM pipeline_runs WHERE digest_type = ? ORDER BY started_at DESC LIMIT ?",
                (digest_type.value, limit),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [_row_to_run(row) for row in rows]


def _row_to_run(row: sqlite3.Row):
    # Imported lazily to avoid storage <-> pipeline import-order coupling at
    # module load time (pipeline.runner already imports research_assistant.storage).
    from research_assistant.pipeline.runner import PipelineRunResult, RunStatus

    row_keys = row.keys()
    return PipelineRunResult(
        run_id=row["run_id"], digest_type=DigestType(row["digest_type"]), started_at=dt_from_str(row["started_at"]),
        finished_at=dt_from_str(row["finished_at"]), status=RunStatus(row["status"]),
        diagnostics=[PipelineDiagnostic.model_validate(d) for d in json_load(row["diagnostics_json"], [])],
        counts=json_load(row["counts_json"], {}),
        # Pre-Phase-14 rows (migrated via _add_column_if_missing) carry the
        # column's SQL-level DEFAULT ('manual' / 'not_attempted' / 0) --
        # these are never re-derived here, only read back as stored.
        execution_mode=ExecutionMode(row["execution_mode"]) if "execution_mode" in row_keys and row["execution_mode"] else ExecutionMode.MANUAL,
        period_start=dt_from_str(row["period_start"]) if "period_start" in row_keys and row["period_start"] else None,
        period_end=dt_from_str(row["period_end"]) if "period_end" in row_keys and row["period_end"] else None,
        digest_attempted=bool(row["digest_attempted"]) if "digest_attempted" in row_keys else False,
        digest_persisted=bool(row["digest_persisted"]) if "digest_persisted" in row_keys else False,
        digest_persistence_status=(
            DigestPersistenceStatus(row["digest_persistence_status"])
            if "digest_persistence_status" in row_keys and row["digest_persistence_status"]
            else DigestPersistenceStatus.NOT_ATTEMPTED
        ),
        digest_id=row["digest_id"] if "digest_id" in row_keys else None,
    )
