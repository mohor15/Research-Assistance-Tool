"""Phase 14 schema migration (v3 -> v4): new `scheduler_state` table +
7 new `pipeline_runs` columns, both additive-only. Verifies a database
created under the OLD (v3) schema migrates safely with its existing data
intact -- this is the exact safety property the implementation task
required before any production promotion."""

from __future__ import annotations

from pathlib import Path

from research_assistant.storage.database import CURRENT_SCHEMA_VERSION, Database

_V3_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    digest_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    status TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL DEFAULT '[]',
    counts_json TEXT NOT NULL DEFAULT '{}'
);
"""


def _build_pre_phase14_database(path: Path) -> None:
    """Simulates a real pre-Phase-14 production database: the OLD schema,
    with one real pipeline_runs row already in it, and no scheduler_state
    table at all."""
    import sqlite3

    conn = sqlite3.connect(path)
    conn.executescript(_V3_SCHEMA_SQL)
    conn.execute(
        "INSERT INTO pipeline_runs (run_id, digest_type, started_at, finished_at, status, diagnostics_json, counts_json) "
        "VALUES ('pre-existing-run', 'morning', '2026-09-30T12:00:00+00:00', '2026-09-30T12:05:00+00:00', "
        "'success', '[]', '{\"events_formed\": 3}')"
    )
    conn.execute("INSERT INTO schema_meta(key, value) VALUES ('schema_version', '3')")
    conn.commit()
    conn.close()


def test_migrating_a_pre_phase14_database_adds_scheduler_state_table(tmp_path: Path):
    db_path = tmp_path / "pre_existing.db"
    _build_pre_phase14_database(db_path)

    db = Database(db_path)
    db.initialize()

    conn = db.connect()
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()

    assert "scheduler_state" in tables


def test_migrating_a_pre_phase14_database_adds_pipeline_runs_columns_with_safe_defaults(tmp_path: Path):
    db_path = tmp_path / "pre_existing.db"
    _build_pre_phase14_database(db_path)

    db = Database(db_path)
    db.initialize()

    conn = db.connect()
    row = conn.execute("SELECT * FROM pipeline_runs WHERE run_id = 'pre-existing-run'").fetchone()
    conn.close()

    assert row["execution_mode"] == "manual"
    assert row["digest_attempted"] == 0
    assert row["digest_persisted"] == 0
    assert row["digest_persistence_status"] == "not_attempted"
    assert row["period_start"] is None
    assert row["digest_id"] is None


def test_migrating_a_pre_phase14_database_preserves_the_existing_run_row_unchanged(tmp_path: Path):
    db_path = tmp_path / "pre_existing.db"
    _build_pre_phase14_database(db_path)

    db = Database(db_path)
    db.initialize()

    conn = db.connect()
    row = conn.execute("SELECT * FROM pipeline_runs WHERE run_id = 'pre-existing-run'").fetchone()
    count = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
    conn.close()

    assert count == 1  # not duplicated, not dropped
    assert row["status"] == "success"
    assert row["counts_json"] == '{"events_formed": 3}'


def test_schema_version_is_bumped_to_current_after_migration(tmp_path: Path):
    db_path = tmp_path / "pre_existing.db"
    _build_pre_phase14_database(db_path)

    db = Database(db_path)
    db.initialize()

    # Not hardcoded to a specific number -- later phases (e.g. Phase 15's
    # grounding_type columns) legitimately bump this further; the real
    # assertion is that a pre-existing database ends up at whatever the
    # CURRENT running code considers current, not stuck at an old version.
    assert db.schema_version() == CURRENT_SCHEMA_VERSION


def test_a_fresh_database_already_has_all_phase14_columns_and_tables(tmp_path: Path):
    db = Database(tmp_path / "fresh.db")
    db.initialize()

    conn = db.connect()
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(pipeline_runs)")}
    conn.close()

    assert "scheduler_state" in tables
    assert {
        "execution_mode", "period_start", "period_end",
        "digest_attempted", "digest_persisted", "digest_persistence_status", "digest_id",
    }.issubset(columns)


def test_migration_is_idempotent_when_run_twice(tmp_path: Path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()
    db.initialize()  # must not raise, must not duplicate columns

    conn = db.connect()
    columns = [row["name"] for row in conn.execute("PRAGMA table_info(pipeline_runs)")]
    conn.close()
    assert columns.count("execution_mode") == 1
