"""Reliability patch schema migration (v10 -> v11): one new, additive
`candidate_backlog.defer_count` column. Verifies a database created under
the OLD (v10) schema -- i.e. `candidate_backlog` already exists but without
this column -- migrates safely with its existing row intact, and that a
brand-new database already gets the column from the `CREATE TABLE`
statement itself."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from research_assistant.storage.database import CURRENT_SCHEMA_VERSION, Database

_V10_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS candidate_backlog (
    article_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    triage_relevant INTEGER,
    triage_confidence REAL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    first_seen_run_id TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_attempt_run_id TEXT,
    last_attempt_at TEXT,
    research_item_id TEXT,
    updated_at TEXT NOT NULL
);
"""


def _build_pre_patch_database(path: Path) -> None:
    """Simulates a real pre-reliability-patch (v10) production database:
    `candidate_backlog` already exists, with one real pending row, but has
    no `defer_count` column at all."""
    conn = sqlite3.connect(path)
    conn.executescript(_V10_SCHEMA_SQL)
    conn.execute(
        "INSERT INTO candidate_backlog (article_id, agent, status, reason, attempt_count, first_seen_run_id, "
        "first_seen_at, updated_at) VALUES ('art1', 'ai_product', 'pending_triage_budget', 'pre-existing row', "
        "2, 'old-run', '2026-10-09T12:00:00+00:00', '2026-10-09T12:00:00+00:00')"
    )
    conn.execute("INSERT INTO schema_meta(key, value) VALUES ('schema_version', '10')")
    conn.commit()
    conn.close()


def test_migrating_a_pre_patch_database_adds_defer_count_column(tmp_path: Path):
    db_path = tmp_path / "pre_existing.db"
    _build_pre_patch_database(db_path)

    db = Database(db_path)
    db.initialize()

    conn = db.connect()
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(candidate_backlog)")}
    conn.close()

    assert "defer_count" in columns


def test_migrating_preserves_the_existing_rows_defer_count_as_zero_not_a_guess(tmp_path: Path):
    db_path = tmp_path / "pre_existing.db"
    _build_pre_patch_database(db_path)

    db = Database(db_path)
    db.initialize()

    conn = db.connect()
    row = conn.execute("SELECT * FROM candidate_backlog WHERE article_id = 'art1'").fetchone()
    conn.close()

    assert row["defer_count"] == 0  # unknown deferral history before this column existed -- 0, never guessed upward
    assert row["attempt_count"] == 2  # the pre-existing column's real data is untouched
    assert row["status"] == "pending_triage_budget"  # the pre-existing row survives migration intact


def test_schema_version_is_bumped_to_11(tmp_path: Path):
    db_path = tmp_path / "pre_existing.db"
    _build_pre_patch_database(db_path)

    db = Database(db_path)
    db.initialize()

    assert db.schema_version() == CURRENT_SCHEMA_VERSION
    assert CURRENT_SCHEMA_VERSION >= 11


def test_a_brand_new_database_already_has_defer_count_from_the_create_statement(tmp_path: Path):
    db = Database(tmp_path / "fresh.db")
    db.initialize()

    conn = db.connect()
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(candidate_backlog)")}
    conn.close()

    assert "defer_count" in columns


def test_migration_is_idempotent_running_initialize_twice_is_safe(tmp_path: Path):
    db_path = tmp_path / "pre_existing.db"
    _build_pre_patch_database(db_path)

    db = Database(db_path)
    db.initialize()
    db.initialize()  # a second call must never fail or duplicate/reset data

    conn = db.connect()
    row = conn.execute("SELECT * FROM candidate_backlog WHERE article_id = 'art1'").fetchone()
    count = conn.execute("SELECT COUNT(*) AS n FROM candidate_backlog").fetchone()["n"]
    conn.close()

    assert count == 1
    assert row["attempt_count"] == 2
