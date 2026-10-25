"""Two-tier CURRENT digest schema migration (v11 -> v12): removes the
`newsletter_selections.event_id REFERENCES events(id)` foreign key so a
Tier B article selection (a `tb:`-prefixed id, never a real event id) can
be stored in the same table. Verifies a database created under the OLD
(v11) schema -- the FK still present -- migrates safely with its existing
selection rows intact, and that a brand-new database has no such FK at
all."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from research_assistant.storage.database import CURRENT_SCHEMA_VERSION, Database

_V11_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS newsletter_selections (
    event_id TEXT PRIMARY KEY REFERENCES events(id),
    selected_at TEXT NOT NULL
);
"""
# Deliberately does NOT pre-create `events` -- SQLite does not validate a
# foreign key's target table at CREATE TABLE time, so this still
# faithfully reproduces "a v11 newsletter_selections table with the old
# FK declaration" without needing to duplicate the real `events` table's
# full (and schema-version-sensitive) column list; the real `_SCHEMA_SQL`
# creates the genuine `events` table fresh during `initialize()` since it
# does not exist yet in this stub.


def _build_pre_patch_database(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(_V11_SCHEMA_SQL)
    conn.execute(
        "INSERT INTO newsletter_selections (event_id, selected_at) VALUES ('ev1', '2026-10-09T12:00:00+00:00')"
    )
    conn.execute("INSERT INTO schema_meta(key, value) VALUES ('schema_version', '11')")
    conn.commit()
    conn.close()


def test_migrating_a_pre_patch_database_removes_the_event_fk(tmp_path: Path):
    db_path = tmp_path / "pre_existing.db"
    _build_pre_patch_database(db_path)

    db = Database(db_path)
    db.initialize()

    conn = db.connect()
    fk_rows = conn.execute("PRAGMA foreign_key_list(newsletter_selections)").fetchall()
    conn.close()

    assert fk_rows == []


def test_migration_preserves_existing_selection_rows(tmp_path: Path):
    db_path = tmp_path / "pre_existing.db"
    _build_pre_patch_database(db_path)

    db = Database(db_path)
    db.initialize()

    conn = db.connect()
    row = conn.execute("SELECT * FROM newsletter_selections WHERE event_id = 'ev1'").fetchone()
    count = conn.execute("SELECT COUNT(*) AS n FROM newsletter_selections").fetchone()["n"]
    conn.close()

    assert count == 1
    assert row["event_id"] == "ev1"
    assert row["selected_at"] == "2026-10-09T12:00:00+00:00"


def test_migrated_table_accepts_a_tier_b_selection_id(tmp_path: Path):
    """The whole point of the migration: a `tb:`-prefixed id (never a row
    in `events`) must insert cleanly after migration, where it would have
    violated the old FK."""
    db_path = tmp_path / "pre_existing.db"
    _build_pre_patch_database(db_path)

    db = Database(db_path)
    db.initialize()

    conn = db.connect()
    conn.execute(
        "INSERT INTO newsletter_selections (event_id, selected_at) VALUES ('tb:art1', '2026-10-12T12:00:00+00:00')"
    )
    conn.commit()
    row = conn.execute("SELECT * FROM newsletter_selections WHERE event_id = 'tb:art1'").fetchone()
    conn.close()

    assert row is not None


def test_schema_version_is_bumped_to_12(tmp_path: Path):
    db_path = tmp_path / "pre_existing.db"
    _build_pre_patch_database(db_path)

    db = Database(db_path)
    db.initialize()

    assert db.schema_version() == CURRENT_SCHEMA_VERSION
    assert CURRENT_SCHEMA_VERSION >= 12


def test_a_brand_new_database_has_no_event_fk_on_newsletter_selections(tmp_path: Path):
    db = Database(tmp_path / "fresh.db")
    db.initialize()

    conn = db.connect()
    fk_rows = conn.execute("PRAGMA foreign_key_list(newsletter_selections)").fetchall()
    conn.close()

    assert fk_rows == []


def test_migration_is_idempotent_running_initialize_twice_is_safe(tmp_path: Path):
    db_path = tmp_path / "pre_existing.db"
    _build_pre_patch_database(db_path)

    db = Database(db_path)
    db.initialize()
    db.initialize()  # a second call must never fail or duplicate/reset data

    conn = db.connect()
    row = conn.execute("SELECT * FROM newsletter_selections WHERE event_id = 'ev1'").fetchone()
    count = conn.execute("SELECT COUNT(*) AS n FROM newsletter_selections").fetchone()["n"]
    conn.close()

    assert count == 1
    assert row["selected_at"] == "2026-10-09T12:00:00+00:00"
