"""Database creation, schema creation, foreign keys, indexes, clean and
repeated initialization."""

from __future__ import annotations

from pathlib import Path

from research_assistant.storage.database import CURRENT_SCHEMA_VERSION, Database

_EXPECTED_TABLES = {
    "schema_meta", "articles", "evidence", "research_items", "events",
    "event_research_items", "event_articles", "reports", "digests", "digest_events",
}

_EXPECTED_INDEXES = {
    "idx_articles_canonical_url", "idx_articles_content_hash", "idx_articles_publication_date",
    "idx_evidence_article_id", "idx_research_items_article_id", "idx_research_items_agent",
    "idx_research_items_company", "idx_research_items_event_type", "idx_events_event_date",
    "idx_events_primary_company", "idx_events_event_type", "idx_events_relevance_score",
    "idx_events_confidence_score", "idx_digests_digest_type", "idx_digests_generated_at",
}


def test_database_creates_file_in_a_clean_temporary_directory(tmp_path: Path):
    db_path = tmp_path / "nested" / "dir" / "research.db"
    assert not db_path.exists()

    db = Database(db_path)
    db.initialize()

    assert db_path.exists()


def test_schema_creates_all_expected_tables(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    conn = db.connect()
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()

    assert _EXPECTED_TABLES.issubset(tables)


def test_schema_creates_all_expected_indexes(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    conn = db.connect()
    indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")}
    conn.close()

    assert _EXPECTED_INDEXES.issubset(indexes)


def test_foreign_keys_are_enforced(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    conn = db.connect()

    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    import pytest
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO evidence (evidence_id, article_id, source_url, canonical_url, retrieved_at, "
            "content, content_hash, source_type, source_priority, extraction_method) "
            "VALUES ('e1', 'nonexistent-article', 'u', 'u', '2025-01-01T00:00:00+00:00', 'c', 'h', "
            "'primary', 'high', 'bs4_fallback')"
        )
    conn.close()


def test_clean_initialization_records_schema_version(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    db.initialize()

    assert db.schema_version() == CURRENT_SCHEMA_VERSION


def test_reinitializing_an_existing_database_preserves_data(tmp_path: Path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()
    conn = db.connect()
    conn.execute(
        "INSERT INTO articles (id, url, canonical_url, domain, source_name, source_type, source_priority, "
        "publication_date_precision, retrieved_at, extraction_method, extraction_status) "
        "VALUES ('a1','u','cu','d','s','primary','high','unknown','2025-01-01T00:00:00+00:00','bs4_fallback','success')"
    )
    conn.commit()
    conn.close()

    # Re-initialize against the same existing database file.
    db2 = Database(db_path)
    db2.initialize()

    conn = db2.connect()
    count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()
    assert count == 1


def test_connect_returns_row_factory_that_supports_column_access(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    conn = db.connect()
    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES ('test_key', 'test_value') "
        "ON CONFLICT(key) DO NOTHING"
    )
    row = conn.execute("SELECT * FROM schema_meta WHERE key = 'test_key'").fetchone()
    conn.close()
    assert row["value"] == "test_value"
