"""ArticleRepository: insert, retrieve, upsert, canonical-URL idempotency,
content-hash idempotency."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.storage.database import Database
from research_assistant.storage.repositories.article import ArticleRepository

from tests.storage.conftest import D, make_article


def test_insert_and_retrieve_article(database: Database):
    repo = ArticleRepository()
    conn = database.connect()
    repo.upsert(conn, make_article("a1"))
    conn.commit()

    fetched = repo.get(conn, "a1")
    conn.close()

    assert fetched is not None
    assert fetched.id == "a1"
    assert fetched.title == "An article title"
    assert fetched.clean_text == "Clean article text."
    assert fetched.metadata == {"reason": "test"}


def test_get_missing_article_returns_none(database: Database):
    conn = database.connect()
    fetched = ArticleRepository().get(conn, "does-not-exist")
    conn.close()
    assert fetched is None


def test_upsert_updates_existing_row_rather_than_duplicating(database: Database):
    repo = ArticleRepository()
    conn = database.connect()
    repo.upsert(conn, make_article("a1", clean_text="Original text."))
    repo.upsert(conn, make_article("a1", clean_text="Updated text."))
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    fetched = repo.get(conn, "a1")
    conn.close()

    assert count == 1
    assert fetched.clean_text == "Updated text."


def test_get_by_canonical_url_finds_the_article(database: Database):
    repo = ArticleRepository()
    conn = database.connect()
    repo.upsert(conn, make_article("a1", canonical_url="https://example.com/canonical"))
    conn.commit()

    fetched = repo.get_by_canonical_url(conn, "https://example.com/canonical")
    conn.close()

    assert fetched is not None
    assert fetched.id == "a1"


def test_canonical_url_idempotency_prevents_reprocessing_the_same_article(database: Database):
    """Simulates the real idempotency use case: before normalizing a
    re-crawled article, a caller checks get_by_canonical_url() first."""
    repo = ArticleRepository()
    conn = database.connect()
    repo.upsert(conn, make_article("a1", canonical_url="https://example.com/story"))
    conn.commit()

    already_known = repo.get_by_canonical_url(conn, "https://example.com/story")
    conn.close()

    assert already_known is not None
    assert already_known.id == "a1"


def test_content_hash_idempotency_detects_duplicate_content_under_a_different_url(database: Database):
    repo = ArticleRepository()
    conn = database.connect()
    repo.upsert(conn, make_article("a1", url="https://mirror1.example.com/x", content_hash="shared-hash"))
    conn.commit()

    duplicate = repo.get_by_content_hash(conn, "shared-hash")
    conn.close()

    assert duplicate is not None
    assert duplicate.id == "a1"


def test_get_by_content_hash_returns_none_when_hash_is_null(database: Database):
    repo = ArticleRepository()
    conn = database.connect()
    repo.upsert(conn, make_article("a1", content_hash=None))
    conn.commit()

    fetched = repo.get_by_content_hash(conn, "some-hash-that-was-never-stored")
    conn.close()

    assert fetched is None


def test_get_by_source_name_returns_only_that_entitys_articles(database: Database):
    repo = ArticleRepository()
    conn = database.connect()
    repo.upsert(conn, make_article("a1", company="JPMorgan Chase"))
    repo.upsert(conn, make_article("a2", company="Bank of America"))
    conn.commit()

    jpmorgan_articles = repo.get_by_source_name(conn, "JPMorgan Chase")
    conn.close()

    assert {a.id for a in jpmorgan_articles} == {"a1"}


def test_get_by_source_name_with_no_articles_returns_empty_list(database: Database):
    conn = database.connect()
    result = ArticleRepository().get_by_source_name(conn, "Morgan Stanley")
    conn.close()
    assert result == []


def test_get_by_source_name_period_bound_excludes_out_of_window_dated_articles(database: Database):
    repo = ArticleRepository()
    conn = database.connect()
    repo.upsert(conn, make_article("in", company="JPMorgan Chase", publication_date=D))
    repo.upsert(conn, make_article("old", company="JPMorgan Chase", publication_date=D - timedelta(days=30)))
    conn.commit()

    result = repo.get_by_source_name(conn, "JPMorgan Chase", since=D - timedelta(hours=1), until=D + timedelta(hours=1))
    conn.close()

    assert {a.id for a in result} == {"in"}


def test_get_by_source_name_period_bound_never_excludes_a_missing_publication_date(database: Database):
    repo = ArticleRepository()
    conn = database.connect()
    repo.upsert(conn, make_article("dateless", company="JPMorgan Chase", publication_date=None))
    conn.commit()

    result = repo.get_by_source_name(conn, "JPMorgan Chase", since=D - timedelta(hours=1), until=D + timedelta(hours=1))
    conn.close()

    assert {a.id for a in result} == {"dateless"}
