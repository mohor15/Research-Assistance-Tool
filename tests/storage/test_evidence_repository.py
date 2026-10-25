"""EvidenceRepository: persistence, retrieval, and the article relationship."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.evidence import article_to_evidence
from research_assistant.storage.database import Database
from research_assistant.storage.repositories.article import ArticleRepository
from research_assistant.storage.repositories.evidence import EvidenceRepository

from tests.storage.conftest import make_article


def _evidence_for(article):
    evidence = article_to_evidence(article)
    assert evidence is not None
    return evidence


def test_persist_and_retrieve_evidence(database: Database):
    article = make_article("a1")
    evidence = _evidence_for(article)

    conn = database.connect()
    ArticleRepository().upsert(conn, article)
    EvidenceRepository().upsert(conn, evidence)
    conn.commit()

    fetched = EvidenceRepository().get(conn, evidence.evidence_id)
    conn.close()

    assert fetched is not None
    assert fetched.article_id == "a1"
    assert fetched.content == article.clean_text


def test_upsert_evidence_does_not_duplicate(database: Database):
    article = make_article("a1")
    evidence = _evidence_for(article)

    conn = database.connect()
    ArticleRepository().upsert(conn, article)
    EvidenceRepository().upsert(conn, evidence)
    EvidenceRepository().upsert(conn, evidence)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    conn.close()
    assert count == 1


def test_get_by_article_id_returns_the_article_relationship(database: Database):
    article = make_article("a1")
    evidence = _evidence_for(article)

    conn = database.connect()
    ArticleRepository().upsert(conn, article)
    EvidenceRepository().upsert(conn, evidence)
    conn.commit()

    related = EvidenceRepository().get_by_article_id(conn, "a1")
    conn.close()

    assert len(related) == 1
    assert related[0].evidence_id == evidence.evidence_id


def test_get_by_article_id_returns_empty_list_for_unrelated_article(database: Database):
    article = make_article("a1")
    evidence = _evidence_for(article)

    conn = database.connect()
    ArticleRepository().upsert(conn, article)
    EvidenceRepository().upsert(conn, evidence)
    conn.commit()

    related = EvidenceRepository().get_by_article_id(conn, "some-other-article")
    conn.close()

    assert related == []
