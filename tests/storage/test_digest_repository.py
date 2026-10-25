"""DigestRepository: persistence, event references (ranked order), and themes."""

from __future__ import annotations

from research_assistant.models.digest import DigestType
from research_assistant.storage.database import Database
from research_assistant.storage.repositories.article import ArticleRepository
from research_assistant.storage.repositories.digest import DigestRepository
from research_assistant.storage.repositories.event import EventRepository
from research_assistant.storage.repositories.research_item import ResearchItemRepository

from tests.storage.conftest import make_article, make_digest, make_event, make_research_item


def _seed_events(conn):
    for suffix in ("1", "2"):
        ArticleRepository().upsert(conn, make_article(f"art{suffix}"))
        ResearchItemRepository().upsert(conn, make_research_item(f"ri{suffix}", article_id=f"art{suffix}"))
        EventRepository().upsert(conn, make_event(f"ev{suffix}", research_item_ids=[f"ri{suffix}"], article_ids=[f"art{suffix}"]))


def test_persist_and_retrieve_digest(database: Database):
    conn = database.connect()
    _seed_events(conn)
    DigestRepository().upsert(conn, make_digest("dg1", event_ids=["ev1", "ev2"]))
    conn.commit()

    fetched = DigestRepository().get(conn, "dg1")
    conn.close()

    assert fetched is not None
    assert fetched.digest_type == DigestType.MORNING


def test_event_references_preserve_ranked_order(database: Database):
    conn = database.connect()
    _seed_events(conn)
    DigestRepository().upsert(conn, make_digest("dg1", event_ids=["ev2", "ev1"]))  # deliberately not alphabetical
    conn.commit()

    fetched = DigestRepository().get(conn, "dg1")
    conn.close()

    assert fetched.events == ["ev2", "ev1"]


def test_themes_are_persisted_with_their_event_ids(database: Database):
    conn = database.connect()
    _seed_events(conn)
    DigestRepository().upsert(conn, make_digest("dg1", event_ids=["ev1", "ev2"]))
    conn.commit()

    fetched = DigestRepository().get(conn, "dg1")
    conn.close()

    assert len(fetched.themes) == 1
    assert fetched.themes[0].label == "enterprise AI"
    assert fetched.themes[0].event_ids == ["ev1", "ev2"]


def test_upsert_does_not_duplicate(database: Database):
    conn = database.connect()
    _seed_events(conn)
    DigestRepository().upsert(conn, make_digest("dg1"))
    DigestRepository().upsert(conn, make_digest("dg1"))
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM digests").fetchone()[0]
    conn.close()
    assert count == 1


def test_get_recent_filters_by_digest_type(database: Database):
    conn = database.connect()
    _seed_events(conn)
    DigestRepository().upsert(conn, make_digest("dg1", digest_type=DigestType.MORNING))
    DigestRepository().upsert(conn, make_digest("dg2", digest_type=DigestType.EVENING))
    conn.commit()

    morning = DigestRepository().get_recent(conn, digest_type=DigestType.MORNING)
    conn.close()

    assert [d.id for d in morning] == ["dg1"]
