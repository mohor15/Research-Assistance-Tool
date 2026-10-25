"""EventRepository: persistence, upsert, research-item/article relationships,
scoring-breakdown persistence, provenance, and the historical query methods."""

from __future__ import annotations

from datetime import timedelta

from research_assistant.models.enums import EventType
from research_assistant.storage.database import Database
from research_assistant.storage.repositories.article import ArticleRepository
from research_assistant.storage.repositories.event import EventRepository
from research_assistant.storage.repositories.research_item import ResearchItemRepository

from tests.storage.conftest import D, make_article, make_event, make_research_item


def _seed(conn, *, event_id="ev1", company="JPMorgan Chase", event_type=EventType.PRODUCT_LAUNCH, event_date=D, relevance=0.85, confidence=0.75):
    ArticleRepository().upsert(conn, make_article("art1", company=company))
    ResearchItemRepository().upsert(conn, make_research_item("ri1", article_id="art1", company=company, event_type=event_type, publication_date=event_date))
    event = make_event(event_id, company=company, event_type=event_type, event_date=event_date, relevance_score=relevance, confidence_score=confidence)
    EventRepository().upsert(conn, event)
    return event


def test_persist_and_retrieve_event(database: Database):
    conn = database.connect()
    _seed(conn)
    conn.commit()

    fetched = EventRepository().get(conn, "ev1")
    conn.close()

    assert fetched is not None
    assert fetched.canonical_title == "An event title"
    assert fetched.primary_company == "JPMorgan Chase"
    assert fetched.relevance_score == 0.85


def test_upsert_updates_rather_than_duplicates(database: Database):
    conn = database.connect()
    _seed(conn, relevance=0.5)
    _seed(conn, relevance=0.9)  # same event_id, different score
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    fetched = EventRepository().get(conn, "ev1")
    conn.close()

    assert count == 1
    assert fetched.relevance_score == 0.9


def test_research_item_relationship_is_preserved(database: Database):
    conn = database.connect()
    ArticleRepository().upsert(conn, make_article("art1"))
    ArticleRepository().upsert(conn, make_article("art2"))
    ResearchItemRepository().upsert(conn, make_research_item("ri1", article_id="art1"))
    ResearchItemRepository().upsert(conn, make_research_item("ri2", article_id="art2"))
    event = make_event("ev1", research_item_ids=["ri1", "ri2"], article_ids=["art1", "art2"])
    EventRepository().upsert(conn, event)
    conn.commit()

    fetched = EventRepository().get(conn, "ev1")
    conn.close()

    assert fetched.research_items == ["ri1", "ri2"]


def test_article_relationship_is_preserved(database: Database):
    conn = database.connect()
    ArticleRepository().upsert(conn, make_article("art1"))
    ArticleRepository().upsert(conn, make_article("art2"))
    ResearchItemRepository().upsert(conn, make_research_item("ri1", article_id="art1"))
    ResearchItemRepository().upsert(conn, make_research_item("ri2", article_id="art2"))
    event = make_event("ev1", research_item_ids=["ri1", "ri2"], article_ids=["art1", "art2"])
    EventRepository().upsert(conn, event)
    conn.commit()

    fetched = EventRepository().get(conn, "ev1")
    conn.close()

    assert fetched.articles == ["art1", "art2"]


def test_scoring_breakdown_round_trips_exactly(database: Database):
    conn = database.connect()
    event = _seed(conn)
    conn.commit()

    fetched = EventRepository().get(conn, "ev1")
    conn.close()

    assert fetched.scoring_breakdown == event.scoring_breakdown


def test_provenance_source_refs_round_trip_exactly(database: Database):
    conn = database.connect()
    event = _seed(conn)
    conn.commit()

    fetched = EventRepository().get(conn, "ev1")
    conn.close()

    assert fetched.primary_source == event.primary_source
    assert fetched.primary_source.url == "https://example.com/art1"


def test_get_recent_orders_by_event_date_descending(database: Database):
    conn = database.connect()
    _seed(conn, event_id="ev1", event_date=D)
    _seed(conn, event_id="ev2", event_date=D + timedelta(days=1))
    conn.commit()

    recent = EventRepository().get_recent(conn, limit=10)
    conn.close()

    assert [e.id for e in recent] == ["ev2", "ev1"]


def test_get_by_company_filters_correctly(database: Database):
    conn = database.connect()
    _seed(conn, event_id="ev1", company="JPMorgan Chase")
    _seed(conn, event_id="ev2", company="OpenAI")
    conn.commit()

    results = EventRepository().get_by_company(conn, "JPMorgan Chase")
    conn.close()

    assert [e.id for e in results] == ["ev1"]


def test_get_by_event_type_filters_correctly(database: Database):
    conn = database.connect()
    _seed(conn, event_id="ev1", event_type=EventType.FUNDING)
    _seed(conn, event_id="ev2", event_type=EventType.ACQUISITION)
    conn.commit()

    results = EventRepository().get_by_event_type(conn, EventType.FUNDING)
    conn.close()

    assert [e.id for e in results] == ["ev1"]


def test_get_by_min_relevance_filters_correctly(database: Database):
    conn = database.connect()
    _seed(conn, event_id="ev1", relevance=0.9)
    _seed(conn, event_id="ev2", relevance=0.3)
    conn.commit()

    results = EventRepository().get_by_min_relevance(conn, 0.5)
    conn.close()

    assert [e.id for e in results] == ["ev1"]


def test_get_by_min_confidence_filters_correctly(database: Database):
    conn = database.connect()
    _seed(conn, event_id="ev1", confidence=0.9)
    _seed(conn, event_id="ev2", confidence=0.3)
    conn.commit()

    results = EventRepository().get_by_min_confidence(conn, 0.5)
    conn.close()

    assert [e.id for e in results] == ["ev1"]


def test_get_by_agent_filters_correctly(database: Database):
    from research_assistant.models.enums import AgentName

    conn = database.connect()
    ArticleRepository().upsert(conn, make_article("art1"))
    ArticleRepository().upsert(conn, make_article("art2"))
    ResearchItemRepository().upsert(conn, make_research_item("ri1", article_id="art1", agent=AgentName.PEER))
    ResearchItemRepository().upsert(conn, make_research_item("ri2", article_id="art2", agent=AgentName.ANALYST))
    EventRepository().upsert(conn, make_event("ev1", research_item_ids=["ri1"], article_ids=["art1"]))
    EventRepository().upsert(conn, make_event("ev2", research_item_ids=["ri2"], article_ids=["art2"]))
    conn.commit()

    results = EventRepository().get_by_agent(conn, "peer")
    conn.close()

    assert [e.id for e in results] == ["ev1"]
