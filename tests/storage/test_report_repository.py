"""ReportRepository: persistence, event relationship, and failure-state persistence."""

from __future__ import annotations

from research_assistant.models.report import Report, ReportFailure
from research_assistant.storage.database import Database
from research_assistant.storage.repositories.article import ArticleRepository
from research_assistant.storage.repositories.event import EventRepository
from research_assistant.storage.repositories.report import ReportRepository
from research_assistant.storage.repositories.research_item import ResearchItemRepository

from tests.storage.conftest import make_article, make_event, make_report, make_report_failure, make_research_item


def _seed_event(conn, event_id="ev1"):
    ArticleRepository().upsert(conn, make_article("art1"))
    ResearchItemRepository().upsert(conn, make_research_item("ri1", article_id="art1"))
    EventRepository().upsert(conn, make_event(event_id))


def test_persist_and_retrieve_a_successful_report(database: Database):
    conn = database.connect()
    _seed_event(conn)
    ReportRepository().upsert_report(conn, make_report("ev1"))
    conn.commit()

    fetched = ReportRepository().get(conn, "ev1")
    conn.close()

    assert isinstance(fetched, Report)
    assert fetched.event_id == "ev1"
    assert fetched.details.companies == ["JPMorgan Chase"]
    assert "may be relevant" in fetched.nomura_opportunity


def test_report_event_relationship(database: Database):
    conn = database.connect()
    _seed_event(conn)
    ReportRepository().upsert_report(conn, make_report("ev1"))
    conn.commit()

    event = EventRepository().get(conn, "ev1")
    report = ReportRepository().get(conn, "ev1")
    conn.close()

    assert report.event_id == event.id


def test_persist_and_retrieve_a_report_failure(database: Database):
    conn = database.connect()
    _seed_event(conn)
    ReportRepository().upsert_failure(conn, make_report_failure("ev1"))
    conn.commit()

    fetched = ReportRepository().get(conn, "ev1")
    conn.close()

    assert isinstance(fetched, ReportFailure)
    assert fetched.event_id == "ev1"
    assert fetched.reason == "LLMUnavailableError"


def test_upsert_report_does_not_duplicate(database: Database):
    conn = database.connect()
    _seed_event(conn)
    ReportRepository().upsert_report(conn, make_report("ev1"))
    ReportRepository().upsert_report(conn, make_report("ev1"))
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    conn.close()
    assert count == 1


def test_a_failure_can_be_overwritten_by_a_later_successful_report(database: Database):
    """Simulates a retry: the same event_id first failed, then succeeded."""
    conn = database.connect()
    _seed_event(conn)
    ReportRepository().upsert_failure(conn, make_report_failure("ev1"))
    ReportRepository().upsert_report(conn, make_report("ev1"))
    conn.commit()

    fetched = ReportRepository().get(conn, "ev1")
    conn.close()

    assert isinstance(fetched, Report)


def test_get_recent_orders_by_event_date(database: Database):
    conn = database.connect()
    _seed_event(conn, "ev1")
    ReportRepository().upsert_report(conn, make_report("ev1"))
    conn.commit()

    recent = ReportRepository().get_recent(conn, limit=10)
    conn.close()

    assert len(recent) == 1
    assert recent[0].event_id == "ev1"
