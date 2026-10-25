"""Newsletter selection persistence (Phase 16) -- a bare event_id pointer
table that must never duplicate a selection and must never touch the
underlying Event row. `event_id` is a real FK against `events`, so every
test persists a real Event first, exactly like `event_articles`/
`digest_events` tests do elsewhere in this package."""

from __future__ import annotations

from datetime import timedelta

from research_assistant.storage.persistence_service import PersistenceService

from tests.storage.conftest import D, make_article, make_event, make_research_item


def _persist_event(service: PersistenceService, event_id: str = "ev1") -> None:
    article_id = f"art-{event_id}"
    research_item_id = f"ri-{event_id}"
    service.persist_pipeline_result(
        articles=[make_article(article_id)],
        research_items=[make_research_item(research_item_id, article_id=article_id)],
        events=[make_event(event_id, research_item_ids=[research_item_id], article_ids=[article_id])],
    )


def test_add_then_list_returns_the_selected_event_id(persistence_service: PersistenceService):
    _persist_event(persistence_service)
    conn = persistence_service.database.connect()
    try:
        persistence_service.newsletter.add(conn, "ev1", selected_at=D)
        assert persistence_service.newsletter.list_ids(conn) == ["ev1"]
    finally:
        conn.close()


def test_adding_the_same_event_id_twice_does_not_duplicate(persistence_service: PersistenceService):
    _persist_event(persistence_service)
    conn = persistence_service.database.connect()
    try:
        persistence_service.newsletter.add(conn, "ev1", selected_at=D)
        persistence_service.newsletter.add(conn, "ev1", selected_at=D + timedelta(seconds=1))
        assert persistence_service.newsletter.list_ids(conn) == ["ev1"]
    finally:
        conn.close()


def test_remove_deletes_the_selection(persistence_service: PersistenceService):
    _persist_event(persistence_service)
    conn = persistence_service.database.connect()
    try:
        persistence_service.newsletter.add(conn, "ev1", selected_at=D)
        persistence_service.newsletter.remove(conn, "ev1")
        assert persistence_service.newsletter.list_ids(conn) == []
    finally:
        conn.close()


def test_is_selected_reflects_current_state(persistence_service: PersistenceService):
    _persist_event(persistence_service)
    conn = persistence_service.database.connect()
    try:
        assert persistence_service.newsletter.is_selected(conn, "ev1") is False
        persistence_service.newsletter.add(conn, "ev1", selected_at=D)
        assert persistence_service.newsletter.is_selected(conn, "ev1") is True
    finally:
        conn.close()


def test_selection_order_is_preserved(persistence_service: PersistenceService):
    _persist_event(persistence_service, "ev1")
    _persist_event(persistence_service, "ev2")
    conn = persistence_service.database.connect()
    try:
        persistence_service.newsletter.add(conn, "ev2", selected_at=D)
        persistence_service.newsletter.add(conn, "ev1", selected_at=D + timedelta(seconds=1))
        assert persistence_service.newsletter.list_ids(conn) == ["ev2", "ev1"]
    finally:
        conn.close()


def test_selecting_an_event_does_not_mutate_the_underlying_event_row(persistence_service: PersistenceService):
    _persist_event(persistence_service)

    conn = persistence_service.database.connect()
    try:
        before = persistence_service.events.get(conn, "ev1")
        persistence_service.newsletter.add(conn, "ev1", selected_at=D)
        after = persistence_service.events.get(conn, "ev1")
    finally:
        conn.close()

    assert before == after
