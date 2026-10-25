"""Regression test for the production persistence failure on run
`e05f28043756b4f7`:

    UNIQUE constraint failed: event_articles.event_id, event_articles.article_id

Root cause (confirmed by direct code inspection, not inferred): `events`
has one PRIMARY KEY (event_id, article_id), and `EventRepository.upsert()`
builds its `event_articles` insert directly from `event.articles` with NO
deduplication of its own -- it trusts the caller to have already produced
a list with no repeated article id. The ONLY in-repo path that builds a
fresh `Event` (`events.service.EventService._score_draft`, fed by
`events.builder.build_event_draft`) already deduplicates
(`articles=list(dict.fromkeys(...))`, see that function's own docstring:
"the same article can be routed to more than one agent, so two distinct
ResearchItems in this cluster may share an article_id"). That existing
dedup is proven below (`test_build_event_draft_already_dedupes...`) to
already prevent this for the one real caller -- which is why the failure
cannot be reproduced through `EventService.build_events()` under the
CURRENT codebase, and most likely means the historical run predates that
existing dedup line.

What was NOT previously true: `EventRepository.upsert()` itself has no
defense of its own -- ANY caller that builds an `Event` without going
through `build_event_draft` (nothing in `Event`'s own Pydantic model
forbids a duplicate in `articles: List[str]`) hits this exact SQLite
IntegrityError. This file reproduces THAT gap directly against a real,
temporary, schema-initialized SQLite database (never `data/research.db`),
then proves the minimal fix: an explicit, order-preserving dedup
(the same `dict.fromkeys` idiom already used in `events/builder.py`,
applied at the actual write boundary instead of only at one caller).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from research_assistant.events.models import (
    ClusterMetadata,
    ConfidenceBreakdown,
    Event,
    EventScoringBreakdown,
    RelevanceBreakdown,
    SourceRef,
)
from research_assistant.events.service import EventService
from research_assistant.models.enums import AgentName, SourcePriority, SourceType
from research_assistant.storage.database import Database
from research_assistant.storage.repositories.article import ArticleRepository
from research_assistant.storage.repositories.event import EventRepository
from research_assistant.storage.repositories.research_item import ResearchItemRepository

from tests.events.conftest import config, make_item  # noqa: F401 -- `config` is a pytest fixture


def _make_article(article_id: str):
    from datetime import datetime, timezone

    from research_assistant.models.article import Article, DatePrecision

    return Article(
        id=article_id, url=f"https://example.com/{article_id}", canonical_url=f"https://example.com/{article_id}",
        domain="example.com", source_name="Example Co", source_type=SourceType.PRIMARY,
        source_priority=SourcePriority.HIGH, publisher="Example Co", title="t", author=None,
        publication_date=None, publication_date_precision=DatePrecision.UNKNOWN,
        retrieved_at=datetime.now(timezone.utc),
        raw_text="x" * 50, clean_text="x" * 50, content_hash=f"hash-{article_id}",
        extraction_method="trafilatura", extraction_status="success",
    )


def _persist_research_items_for_fk(conn, research_item_ids: list, article_id: str) -> None:
    """`event_research_items.research_item_id` is a real FOREIGN KEY
    (enforced -- `Database.connect()` turns `PRAGMA foreign_keys = ON`) --
    satisfy it the same way a real pipeline run would have, one minimal
    persisted ResearchItem per id."""
    repo = ResearchItemRepository()
    for research_item_id in research_item_ids:
        repo.upsert(conn, make_item(research_item_id, company="Example Co", article_id=article_id))


def _hand_built_event(event_id: str, *, research_item_ids: list, article_ids: list) -> Event:
    """Bypasses `build_event_draft` entirely -- exactly what any future (or
    historical, pre-dedup-fix) caller could hand `EventRepository.upsert()`
    directly. Pydantic's own `Event.articles: List[str]` has no uniqueness
    validator, so this is a legal domain object as far as the model is
    concerned -- the gap is purely at the persistence write boundary."""
    primary = research_item_ids[0]
    return Event(
        id=event_id, canonical_title="Test event", event_type=None, primary_company="Example Co",
        related_companies=[], event_date=None, description=None,
        research_items=research_item_ids, articles=article_ids,
        primary_source=SourceRef(
            research_item_id=primary, article_id=article_ids[0], url="https://example.com/a",
            publisher="Example Co", source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
        ),
        secondary_sources=[],
        relevance_score=0.5, confidence_score=0.5,
        scoring_breakdown=EventScoringBreakdown(
            relevance=RelevanceBreakdown(
                ai_relevance=0.5, financial_services_relevance=0.5, strategic_significance=0.5,
                recency=0.5, novelty=0.5, weighted_total=0.5,
            ),
            confidence=ConfidenceBreakdown(
                primary_source_confirmation=0.5, multiple_source_corroboration=0.5,
                publication_date_certainty=0.5, entity_extraction_confidence=0.5,
                event_classification_confidence=0.5, content_completeness=0.5, weighted_total=0.5,
            ),
        ),
        cluster_metadata=ClusterMetadata(member_count=len(research_item_ids), matches=[]),
        run_id="test-run-0001",
    )


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "event_articles_dup.db")
    database.initialize()
    return database


def test_a_duplicate_article_in_one_event_no_longer_raises_and_persists_once(db: Database):
    """Proves the fix: the exact shape that reproduced the original
    production failure (`UNIQUE constraint failed: event_articles.
    event_id, event_articles.article_id` on run `e05f28043756b4f7`) --
    `EventRepository.upsert()` dedupes it instead of raising, and the
    duplicate collapses to exactly one `event_articles` row, never two."""
    conn = db.connect()
    try:
        ArticleRepository().upsert(conn, _make_article("art-shared"))
        _persist_research_items_for_fk(conn, ["ri-a", "ri-b"], "art-shared")
        event = _hand_built_event("evt0000000000001", research_item_ids=["ri-a", "ri-b"], article_ids=["art-shared", "art-shared"])

        EventRepository().upsert(conn, event)  # no longer raises
        conn.commit()

        rows = conn.execute(
            "SELECT article_id, position FROM event_articles WHERE event_id = ?", (event.id,)
        ).fetchall()
    finally:
        conn.close()
    assert [(r["article_id"], r["position"]) for r in rows] == [("art-shared", 0)]  # one row, not two

    # Both DISTINCT research-item members are still fully tracked -- the fix
    # only ever collapses the repeated ARTICLE id, never drops a real member.
    conn = db.connect()
    try:
        member_rows = conn.execute(
            "SELECT research_item_id FROM event_research_items WHERE event_id = ? ORDER BY position",
            (event.id,),
        ).fetchall()
    finally:
        conn.close()
    assert [r["research_item_id"] for r in member_rows] == ["ri-a", "ri-b"]


def test_build_event_draft_already_dedupes_a_shared_article_id_across_two_research_items(config):
    """Confirms the ONE real production caller (`EventService.build_events`,
    via `events.builder.build_event_draft`) already prevents this: two
    DIFFERENT ResearchItems (different agents analyzing the identical
    article) that cluster into one Event never produce a duplicate in
    `event.articles`. This is why the failure could not be reproduced
    through the normal pipeline under the CURRENT codebase -- the historical
    run most likely predates this existing dedup."""
    item_a = make_item(
        "ri-peer-shared-article", company="Example Co", title="Example Co today announced a new AI platform",
        article_id="art-shared-001", agent=AgentName.PEER,
    )
    item_b = make_item(
        "ri-aiproduct-shared-article", company="Example Co", title="Example Co today announced a new AI platform",
        article_id="art-shared-001", agent=AgentName.AI_PRODUCT,
    )

    events = EventService(config).build_events([item_a, item_b])

    assert len(events) == 1  # the two items clustered into one event
    assert events[0].articles == ["art-shared-001"]  # deduped, not ["art-shared-001", "art-shared-001"]
    assert sorted(events[0].research_items) == sorted([item_a.id, item_b.id])  # both members still tracked


def test_two_distinct_events_sharing_one_article_are_both_preserved(db: Database):
    """Negative control: this is NOT a case the fix may ever collapse --
    two DIFFERENT events (different event_id) legitimately referencing the
    same article (e.g. two unrelated real-world developments both citing
    the same background article) must both persist as their own row."""
    conn = db.connect()
    try:
        ArticleRepository().upsert(conn, _make_article("art-shared-2"))
        _persist_research_items_for_fk(conn, ["ri-c", "ri-d"], "art-shared-2")
        event_1 = _hand_built_event("evt0000000000002", research_item_ids=["ri-c"], article_ids=["art-shared-2"])
        event_2 = _hand_built_event("evt0000000000003", research_item_ids=["ri-d"], article_ids=["art-shared-2"])

        EventRepository().upsert(conn, event_1)
        EventRepository().upsert(conn, event_2)
        conn.commit()

        rows = conn.execute(
            "SELECT event_id, article_id FROM event_articles WHERE article_id = ? ORDER BY event_id", ("art-shared-2",)
        ).fetchall()
    finally:
        conn.close()
    assert [(r["event_id"], r["article_id"]) for r in rows] == [
        ("evt0000000000002", "art-shared-2"), ("evt0000000000003", "art-shared-2"),
    ]


def test_upserting_the_same_event_twice_is_idempotent(db: Database):
    conn = db.connect()
    try:
        ArticleRepository().upsert(conn, _make_article("art-idem"))
        _persist_research_items_for_fk(conn, ["ri-e"], "art-idem")
        event = _hand_built_event("evt0000000000004", research_item_ids=["ri-e"], article_ids=["art-idem"])

        EventRepository().upsert(conn, event)
        EventRepository().upsert(conn, event)  # same event, called again
        conn.commit()

        event_rows = conn.execute("SELECT COUNT(*) FROM events WHERE id = ?", (event.id,)).fetchone()[0]
        junction_rows = conn.execute(
            "SELECT COUNT(*) FROM event_articles WHERE event_id = ?", (event.id,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert event_rows == 1
    assert junction_rows == 1  # not 2 -- re-upserting never duplicates the junction row either


def test_an_unrelated_persistence_failure_still_rolls_back_atomically(db: Database):
    """A genuine, different integrity problem (a foreign-key violation --
    `research_item_id` referencing a row that was never persisted) must
    still raise, and nothing from the same transaction must survive the
    rollback. The fix below must narrow ONLY the specific duplicate-article
    case -- it must never make an invalid association silently succeed."""
    conn = db.connect()
    try:
        ArticleRepository().upsert(conn, _make_article("art-fk-test"))
        event = _hand_built_event(
            "evt0000000000005", research_item_ids=["ri-does-not-exist"], article_ids=["art-fk-test"],
        )

        with pytest.raises(sqlite3.IntegrityError) as excinfo:
            EventRepository().upsert(conn, event)
        assert "FOREIGN KEY" in str(excinfo.value)
        conn.rollback()

        remaining = conn.execute("SELECT COUNT(*) FROM events WHERE id = ?", (event.id,)).fetchone()[0]
    finally:
        conn.close()
    assert remaining == 0  # the events row itself never survived the rollback either
