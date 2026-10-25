"""Event persistence, including the event<->research_item and event<->article
junction tables (normalized, since these are plain id references — see
`storage.database`'s module docstring for why).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import List, Optional

from research_assistant.events.models import (
    ClusterMetadata,
    ConfidenceBreakdown,
    Event,
    EventScoringBreakdown,
    RelevanceBreakdown,
    SourceRef,
)
from research_assistant.models.enums import EventType, GroundingType
from research_assistant.storage.serde import dt_from_str, dt_to_str, json_dump, json_load


class EventRepository:
    def upsert(self, conn: sqlite3.Connection, event: Event) -> None:
        conn.execute(
            """
            INSERT INTO events (
                id, canonical_title, event_type, primary_company, related_companies_json, event_date,
                description, primary_source_json, secondary_sources_json, relevance_score, confidence_score,
                scoring_breakdown_json, cluster_metadata_json, grounding_type, run_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                canonical_title=excluded.canonical_title, event_type=excluded.event_type,
                primary_company=excluded.primary_company, related_companies_json=excluded.related_companies_json,
                event_date=excluded.event_date, description=excluded.description,
                primary_source_json=excluded.primary_source_json,
                secondary_sources_json=excluded.secondary_sources_json, relevance_score=excluded.relevance_score,
                confidence_score=excluded.confidence_score, scoring_breakdown_json=excluded.scoring_breakdown_json,
                cluster_metadata_json=excluded.cluster_metadata_json, grounding_type=excluded.grounding_type,
                run_id=excluded.run_id
            """,
            (
                event.id, event.canonical_title, event.event_type.value if event.event_type else None,
                event.primary_company, json_dump(event.related_companies), dt_to_str(event.event_date),
                event.description, json_dump(event.primary_source.model_dump(mode="json")),
                json_dump([ref.model_dump(mode="json") for ref in event.secondary_sources]),
                event.relevance_score, event.confidence_score,
                json_dump(event.scoring_breakdown.model_dump(mode="json")),
                json_dump(event.cluster_metadata.model_dump(mode="json")),
                event.grounding_type.value, event.run_id,
            ),
        )

        # Refresh junction tables idempotently: delete-then-reinsert avoids
        # duplicate rows or stale members if the same event id is persisted
        # again with a (in practice unlikely, since ids are content-derived)
        # different member set.
        #
        # Defense in depth (confirmed production failure, run
        # `e05f28043756b4f7`: "UNIQUE constraint failed: event_articles.
        # event_id, event_articles.article_id"): both junction tables have a
        # PRIMARY KEY on (event_id, member_id), and `events.builder.
        # build_event_draft` -- the one real caller that builds a fresh
        # `Event` -- already deduplicates `articles` for exactly this reason
        # ("the same article can be routed to more than one agent, so two
        # distinct ResearchItems in this cluster may share an article_id").
        # Nothing in `Event`'s own Pydantic model enforces that, though, so
        # this repository -- the actual write boundary, and the only place
        # that can authoritatively guarantee what SQLite will accept --
        # must not simply trust that every caller already deduplicated.
        # Order-preserving (`dict.fromkeys`), so position still reflects
        # each member's first occurrence; a genuine duplicate collapsing to
        # one row is not a lost association (it was one member referenced
        # twice, not two different members), and two DIFFERENT events still
        # each get their own row for a shared article/research item --
        # this only ever removes an exact repeated id within ONE event's
        # own list, never across events, and never a different article.
        deduped_research_items = list(dict.fromkeys(event.research_items))
        deduped_articles = list(dict.fromkeys(event.articles))

        conn.execute("DELETE FROM event_research_items WHERE event_id = ?", (event.id,))
        conn.executemany(
            "INSERT INTO event_research_items (event_id, research_item_id, position) VALUES (?,?,?)",
            [(event.id, research_item_id, position) for position, research_item_id in enumerate(deduped_research_items)],
        )
        conn.execute("DELETE FROM event_articles WHERE event_id = ?", (event.id,))
        conn.executemany(
            "INSERT INTO event_articles (event_id, article_id, position) VALUES (?,?,?)",
            [(event.id, article_id, position) for position, article_id in enumerate(deduped_articles)],
        )

    def get(self, conn: sqlite3.Connection, event_id: str) -> Optional[Event]:
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            return None
        return self._hydrate(conn, row)

    def get_recent(self, conn: sqlite3.Connection, *, limit: int = 50, since: Optional[datetime] = None) -> List[Event]:
        if since is not None:
            rows = conn.execute(
                "SELECT * FROM events WHERE event_date >= ? ORDER BY event_date DESC LIMIT ?",
                (dt_to_str(since), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY event_date DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._hydrate(conn, row) for row in rows]

    def get_by_company(self, conn: sqlite3.Connection, company: str, *, limit: int = 50) -> List[Event]:
        rows = conn.execute(
            "SELECT * FROM events WHERE primary_company = ? OR related_companies_json LIKE ? "
            "ORDER BY event_date DESC LIMIT ?",
            (company, f'%"{company}"%', limit),
        ).fetchall()
        return [self._hydrate(conn, row) for row in rows]

    def get_by_event_type(self, conn: sqlite3.Connection, event_type: EventType, *, limit: int = 50) -> List[Event]:
        rows = conn.execute(
            "SELECT * FROM events WHERE event_type = ? ORDER BY event_date DESC LIMIT ?",
            (event_type.value, limit),
        ).fetchall()
        return [self._hydrate(conn, row) for row in rows]

    def get_by_agent(self, conn: sqlite3.Connection, agent: str, *, limit: int = 50) -> List[Event]:
        rows = conn.execute(
            """
            SELECT DISTINCT events.* FROM events
            JOIN event_research_items ON event_research_items.event_id = events.id
            JOIN research_items ON research_items.id = event_research_items.research_item_id
            WHERE research_items.agent = ?
            ORDER BY events.event_date DESC LIMIT ?
            """,
            (agent, limit),
        ).fetchall()
        return [self._hydrate(conn, row) for row in rows]

    def get_by_run_id(self, conn: sqlite3.Connection, run_id: str) -> List[Event]:
        """Explicit run ownership (Phase 17, Section 4) -- every Event this
        specific run actually formed, never reconstructed from a mutable
        Article timestamp."""
        rows = conn.execute("SELECT * FROM events WHERE run_id = ?", (run_id,)).fetchall()
        return [self._hydrate(conn, row) for row in rows]

    def get_by_min_relevance(self, conn: sqlite3.Connection, min_relevance: float, *, limit: int = 50) -> List[Event]:
        rows = conn.execute(
            "SELECT * FROM events WHERE relevance_score >= ? ORDER BY relevance_score DESC LIMIT ?",
            (min_relevance, limit),
        ).fetchall()
        return [self._hydrate(conn, row) for row in rows]

    def get_by_min_confidence(self, conn: sqlite3.Connection, min_confidence: float, *, limit: int = 50) -> List[Event]:
        rows = conn.execute(
            "SELECT * FROM events WHERE confidence_score >= ? ORDER BY confidence_score DESC LIMIT ?",
            (min_confidence, limit),
        ).fetchall()
        return [self._hydrate(conn, row) for row in rows]

    def search(
        self,
        conn: sqlite3.Connection,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        company: Optional[str] = None,
        event_type: Optional[EventType] = None,
        agent: Optional[str] = None,
        min_relevance: Optional[float] = None,
        min_confidence: Optional[float] = None,
        limit: int = 50,
    ) -> List[Event]:
        """Combined dashboard filter — every argument is optional and AND-ed
        together. This is the one place the Streamlit filter panel (Phase 7
        Part 7) should query through, rather than composing raw SQL itself."""
        clauses: List[str] = []
        params: List[object] = []

        if agent is not None:
            base = (
                "SELECT DISTINCT events.* FROM events "
                "JOIN event_research_items ON event_research_items.event_id = events.id "
                "JOIN research_items ON research_items.id = event_research_items.research_item_id"
            )
            clauses.append("research_items.agent = ?")
            params.append(agent)
        else:
            base = "SELECT * FROM events"

        if since is not None:
            clauses.append("events.event_date >= ?" if agent else "event_date >= ?")
            params.append(dt_to_str(since))
        if until is not None:
            clauses.append("events.event_date <= ?" if agent else "event_date <= ?")
            params.append(dt_to_str(until))
        if company is not None:
            clauses.append(
                "(events.primary_company = ? OR events.related_companies_json LIKE ?)"
                if agent
                else "(primary_company = ? OR related_companies_json LIKE ?)"
            )
            params.extend([company, f'%"{company}"%'])
        if event_type is not None:
            clauses.append("events.event_type = ?" if agent else "event_type = ?")
            params.append(event_type.value)
        if min_relevance is not None:
            clauses.append("events.relevance_score >= ?" if agent else "relevance_score >= ?")
            params.append(min_relevance)
        if min_confidence is not None:
            clauses.append("events.confidence_score >= ?" if agent else "confidence_score >= ?")
            params.append(min_confidence)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        order_by = " ORDER BY events.relevance_score DESC" if agent else " ORDER BY relevance_score DESC"
        sql = f"{base}{where}{order_by} LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return [self._hydrate(conn, row) for row in rows]

    def count_by_company(self, conn: sqlite3.Connection, *, limit: int = 10) -> List[sqlite3.Row]:
        return conn.execute(
            "SELECT primary_company AS company, COUNT(*) AS event_count FROM events "
            "GROUP BY primary_company ORDER BY event_count DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def count_by_event_type(self, conn: sqlite3.Connection) -> List[sqlite3.Row]:
        return conn.execute(
            "SELECT event_type, COUNT(*) AS event_count FROM events "
            "WHERE event_type IS NOT NULL GROUP BY event_type ORDER BY event_count DESC"
        ).fetchall()
        return [self._hydrate(conn, row) for row in rows]

    def get_research_item_ids(self, conn: sqlite3.Connection, event_id: str) -> List[str]:
        rows = conn.execute(
            "SELECT research_item_id FROM event_research_items WHERE event_id = ? ORDER BY position", (event_id,)
        ).fetchall()
        return [row["research_item_id"] for row in rows]

    def get_article_ids(self, conn: sqlite3.Connection, event_id: str) -> List[str]:
        rows = conn.execute(
            "SELECT article_id FROM event_articles WHERE event_id = ? ORDER BY position", (event_id,)
        ).fetchall()
        return [row["article_id"] for row in rows]

    def _hydrate(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"], canonical_title=row["canonical_title"],
            event_type=EventType(row["event_type"]) if row["event_type"] else None,
            primary_company=row["primary_company"], related_companies=json_load(row["related_companies_json"], []),
            event_date=dt_from_str(row["event_date"]), description=row["description"],
            research_items=self.get_research_item_ids(conn, row["id"]),
            articles=self.get_article_ids(conn, row["id"]),
            primary_source=SourceRef.model_validate(json_load(row["primary_source_json"], {})),
            secondary_sources=[SourceRef.model_validate(ref) for ref in json_load(row["secondary_sources_json"], [])],
            relevance_score=row["relevance_score"], confidence_score=row["confidence_score"],
            scoring_breakdown=EventScoringBreakdown.model_validate(json_load(row["scoring_breakdown_json"], {})),
            cluster_metadata=ClusterMetadata.model_validate(json_load(row["cluster_metadata_json"], {})),
            grounding_type=(
                GroundingType(row["grounding_type"])
                if "grounding_type" in row.keys() and row["grounding_type"]
                else GroundingType.CONTENT
            ),
            run_id=row["run_id"] if "run_id" in row.keys() else None,
        )
