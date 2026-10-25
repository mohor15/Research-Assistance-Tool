"""Article persistence — upsert-by-id, with lookups by canonical URL and
content hash (both used to detect and skip re-processing the same
underlying content, per Phase 7's idempotency requirement)."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import List, Optional

from research_assistant.models.article import Article, DatePrecision
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.models import ExtractionMethod, ExtractionStatus
from research_assistant.storage.serde import dt_from_str, dt_to_str, json_dump, json_load


class ArticleRepository:
    def upsert(self, conn: sqlite3.Connection, article: Article) -> None:
        conn.execute(
            """
            INSERT INTO articles (
                id, url, canonical_url, domain, source_name, source_type, source_priority,
                publisher, title, author, publication_date, publication_date_precision, retrieved_at,
                raw_text, clean_text, content_hash, extraction_method, extraction_status, metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                url=excluded.url, canonical_url=excluded.canonical_url, domain=excluded.domain,
                source_name=excluded.source_name, source_type=excluded.source_type,
                source_priority=excluded.source_priority, publisher=excluded.publisher, title=excluded.title,
                author=excluded.author, publication_date=excluded.publication_date,
                publication_date_precision=excluded.publication_date_precision, retrieved_at=excluded.retrieved_at,
                raw_text=excluded.raw_text, clean_text=excluded.clean_text, content_hash=excluded.content_hash,
                extraction_method=excluded.extraction_method, extraction_status=excluded.extraction_status,
                metadata_json=excluded.metadata_json
            """,
            (
                article.id, article.url, article.canonical_url, article.domain, article.source_name,
                article.source_type.value, article.source_priority.value, article.publisher, article.title,
                article.author, dt_to_str(article.publication_date), article.publication_date_precision.value,
                dt_to_str(article.retrieved_at), article.raw_text, article.clean_text, article.content_hash,
                article.extraction_method.value, article.extraction_status.value, json_dump(article.metadata),
            ),
        )

    def get(self, conn: sqlite3.Connection, article_id: str) -> Optional[Article]:
        row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        return _row_to_article(row) if row else None

    def get_by_canonical_url(self, conn: sqlite3.Connection, canonical_url: str) -> Optional[Article]:
        row = conn.execute("SELECT * FROM articles WHERE canonical_url = ?", (canonical_url,)).fetchone()
        return _row_to_article(row) if row else None

    def get_by_content_hash(self, conn: sqlite3.Connection, content_hash: str) -> Optional[Article]:
        row = conn.execute("SELECT * FROM articles WHERE content_hash = ?", (content_hash,)).fetchone()
        return _row_to_article(row) if row else None

    def get_by_source_name(
        self,
        conn: sqlite3.Connection,
        source_name: str,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 200,
    ) -> List[Article]:
        """All articles retrieved for one configured entity (Phase 1's
        entity-centric monitoring pages) — optionally bounded to a
        `publication_date` window, exactly like the rest of this project's
        period filtering (never `retrieved_at`)."""
        # A NULL publication_date is never excluded by a period bound here,
        # exactly mirroring pipeline.nodes._filter_by_period's documented
        # "missing date is retained, never silently dropped" rule.
        query = "SELECT * FROM articles WHERE source_name = ?"
        params: list = [source_name]
        if since is not None:
            query += " AND (publication_date IS NULL OR publication_date >= ?)"
            params.append(dt_to_str(since))
        if until is not None:
            query += " AND (publication_date IS NULL OR publication_date <= ?)"
            params.append(dt_to_str(until))
        query += " ORDER BY publication_date DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [_row_to_article(row) for row in rows]

    def get_retrieved_between(
        self, conn: sqlite3.Connection, start: datetime, end: datetime, *, limit: int = 2000,
    ) -> List[Article]:
        """All articles whose `retrieved_at` falls in `[start, end]` --
        used by the CURRENT Scan Explorer (`ui.dashboard_data`) as a
        best-effort reconstruction of "which articles this run retrieved",
        since no `run_id` is persisted on this table (see that module's
        docstring for why). `retrieved_at` is overwritten on every re-fetch
        of the same article, so this is exact for the most recent run that
        touched a given row, but can misattribute a row to an older run if
        a newer run has since re-fetched the identical URL."""
        rows = conn.execute(
            "SELECT * FROM articles WHERE retrieved_at >= ? AND retrieved_at <= ? ORDER BY retrieved_at DESC LIMIT ?",
            (dt_to_str(start), dt_to_str(end), limit),
        ).fetchall()
        return [_row_to_article(row) for row in rows]


def _row_to_article(row: sqlite3.Row) -> Article:
    return Article(
        id=row["id"], url=row["url"], canonical_url=row["canonical_url"], domain=row["domain"],
        source_name=row["source_name"], source_type=SourceType(row["source_type"]),
        source_priority=SourcePriority(row["source_priority"]), publisher=row["publisher"], title=row["title"],
        author=row["author"], publication_date=dt_from_str(row["publication_date"]),
        publication_date_precision=DatePrecision(row["publication_date_precision"]),
        retrieved_at=dt_from_str(row["retrieved_at"]), raw_text=row["raw_text"], clean_text=row["clean_text"],
        content_hash=row["content_hash"], extraction_method=ExtractionMethod(row["extraction_method"]),
        extraction_status=ExtractionStatus(row["extraction_status"]), metadata=json_load(row["metadata_json"], {}),
    )
