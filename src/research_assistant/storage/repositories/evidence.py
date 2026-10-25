"""Evidence persistence — always tied to its Article via `article_id`."""

from __future__ import annotations

import sqlite3
from typing import List, Optional

from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.models.evidence import Evidence
from research_assistant.retrieval.models import ExtractionMethod
from research_assistant.storage.serde import dt_from_str, dt_to_str, json_dump, json_load


class EvidenceRepository:
    def upsert(self, conn: sqlite3.Connection, evidence: Evidence) -> None:
        conn.execute(
            """
            INSERT INTO evidence (
                evidence_id, article_id, source_url, canonical_url, publisher, title, publication_date,
                retrieved_at, content, content_hash, source_type, source_priority, extraction_method,
                evidence_span, metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(evidence_id) DO UPDATE SET
                article_id=excluded.article_id, source_url=excluded.source_url, canonical_url=excluded.canonical_url,
                publisher=excluded.publisher, title=excluded.title, publication_date=excluded.publication_date,
                retrieved_at=excluded.retrieved_at, content=excluded.content, content_hash=excluded.content_hash,
                source_type=excluded.source_type, source_priority=excluded.source_priority,
                extraction_method=excluded.extraction_method, evidence_span=excluded.evidence_span,
                metadata_json=excluded.metadata_json
            """,
            (
                evidence.evidence_id, evidence.article_id, evidence.source_url, evidence.canonical_url,
                evidence.publisher, evidence.title, dt_to_str(evidence.publication_date),
                dt_to_str(evidence.retrieved_at), evidence.content, evidence.content_hash,
                evidence.source_type.value, evidence.source_priority.value, evidence.extraction_method.value,
                evidence.evidence_span, json_dump(evidence.metadata),
            ),
        )

    def get(self, conn: sqlite3.Connection, evidence_id: str) -> Optional[Evidence]:
        row = conn.execute("SELECT * FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone()
        return _row_to_evidence(row) if row else None

    def get_by_article_id(self, conn: sqlite3.Connection, article_id: str) -> List[Evidence]:
        rows = conn.execute("SELECT * FROM evidence WHERE article_id = ?", (article_id,)).fetchall()
        return [_row_to_evidence(row) for row in rows]


def _row_to_evidence(row: sqlite3.Row) -> Evidence:
    return Evidence(
        evidence_id=row["evidence_id"], article_id=row["article_id"], source_url=row["source_url"],
        canonical_url=row["canonical_url"], publisher=row["publisher"], title=row["title"],
        publication_date=dt_from_str(row["publication_date"]), retrieved_at=dt_from_str(row["retrieved_at"]),
        content=row["content"], content_hash=row["content_hash"], source_type=SourceType(row["source_type"]),
        source_priority=SourcePriority(row["source_priority"]), extraction_method=ExtractionMethod(row["extraction_method"]),
        evidence_span=row["evidence_span"], metadata=json_load(row["metadata_json"], {}),
    )
