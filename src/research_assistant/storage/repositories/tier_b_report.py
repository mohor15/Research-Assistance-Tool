"""Tier B newsletter report persistence/caching -- one row per article_id,
holding either a successful `TierBReport` or a `TierBReportFailure` (via
`status`), exactly mirroring `ReportRepository`'s own shape for Tier A.
Never keyed by/associated with an Event -- a Tier B article has none by
definition; see `storage.database`'s `tier_b_reports` table comment for
why this is a separate table rather than a reuse of `reports`.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import NamedTuple, Optional, Union

from research_assistant.models.report import ReportSource, TierBReport, TierBReportFailure
from research_assistant.storage.serde import dt_from_str, dt_to_str, json_dump, json_load

_STATUS_SUCCESS = "success"
_STATUS_FAILURE = "failure"


class CachedTierBReport(NamedTuple):
    """What `TierBReportRepository.get` returns -- the stored
    report/failure PLUS the cache-identity metadata it was stored with, so
    the resolver (`reporting.tier_b_reports`) can decide validity (does
    the current article content_hash / schema_version still match?)
    without a second query."""

    result: Union[TierBReport, TierBReportFailure]
    content_hash: Optional[str]
    schema_version: str
    generated_at: datetime


class TierBReportRepository:
    def upsert_report(
        self, conn: sqlite3.Connection, report: TierBReport, *, content_hash: Optional[str],
        schema_version: str, generated_at: datetime,
    ) -> None:
        conn.execute(
            """
            INSERT INTO tier_b_reports (
                article_id, status, content_hash, schema_version, title, opening_context,
                key_insights_json, strategic_implications, nomura_opportunity, content_partial,
                source_json, failure_reason, failure_detail, generated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,?)
            ON CONFLICT(article_id) DO UPDATE SET
                status=excluded.status, content_hash=excluded.content_hash, schema_version=excluded.schema_version,
                title=excluded.title, opening_context=excluded.opening_context,
                key_insights_json=excluded.key_insights_json, strategic_implications=excluded.strategic_implications,
                nomura_opportunity=excluded.nomura_opportunity, content_partial=excluded.content_partial,
                source_json=excluded.source_json, failure_reason=NULL, failure_detail=NULL,
                generated_at=excluded.generated_at
            """,
            (
                report.article_id, _STATUS_SUCCESS, content_hash, schema_version, report.title,
                report.opening_context, json_dump(report.key_insights), report.strategic_implications,
                report.nomura_opportunity, int(report.content_partial),
                json_dump(report.source.model_dump(mode="json")), dt_to_str(generated_at),
            ),
        )

    def upsert_failure(
        self, conn: sqlite3.Connection, failure: TierBReportFailure, *, content_hash: Optional[str],
        schema_version: str, generated_at: datetime,
    ) -> None:
        conn.execute(
            """
            INSERT INTO tier_b_reports (
                article_id, status, content_hash, schema_version, failure_reason, failure_detail, generated_at
            ) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(article_id) DO UPDATE SET
                status=excluded.status, content_hash=excluded.content_hash, schema_version=excluded.schema_version,
                title=NULL, opening_context=NULL, key_insights_json=NULL, strategic_implications=NULL,
                nomura_opportunity=NULL, content_partial=0, source_json=NULL,
                failure_reason=excluded.failure_reason, failure_detail=excluded.failure_detail,
                generated_at=excluded.generated_at
            """,
            (failure.article_id, _STATUS_FAILURE, content_hash, schema_version, failure.reason, failure.detail,
             dt_to_str(generated_at)),
        )

    def get(self, conn: sqlite3.Connection, article_id: str) -> Optional[CachedTierBReport]:
        row = conn.execute("SELECT * FROM tier_b_reports WHERE article_id = ?", (article_id,)).fetchone()
        if row is None:
            return None
        return CachedTierBReport(
            result=_row_to_report_or_failure(row),
            content_hash=row["content_hash"],
            schema_version=row["schema_version"],
            generated_at=dt_from_str(row["generated_at"]),
        )


def _row_to_report_or_failure(row: sqlite3.Row) -> Union[TierBReport, TierBReportFailure]:
    if row["status"] == _STATUS_FAILURE:
        return TierBReportFailure(article_id=row["article_id"], reason=row["failure_reason"], detail=row["failure_detail"])

    source_data = json_load(row["source_json"], {})
    return TierBReport(
        article_id=row["article_id"], title=row["title"], opening_context=row["opening_context"],
        key_insights=json_load(row["key_insights_json"], []), strategic_implications=row["strategic_implications"],
        nomura_opportunity=row["nomura_opportunity"], source=ReportSource.model_validate(source_data),
        content_partial=bool(row["content_partial"]),
    )
