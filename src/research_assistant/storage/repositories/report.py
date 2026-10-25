"""Report persistence — one row per event_id, holding either a successful
Report or a ReportFailure (via `status`), so `get_event_with_report` is a
single lookup regardless of which outcome a given run produced."""

from __future__ import annotations

import sqlite3
from typing import List, Optional, Union

from research_assistant.models.report import Report, ReportDetails, ReportFailure, ReportSource
from research_assistant.storage.serde import json_dump, json_load

_STATUS_SUCCESS = "success"
_STATUS_FAILURE = "failure"


class ReportRepository:
    def upsert_report(self, conn: sqlite3.Connection, report: Report) -> None:
        conn.execute(
            """
            INSERT INTO reports (
                event_id, status, title, opening_context, details_json, strategic_implications,
                nomura_opportunity, sources_json, relevance_score, confidence_score,
                failure_reason, failure_detail
            ) VALUES (?,?,?,?,?,?,?,?,?,?,NULL,NULL)
            ON CONFLICT(event_id) DO UPDATE SET
                status=excluded.status, title=excluded.title, opening_context=excluded.opening_context,
                details_json=excluded.details_json, strategic_implications=excluded.strategic_implications,
                nomura_opportunity=excluded.nomura_opportunity, sources_json=excluded.sources_json,
                relevance_score=excluded.relevance_score, confidence_score=excluded.confidence_score,
                failure_reason=NULL, failure_detail=NULL
            """,
            (
                report.event_id, _STATUS_SUCCESS, report.title, report.opening_context,
                json_dump(report.details.model_dump(mode="json")), report.strategic_implications,
                report.nomura_opportunity,
                json_dump([source.model_dump(mode="json") for source in report.sources]),
                report.relevance_score, report.confidence_score,
            ),
        )

    def upsert_failure(self, conn: sqlite3.Connection, failure: ReportFailure) -> None:
        conn.execute(
            """
            INSERT INTO reports (event_id, status, failure_reason, failure_detail)
            VALUES (?,?,?,?)
            ON CONFLICT(event_id) DO UPDATE SET
                status=excluded.status, failure_reason=excluded.failure_reason,
                failure_detail=excluded.failure_detail,
                title=NULL, opening_context=NULL, details_json=NULL, strategic_implications=NULL,
                nomura_opportunity=NULL, sources_json=NULL, relevance_score=NULL, confidence_score=NULL
            """,
            (failure.event_id, _STATUS_FAILURE, failure.reason, failure.detail),
        )

    def get(self, conn: sqlite3.Connection, event_id: str) -> Optional[Union[Report, ReportFailure]]:
        row = conn.execute("SELECT * FROM reports WHERE event_id = ?", (event_id,)).fetchone()
        return _row_to_report_or_failure(row) if row else None

    def get_recent(self, conn: sqlite3.Connection, *, limit: int = 50) -> List[Union[Report, ReportFailure]]:
        rows = conn.execute(
            "SELECT reports.* FROM reports "
            "JOIN events ON events.id = reports.event_id "
            "ORDER BY events.event_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_report_or_failure(row) for row in rows]


def _row_to_report_or_failure(row: sqlite3.Row) -> Union[Report, ReportFailure]:
    if row["status"] == _STATUS_FAILURE:
        return ReportFailure(event_id=row["event_id"], reason=row["failure_reason"], detail=row["failure_detail"])

    details_data = json_load(row["details_json"], {})
    sources_data = json_load(row["sources_json"], [])
    return Report(
        event_id=row["event_id"], title=row["title"], opening_context=row["opening_context"],
        details=ReportDetails.model_validate(details_data), strategic_implications=row["strategic_implications"],
        nomura_opportunity=row["nomura_opportunity"],
        sources=[ReportSource.model_validate(source) for source in sources_data],
        relevance_score=row["relevance_score"], confidence_score=row["confidence_score"],
    )
