"""ResearchItem persistence — always tied to its Article via `article_id`."""

from __future__ import annotations

import sqlite3
from typing import List, Optional

from research_assistant.models.enums import AgentName, EventType, GroundingType, SourcePriority, SourceType
from research_assistant.models.research_item import (
    AnalystFindings,
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    ResearchItem,
    SourceQuality,
    TopicCategory,
    TypedEntity,
)
from research_assistant.storage.serde import dt_from_str, dt_to_str, json_dump, json_load


class ResearchItemRepository:
    def upsert(self, conn: sqlite3.Connection, item: ResearchItem) -> None:
        conn.execute(
            """
            INSERT INTO research_items (
                id, agent, article_id, publisher, title, url, publication_date, company, category,
                event_type, summary, is_dated_development, development_date_rationale,
                entities_json, people_json, products_json, deal_value,
                partnerships_json, technologies_json, geography_json,
                relevance_label, relevance_reason,
                financial_services_relevance_level, financial_services_relevance_rationale,
                nomura_relevance_level, nomura_relevance_rationale,
                strategic_significance_level, strategic_significance_rationale,
                evidence_ids_json, source_type, source_priority, analyst_findings_json, grounding_type, run_id,
                relevance_override
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                agent=excluded.agent, article_id=excluded.article_id, publisher=excluded.publisher,
                title=excluded.title, url=excluded.url, publication_date=excluded.publication_date,
                company=excluded.company, category=excluded.category, event_type=excluded.event_type,
                summary=excluded.summary, is_dated_development=excluded.is_dated_development,
                development_date_rationale=excluded.development_date_rationale,
                entities_json=excluded.entities_json, people_json=excluded.people_json,
                products_json=excluded.products_json, deal_value=excluded.deal_value,
                partnerships_json=excluded.partnerships_json, technologies_json=excluded.technologies_json,
                geography_json=excluded.geography_json, relevance_label=excluded.relevance_label,
                relevance_reason=excluded.relevance_reason,
                financial_services_relevance_level=excluded.financial_services_relevance_level,
                financial_services_relevance_rationale=excluded.financial_services_relevance_rationale,
                nomura_relevance_level=excluded.nomura_relevance_level,
                nomura_relevance_rationale=excluded.nomura_relevance_rationale,
                strategic_significance_level=excluded.strategic_significance_level,
                strategic_significance_rationale=excluded.strategic_significance_rationale,
                evidence_ids_json=excluded.evidence_ids_json, source_type=excluded.source_type,
                source_priority=excluded.source_priority, analyst_findings_json=excluded.analyst_findings_json,
                grounding_type=excluded.grounding_type, run_id=excluded.run_id,
                relevance_override=excluded.relevance_override
            """,
            (
                item.id, item.agent.value, item.article_id, item.publisher, item.title, item.url,
                dt_to_str(item.publication_date), item.company, item.category.value,
                item.event_type.value if item.event_type else None, item.summary,
                int(item.is_dated_development), item.development_date_rationale,
                json_dump([e.model_dump(mode="json") for e in item.entities]),
                json_dump(item.people), json_dump(item.products), item.deal_value,
                json_dump(item.partnerships), json_dump(item.technologies), json_dump(item.geography),
                item.relevance.label.value, item.relevance.reason,
                item.financial_services_relevance.level.value, item.financial_services_relevance.rationale,
                item.nomura_relevance.level.value, item.nomura_relevance.rationale,
                item.strategic_significance.level.value, item.strategic_significance.rationale,
                json_dump(item.evidence), item.source_quality.source_type.value,
                item.source_quality.source_priority.value,
                json_dump(item.analyst_findings.model_dump(mode="json")) if item.analyst_findings else None,
                item.grounding_type.value, item.run_id, item.relevance_override,
            ),
        )

    def get(self, conn: sqlite3.Connection, item_id: str) -> Optional[ResearchItem]:
        row = conn.execute("SELECT * FROM research_items WHERE id = ?", (item_id,)).fetchone()
        return _row_to_research_item(row) if row else None

    def get_by_article_id(self, conn: sqlite3.Connection, article_id: str) -> List[ResearchItem]:
        rows = conn.execute("SELECT * FROM research_items WHERE article_id = ?", (article_id,)).fetchall()
        return [_row_to_research_item(row) for row in rows]

    def get_by_run_id(self, conn: sqlite3.Connection, run_id: str) -> List[ResearchItem]:
        """Explicit run ownership (Phase 17, Section 4) -- every ResearchItem
        this specific run actually produced, never reconstructed from a
        mutable Article timestamp."""
        rows = conn.execute("SELECT * FROM research_items WHERE run_id = ?", (run_id,)).fetchall()
        return [_row_to_research_item(row) for row in rows]

    def get_many(self, conn: sqlite3.Connection, item_ids: List[str]) -> List[ResearchItem]:
        if not item_ids:
            return []
        placeholders = ",".join("?" * len(item_ids))
        rows = conn.execute(f"SELECT * FROM research_items WHERE id IN ({placeholders})", item_ids).fetchall()
        by_id = {row["id"]: _row_to_research_item(row) for row in rows}
        return [by_id[item_id] for item_id in item_ids if item_id in by_id]

    def get_by_company(
        self,
        conn: sqlite3.Connection,
        company: str,
        *,
        since=None,
        until=None,
        limit: int = 200,
    ) -> List[ResearchItem]:
        """ResearchItems for one configured entity (Phase 1's entity-centric
        monitoring pages). Every persisted ResearchItem already passed its
        agent's relevance judgment (IRRELEVANT ones are never persisted —
        see `ResearchAgent.analyze`), so this is exactly "articles that
        passed relevance filtering" for that entity. A NULL
        `publication_date` is never excluded by a period bound, mirroring
        the pipeline's own "missing date is retained" rule."""
        query = "SELECT * FROM research_items WHERE company = ?"
        params: list = [company]
        if since is not None:
            query += " AND (publication_date IS NULL OR publication_date >= ?)"
            params.append(dt_to_str(since))
        if until is not None:
            query += " AND (publication_date IS NULL OR publication_date <= ?)"
            params.append(dt_to_str(until))
        query += " ORDER BY publication_date DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [_row_to_research_item(row) for row in rows]


def _row_to_research_item(row: sqlite3.Row) -> ResearchItem:
    analyst_findings = None
    if row["analyst_findings_json"]:
        analyst_findings = AnalystFindings.model_validate(json_load(row["analyst_findings_json"], {}))

    return ResearchItem(
        id=row["id"], agent=AgentName(row["agent"]), article_id=row["article_id"], publisher=row["publisher"],
        title=row["title"], url=row["url"], publication_date=dt_from_str(row["publication_date"]),
        company=row["company"], category=TopicCategory(row["category"]),
        event_type=EventType(row["event_type"]) if row["event_type"] else None, summary=row["summary"],
        is_dated_development=bool(row["is_dated_development"]),
        development_date_rationale=row["development_date_rationale"],
        entities=[TypedEntity.model_validate(e) for e in json_load(row["entities_json"], [])],
        people=json_load(row["people_json"], []),
        products=json_load(row["products_json"], []), deal_value=row["deal_value"],
        partnerships=json_load(row["partnerships_json"], []), technologies=json_load(row["technologies_json"], []),
        geography=json_load(row["geography_json"], []),
        relevance=RelevanceDecision(label=RelevanceLabel(row["relevance_label"]), reason=row["relevance_reason"]),
        financial_services_relevance=RelevanceAssessment(
            level=RelevanceLevel(row["financial_services_relevance_level"]),
            rationale=row["financial_services_relevance_rationale"],
        ),
        nomura_relevance=RelevanceAssessment(
            level=RelevanceLevel(row["nomura_relevance_level"]), rationale=row["nomura_relevance_rationale"]
        ),
        strategic_significance=RelevanceAssessment(
            level=RelevanceLevel(row["strategic_significance_level"]),
            rationale=row["strategic_significance_rationale"],
        ),
        evidence=json_load(row["evidence_ids_json"], []),
        source_quality=SourceQuality(
            source_type=SourceType(row["source_type"]), source_priority=SourcePriority(row["source_priority"])
        ),
        analyst_findings=analyst_findings,
        grounding_type=(
            GroundingType(row["grounding_type"])
            if "grounding_type" in row.keys() and row["grounding_type"]
            else GroundingType.CONTENT
        ),
        run_id=row["run_id"] if "run_id" in row.keys() else None,
        relevance_override=row["relevance_override"] if "relevance_override" in row.keys() else None,
    )
