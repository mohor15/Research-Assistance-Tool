"""ResearchItemRepository: persistence and the article relationship."""

from __future__ import annotations

from datetime import timedelta

from research_assistant.models.enums import EntityType, EventType
from research_assistant.storage.database import Database
from research_assistant.storage.repositories.article import ArticleRepository
from research_assistant.storage.repositories.research_item import ResearchItemRepository

from tests.storage.conftest import D, make_article, make_research_item


def test_persist_and_retrieve_research_item(database: Database):
    conn = database.connect()
    ArticleRepository().upsert(conn, make_article("a1"))
    ResearchItemRepository().upsert(conn, make_research_item("ri1", article_id="a1"))
    conn.commit()

    fetched = ResearchItemRepository().get(conn, "ri1")
    conn.close()

    assert fetched is not None
    assert fetched.article_id == "a1"
    assert fetched.event_type == EventType.PRODUCT_LAUNCH
    assert fetched.entities[0].name == "Entity A"
    assert fetched.entities[0].type == EntityType.COMPANY
    assert fetched.people == ["Jane Doe"]
    assert fetched.deal_value == "$10 million"
    assert fetched.financial_services_relevance.level.value == "high"
    assert fetched.relevance.label.value == "relevant"
    assert fetched.relevance_override is None  # the ordinary, non-override path


def test_relevance_override_round_trips(database: Database):
    """Deterministic recall safeguard audit trail: the disagreement
    diagnostic persisted on `ResearchItem.relevance_override` must survive
    a real upsert/fetch cycle, never silently dropped."""
    override_text = (
        "qwen_relevance=irrelevant deterministic_scope_signal=in_scope final_relevance=relevant "
        "override_reason=matched_categories=['ai_core', 'governance_security'] dated_development=true"
    )
    conn = database.connect()
    ArticleRepository().upsert(conn, make_article("a2"))
    ResearchItemRepository().upsert(
        conn, make_research_item("ri2", article_id="a2", relevance_override=override_text)
    )
    conn.commit()

    fetched = ResearchItemRepository().get(conn, "ri2")
    conn.close()

    assert fetched is not None
    assert fetched.relevance_override == override_text


def test_upsert_does_not_duplicate(database: Database):
    conn = database.connect()
    ArticleRepository().upsert(conn, make_article("a1"))
    ResearchItemRepository().upsert(conn, make_research_item("ri1", article_id="a1"))
    ResearchItemRepository().upsert(conn, make_research_item("ri1", article_id="a1"))
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM research_items").fetchone()[0]
    conn.close()
    assert count == 1


def test_get_by_article_id_returns_the_article_relationship(database: Database):
    conn = database.connect()
    ArticleRepository().upsert(conn, make_article("a1"))
    ResearchItemRepository().upsert(conn, make_research_item("ri1", article_id="a1"))
    conn.commit()

    related = ResearchItemRepository().get_by_article_id(conn, "a1")
    conn.close()

    assert len(related) == 1
    assert related[0].id == "ri1"


def test_get_many_preserves_requested_order_and_skips_missing_ids(database: Database):
    conn = database.connect()
    ArticleRepository().upsert(conn, make_article("a1"))
    ResearchItemRepository().upsert(conn, make_research_item("ri1", article_id="a1"))
    ResearchItemRepository().upsert(conn, make_research_item("ri2", article_id="a1"))
    conn.commit()

    results = ResearchItemRepository().get_many(conn, ["ri2", "does-not-exist", "ri1"])
    conn.close()

    assert [item.id for item in results] == ["ri2", "ri1"]


def test_analyst_findings_are_persisted_when_present(database: Database):
    from research_assistant.models.research_item import AnalystFindings, ReportedFinding

    conn = database.connect()
    ArticleRepository().upsert(conn, make_article("a1"))
    item = make_research_item("ri1", article_id="a1")
    item = item.model_copy(update={
        "analyst_findings": AnalystFindings(
            thesis="A thesis.",
            evidence_points=[ReportedFinding(statement="62% of banks adopted GenAI.", is_quantitative=True)],
        )
    })
    ResearchItemRepository().upsert(conn, item)
    conn.commit()

    fetched = ResearchItemRepository().get(conn, "ri1")
    conn.close()

    assert fetched.analyst_findings is not None
    assert fetched.analyst_findings.thesis == "A thesis."
    assert fetched.analyst_findings.evidence_points[0].is_quantitative is True


def test_analyst_findings_are_none_when_not_present(database: Database):
    conn = database.connect()
    ArticleRepository().upsert(conn, make_article("a1"))
    ResearchItemRepository().upsert(conn, make_research_item("ri1", article_id="a1"))
    conn.commit()

    fetched = ResearchItemRepository().get(conn, "ri1")
    conn.close()

    assert fetched.analyst_findings is None


def test_get_by_company_returns_only_that_entitys_research_items(database: Database):
    conn = database.connect()
    ArticleRepository().upsert(conn, make_article("a1", company="JPMorgan Chase"))
    ArticleRepository().upsert(conn, make_article("a2", company="Bank of America"))
    ResearchItemRepository().upsert(conn, make_research_item("ri1", article_id="a1", company="JPMorgan Chase"))
    ResearchItemRepository().upsert(conn, make_research_item("ri2", article_id="a2", company="Bank of America"))
    conn.commit()

    jpmorgan_items = ResearchItemRepository().get_by_company(conn, "JPMorgan Chase")
    conn.close()

    assert {item.id for item in jpmorgan_items} == {"ri1"}


def test_get_by_company_with_no_items_returns_empty_list(database: Database):
    conn = database.connect()
    result = ResearchItemRepository().get_by_company(conn, "Morgan Stanley")
    conn.close()
    assert result == []


def test_get_by_company_period_bound_excludes_out_of_window_items(database: Database):
    conn = database.connect()
    ArticleRepository().upsert(conn, make_article("a1", company="JPMorgan Chase"))
    ArticleRepository().upsert(conn, make_article("a2", company="JPMorgan Chase"))
    ResearchItemRepository().upsert(conn, make_research_item("in", article_id="a1", company="JPMorgan Chase", publication_date=D))
    ResearchItemRepository().upsert(
        conn, make_research_item("old", article_id="a2", company="JPMorgan Chase", publication_date=D - timedelta(days=30))
    )
    conn.commit()

    result = ResearchItemRepository().get_by_company(
        conn, "JPMorgan Chase", since=D - timedelta(hours=1), until=D + timedelta(hours=1)
    )
    conn.close()

    assert {item.id for item in result} == {"in"}
