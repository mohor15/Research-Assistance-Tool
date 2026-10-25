"""Deterministic LINK-grounded ResearchItem construction (no LLM, no Ollama
call) -- the second lifecycle path for an article whose body could not be
retrieved but whose title/metadata establish a genuine, relevant,
substantive development."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.agents.link_grounded import build_link_grounded_research_item
from research_assistant.config import get_config
from research_assistant.models.article import Article, DatePrecision
from research_assistant.models.enums import AgentName, GroundingType, SourcePriority, SourceType
from research_assistant.retrieval.models import ExtractionMethod, ExtractionStatus

CONFIG = get_config()
D = datetime(2026, 10, 4, tzinfo=timezone.utc)


def _unavailable_article(title, *, publication_date=D, precision=DatePrecision.DATE_ONLY, source_name="Financial Times"):
    return Article(
        id="art-link", url="https://www.ft.com/content/aaaa", canonical_url="https://www.ft.com/content/aaaa",
        domain="ft.com", source_name=source_name, source_type=SourceType.SECONDARY,
        source_priority=SourcePriority.HIGH, publisher=source_name, title=title, author=None,
        publication_date=publication_date, publication_date_precision=precision, retrieved_at=D,
        raw_text=None, clean_text=None, content_hash=None,  # content genuinely unavailable
        extraction_method=ExtractionMethod.NONE, extraction_status=ExtractionStatus.FAILED,
        metadata={"reason": "http_error", "content_unavailable_reason": "HTTP_403"},
    )


# ---------------------------------------------------------------------------
# Test 3 / 4: generic title rejected, substantive title with a clear event
# type accepted
# ---------------------------------------------------------------------------


def test_substantive_title_with_clear_event_type_becomes_link_grounded_item():
    article = _unavailable_article("OpenAI signs major enterprise partnership with Microsoft")
    item = build_link_grounded_research_item(article, config=CONFIG, agent=AgentName.FINANCIAL_NEWS)

    assert item is not None
    assert item.grounding_type == GroundingType.LINK
    assert item.relevance.label.value == "relevant"
    assert item.event_type is not None and item.event_type.value == "partnership"


def test_generic_non_substantive_title_is_rejected():
    article = _unavailable_article("AI is changing the world")
    item = build_link_grounded_research_item(article, config=CONFIG, agent=AgentName.FINANCIAL_NEWS)
    assert item is None


def test_title_with_no_event_type_phrase_is_rejected_even_with_topic_keywords():
    article = _unavailable_article("Thoughts on the future of artificial intelligence in banking")
    item = build_link_grounded_research_item(article, config=CONFIG, agent=AgentName.FINANCIAL_NEWS)
    assert item is None  # no concrete event-type phrase ("partnership", "funding", etc.)


def test_short_title_is_rejected_even_with_an_event_type_word():
    article = _unavailable_article("AI funding")  # 2 words, below the generic-title threshold
    item = build_link_grounded_research_item(article, config=CONFIG, agent=AgentName.FINANCIAL_NEWS)
    assert item is None


def test_missing_title_is_rejected():
    article = _unavailable_article(None)
    item = build_link_grounded_research_item(article, config=CONFIG, agent=AgentName.FINANCIAL_NEWS)
    assert item is None


# ---------------------------------------------------------------------------
# Test 5: never a fabricated article-content field
# ---------------------------------------------------------------------------


def test_link_grounded_item_never_carries_fabricated_facts():
    article = _unavailable_article("Anthropic announces major funding round led by new investors")
    item = build_link_grounded_research_item(article, config=CONFIG, agent=AgentName.AI_PRODUCT)

    assert item is not None
    assert item.entities == []
    assert item.people == []
    assert item.products == []
    assert item.technologies == []
    assert item.partnerships == []
    assert item.deal_value is None
    assert item.evidence == []  # no Evidence backs this item -- by construction
    # the summary is a fixed, transparent template -- never claims to have read the article
    assert "automated access to the article content was unavailable" in item.summary


# ---------------------------------------------------------------------------
# Test 6: publication-date precision preserved, never upgraded/invented
# ---------------------------------------------------------------------------


def test_exact_precision_date_is_preserved():
    article = _unavailable_article(
        "JPMorgan announces new AI partnership with OpenAI", precision=DatePrecision.EXACT,
    )
    item = build_link_grounded_research_item(article, config=CONFIG, agent=AgentName.PEER)
    assert item is not None
    assert item.publication_date == D
    assert item.is_dated_development is True


def test_date_only_precision_is_still_trusted_as_dated_development():
    article = _unavailable_article(
        "JPMorgan announces new AI partnership with OpenAI", precision=DatePrecision.DATE_ONLY,
    )
    item = build_link_grounded_research_item(article, config=CONFIG, agent=AgentName.PEER)
    assert item is not None
    assert item.is_dated_development is True


def test_month_precision_is_never_trusted_as_a_dated_development():
    article = _unavailable_article(
        "JPMorgan announces new AI partnership with OpenAI", precision=DatePrecision.MONTH,
    )
    item = build_link_grounded_research_item(article, config=CONFIG, agent=AgentName.PEER)
    assert item is not None
    assert item.publication_date is None  # never carried forward as an exact date
    assert item.is_dated_development is False


# ---------------------------------------------------------------------------
# `company` vs `publisher`: the publisher must never be fed into `company`
# (the field clustering reads as a subject/entity signal) -- see
# tests/events/test_link_grounded_clustering.py for the clustering-level
# proof of why this matters.
# ---------------------------------------------------------------------------


def test_company_is_not_the_publisher():
    article = _unavailable_article("JPMorgan announces new AI partnership with OpenAI")
    item = build_link_grounded_research_item(article, config=CONFIG, agent=AgentName.PEER)
    assert item is not None
    assert item.company == ""  # no genuine subject company is derivable without an LLM call
    assert item.publisher == "Financial Times"  # publisher keeps its own, independent semantics


def test_missing_publication_date_is_never_invented():
    article = _unavailable_article(
        "JPMorgan announces new AI partnership with OpenAI", publication_date=None, precision=DatePrecision.UNKNOWN,
    )
    item = build_link_grounded_research_item(article, config=CONFIG, agent=AgentName.PEER)
    assert item is not None
    assert item.publication_date is None
    assert item.is_dated_development is False
