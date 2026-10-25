"""Validation of the clustering concern raised against link-grounded items:
`build_link_grounded_research_item` sets `company = article.source_name`
(the publisher, e.g. "Financial Times"), not the real subject company.
Does that cause unrelated same-publisher stories to incorrectly merge into
one Event?

These tests call the REAL clustering primitives (`should_cluster`,
`combined_score`) and the REAL `build_link_grounded_research_item` against
synthetic Article fixtures -- no network, no LLM, no production data."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.agents.link_grounded import build_link_grounded_research_item
from research_assistant.config import get_config
from research_assistant.events.similarity import entity_overlap, is_event_type_conflict, should_cluster
from research_assistant.models.article import Article, DatePrecision
from research_assistant.models.enums import AgentName, SourcePriority, SourceType
from research_assistant.retrieval.models import ExtractionMethod, ExtractionStatus

CONFIG = get_config()
D = datetime(2026, 10, 4, tzinfo=timezone.utc)


def _ft_article(title: str, *, suffix: str) -> Article:
    return Article(
        id=f"art-{suffix}", url=f"https://www.ft.com/content/{suffix}",
        canonical_url=f"https://www.ft.com/content/{suffix}", domain="ft.com",
        source_name="Financial Times", source_type=SourceType.SECONDARY,
        source_priority=SourcePriority.HIGH, publisher="Financial Times", title=title, author=None,
        publication_date=D, publication_date_precision=DatePrecision.DATE_ONLY, retrieved_at=D,
        raw_text=None, clean_text=None, content_hash=None,
        extraction_method=ExtractionMethod.NONE, extraction_status=ExtractionStatus.FAILED,
        metadata={"reason": "http_error", "content_unavailable_reason": "HTTP_403"},
    )


def _item(title: str, suffix: str):
    article = _ft_article(title, suffix=suffix)
    item = build_link_grounded_research_item(article, config=CONFIG, agent=AgentName.FINANCIAL_NEWS)
    assert item is not None, f"fixture title {title!r} was rejected by the construction gate"
    return item


def test_case_a_different_event_types_do_not_cluster():
    """Case A: same publisher, DIFFERENT event types (partnership vs
    acquisition). The existing `is_event_type_conflict` hard veto already
    protects this case regardless of the shared `company="Financial Times"`
    field."""
    a = _item("OpenAI announces major partnership with Microsoft", suffix="a")
    b = _item("JPMorgan announces acquisition of an AI trading startup", suffix="b")

    assert is_event_type_conflict(a, b) is True
    assert should_cluster(a, b) is False


def test_case_b_unrelated_same_publisher_stories_no_longer_falsely_cluster():
    """Case B (FIX VERIFIED): two FT stories about completely different
    companies and different deals, with the SAME event type (partnership),
    must produce TWO separate Events -- not one. Before the fix,
    `company="Financial Times"` on both items drove entity_overlap to 1.0,
    which (combined with near-identical boilerplate content_similarity)
    cleared CLUSTER_THRESHOLD despite the stories sharing no real-world
    subject. With `company` no longer carrying the publisher, entity_overlap
    is 0.0 and the two items correctly do not cluster."""
    a = _item("Anthropic announces partnership with Google", suffix="c")
    b = _item("OpenAI announces partnership with Oracle", suffix="d")

    assert a.company == "" and b.company == ""  # publisher no longer leaks into company
    assert entity_overlap(a, b) == 0.0
    assert is_event_type_conflict(a, b) is False
    assert should_cluster(a, b) is False  # fixed: no longer a false positive -- two separate Events


def test_case_c_reworded_same_development_no_longer_clusters_on_wording_alone():
    """Case C: genuinely the same development, reworded with different
    phrasing. HONEST, UNFORCED RESULT after the fix: this pair no longer
    clusters. Before the fix, `company="Financial Times"` on both items
    supplied entity_overlap=1.0 (the single largest-weighted signal,
    0.35) which was doing the real work of pushing combined_score over
    threshold; title_similarity (0.429, moderate wording overlap) and
    content_similarity (inflated by near-identical link-grounded
    boilerplate) alone are not sufficient. This is the accepted,
    unforced trade-off of removing the false entity signal -- see Case E
    in tests/pipeline/test_link_grounded_integration.py for the
    same-headline (title_similarity=1.0) case, which is at the opposite,
    near-passing end of this same boundary."""
    a = _item("OpenAI announces partnership with Microsoft", suffix="e")
    b = _item("OpenAI and Microsoft enter major AI partnership", suffix="f")

    assert entity_overlap(a, b) == 0.0
    assert should_cluster(a, b) is False  # down from True before the fix -- not forced back


def test_case_d_same_real_subject_different_development_do_not_cluster():
    """Case D: same real subject (OpenAI), but two DIFFERENT development
    types (model release vs acquisition) -- correctly does not cluster,
    protected by the same event-type hard veto as Case A."""
    a = _item("OpenAI announces model release of a new AI system", suffix="g")
    b = _item("OpenAI announces acquisition of an AI startup", suffix="h")

    assert is_event_type_conflict(a, b) is True
    assert should_cluster(a, b) is False
