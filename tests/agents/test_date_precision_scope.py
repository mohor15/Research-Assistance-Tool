"""Phase 14 date-recovery implementation: (1) a MONTH-precision date must
never be carried forward into `ResearchItem.publication_date` as if it
were exact (Fix D), and (2) date recovery must never bypass the existing,
unmodified relevance architecture -- an irrelevant article (the real
Michigan LIFT case) stays irrelevant regardless of whether its date is
now recoverable. Both exercised through the real `ResearchAgent.analyze()`
code path, not just bare helper functions."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.evidence.conversion import article_to_evidence
from research_assistant.evidence.normalizer import normalize_article
from research_assistant.models.enums import EventType
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
)
from research_assistant.retrieval.models import ExtractedArticle

from tests.agents.conftest import FakeLLMClient

AI_RELEVANT_TEXT = (
    "Example Bank today announced the expansion of its Ask Global Payments "
    "Solutions platform with new generative AI intelligence capabilities, "
    "giving treasury clients deeper insight into cross-border payment flows."
)

# Verbatim in spirit to the real, live-confirmed Michigan LIFT press
# release: a genuine, substantial corporate announcement with zero AI
# content -- manufacturing/community investment only.
MICHIGAN_LIFT_STYLE_TEXT = (
    "Example Bank today announced the launch of a new regional initiative "
    "to accelerate industrial innovation, scale manufacturing growth, and "
    "strengthen domestic competitiveness. The program brings together "
    "manufacturers, researchers, and policy experts to identify "
    "opportunities for scaling advanced production techniques across "
    "multiple regions over the coming years, continuing the bank's "
    "long-standing commitment to community and economic development."
)


def _relevant_assessment(**overrides) -> AgentAssessment:
    kwargs = dict(
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="A real AI/payments development."),
        category=TopicCategory.ENTERPRISE_AI,
        event_type=EventType.PRODUCT_LAUNCH,
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="x"),
    )
    kwargs.update(overrides)
    return AgentAssessment(**kwargs)


# ---------------------------------------------------------------------------
# Tests 17-22: precision/period behavior, via the real analyze() path.
# ---------------------------------------------------------------------------


def test_17_exact_date_is_preserved_on_the_research_item(config, peer_source):
    """An ordinary day-exact date is unaffected by this change -- the
    ResearchItem gets the real, exact publication_date."""
    extracted = ExtractedArticle(url="https://example.com/exact", title="t", text=AI_RELEVANT_TEXT, published_at=datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc))
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)
    agent = PeerAgent(FakeLLMClient([_relevant_assessment()]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.publication_date == datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)


def test_18_month_only_date_does_not_reach_the_research_item_as_exact(config, peer_source):
    """The core Fix D behavior: a MONTH-precision Article date (e.g. the
    BofA URL fallback's "September 2026") must become None on the
    ResearchItem -- never silently treated as an exact day."""
    extracted = ExtractedArticle(
        url="https://example.com/month", title="t", text=AI_RELEVANT_TEXT,
        published_at=datetime(2026, 9, 1, tzinfo=timezone.utc), published_at_precision="month",
    )
    article = normalize_article(extracted, source=peer_source)
    assert article.publication_date_precision.value == "month"  # confirms the Article itself kept it
    evidence = article_to_evidence(article)
    agent = PeerAgent(FakeLLMClient([_relevant_assessment()]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.publication_date is None  # suppressed, not fabricated into day 1


def test_19_unknown_date_remains_unknown(config, peer_source):
    extracted = ExtractedArticle(url="https://example.com/unknown", title="t", text=AI_RELEVANT_TEXT)
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)
    agent = PeerAgent(FakeLLMClient([_relevant_assessment()]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.publication_date is None


def test_20_month_only_date_is_never_converted_to_day_one_on_the_research_item(config, peer_source):
    """Even though the Article's own placeholder day IS literally day 1
    (by construction of the month fallback), the ResearchItem must not
    expose that placeholder as if it were a real fact -- it must be None,
    not datetime(year, month, 1)."""
    extracted = ExtractedArticle(
        url="https://example.com/month2", title="t", text=AI_RELEVANT_TEXT,
        published_at=datetime(2026, 9, 1, tzinfo=timezone.utc), published_at_precision="month",
    )
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)
    agent = PeerAgent(FakeLLMClient([_relevant_assessment()]), config)

    item = agent.analyze(article, evidence)

    assert item.publication_date != datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert item.publication_date is None


def test_21_month_only_date_is_never_converted_to_the_current_day(config, peer_source):
    extracted = ExtractedArticle(
        url="https://example.com/month3", title="t", text=AI_RELEVANT_TEXT,
        published_at=datetime(2026, 7, 1, tzinfo=timezone.utc), published_at_precision="month",
    )
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)
    agent = PeerAgent(FakeLLMClient([_relevant_assessment()]), config)

    item = agent.analyze(article, evidence)

    assert item.publication_date is None  # not "now", not any invented day


def test_22_a_month_precision_item_can_never_satisfy_an_exact_bounded_event_date(config, peer_source):
    """End-to-end through the real event-building logic (unmodified):
    `events/builder.py::_select_event_date` requires `publication_date is
    not None` -- since Fix D already nulls it for MONTH precision, this
    item can never seed an exact Event.event_date, and therefore can never
    qualify for a bounded operational digest."""
    from research_assistant.events.builder import build_event_draft
    from research_assistant.events.clustering import Cluster

    extracted = ExtractedArticle(
        url="https://example.com/month4", title="t", text=AI_RELEVANT_TEXT,
        published_at=datetime(2026, 9, 1, tzinfo=timezone.utc), published_at_precision="month",
    )
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)
    agent = PeerAgent(FakeLLMClient([_relevant_assessment()]), config)
    item = agent.analyze(article, evidence)
    assert item is not None

    draft = build_event_draft(Cluster(items=[item]))
    assert draft.event_date is None


# ---------------------------------------------------------------------------
# Tests 23-26: scope/relevance regression -- date recovery must never
# bypass the existing, unmodified relevance architecture.
# ---------------------------------------------------------------------------


def test_23_a_dated_but_irrelevant_article_does_not_become_a_research_item(config, peer_source):
    """The real Michigan LIFT regression, reproduced: even with a fully
    recovered, exact publication date, an article with zero AI/fintech
    content must not become a ResearchItem. This reaches the agent's own
    deterministic pre-filter -- `analyze()` returns None before any LLM
    call, exactly as confirmed live."""
    extracted = ExtractedArticle(
        url="https://example.com/michigan-lift", title="Regional industrial initiative launches",
        text=MICHIGAN_LIFT_STYLE_TEXT, published_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
    )
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)
    llm = FakeLLMClient([])  # must NEVER be called -- the pre-filter should reject this first
    agent = PeerAgent(llm, config)

    item = agent.analyze(article, evidence)

    assert item is None
    assert len(llm.calls) == 0  # confirms the LLM was never reached


def test_24_the_same_irrelevant_article_with_a_recovered_bofa_style_month_date_is_still_rejected(config, peer_source):
    """Identical to test 23, except the date comes via the MONTH-precision
    path (simulating a BofA-style URL-recovered date) -- recovering ANY
    date, at any precision, must not change the relevance outcome."""
    extracted = ExtractedArticle(
        url="https://example.com/michigan-lift-2", title="Regional industrial initiative launches",
        text=MICHIGAN_LIFT_STYLE_TEXT, published_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        published_at_precision="month",
    )
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)
    llm = FakeLLMClient([])
    agent = PeerAgent(llm, config)

    item = agent.analyze(article, evidence)

    assert item is None
    assert len(llm.calls) == 0


def test_25_recovering_a_date_does_not_bypass_routing(config):
    """Routing is purely source-category-based and never looks at
    publication_date at all -- confirmed directly, so date recovery
    structurally cannot influence it."""
    from research_assistant.agents.router import route_articles
    from research_assistant.models.enums import AgentName

    extracted = ExtractedArticle(
        url="https://example.com/michigan-lift-3", title="t", text=MICHIGAN_LIFT_STYLE_TEXT,
        published_at=datetime(2026, 9, 1, tzinfo=timezone.utc), published_at_precision="month",
    )
    from tests.agents.conftest import SourceCategory, SourceDefinition, SourcePriority, SourceType

    source = SourceDefinition(
        name="JPMorgan Chase", domain="jpmorganchase.com", urls=["https://jpmorganchase.com/newsroom"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )
    article = normalize_article(extracted, source=source)
    routing = route_articles([article], get_config_for_test())

    assert article in routing.by_agent.get(AgentName.PEER, [])  # routed regardless of date/precision


def get_config_for_test():
    from research_assistant.config import get_config

    return get_config()


def test_26_a_dated_unrelated_article_does_not_become_an_event(config, peer_source):
    """Full chain, one level further than test 23: confirms zero
    ResearchItems means zero Events can ever form from this article,
    using the real EventService."""
    from research_assistant.events.clock import FixedClock
    from research_assistant.events.service import EventService

    extracted = ExtractedArticle(
        url="https://example.com/michigan-lift-4", title="Regional industrial initiative launches",
        text=MICHIGAN_LIFT_STYLE_TEXT, published_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
    )
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)
    agent = PeerAgent(FakeLLMClient([]), config)
    item = agent.analyze(article, evidence)
    assert item is None

    event_service = EventService(config, clock=FixedClock(datetime(2026, 10, 3, tzinfo=timezone.utc)))
    events = event_service.build_events([item for item in [item] if item is not None])
    assert events == []
