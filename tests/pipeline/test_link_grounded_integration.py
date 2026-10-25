"""End-to-end (through the real LangGraph) proof that content-grounded and
link-grounded intelligence coexist correctly: both can reach the digest,
neither is fabricated, link-grounded reports never call the LLM, and
CURRENT/Morning date semantics are unchanged."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.agents.financial_news import FinancialNewsAgent
from research_assistant.agents.report import ReportAgent
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.llm.errors import LLMUnavailableError
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, EventType, GroundingType, SourcePriority, SourceType
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
)
from research_assistant.pipeline import PipelineServices, PipelineState, build_graph
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import ExtractedArticle, ExtractionStatus, SourceCategory, SourceDefinition

from tests.pipeline.conftest import FakeRetrievalService, real_normalization_service
from tests.pipeline.test_graph import FakeLLMClient
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative

D = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)


def _ft_source() -> SourceDefinition:
    # Generic real secondary source (no research_domain override) --
    # routes to FinancialNewsAgent exactly like "Financial Times" used to
    # before FT gained its own AI/Technology specialist route.
    return SourceDefinition(
        name="Reuters", domain="reuters.com", urls=["https://www.reuters.com/technology/artificial-intelligence/"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
    )


def _relevant_assessment(**overrides):
    kwargs = dict(
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="Concrete FT development."),
        category=TopicCategory.FINANCIAL_SERVICES,
        summary="FT reported a concrete, retrieved development.",
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="x"),
    )
    kwargs.update(overrides)
    return AgentAssessment(**kwargs)


_SAME_DEVELOPMENT_TITLE = "Barclays signs major AI infrastructure partnership with Nvidia"


def _mixed_ft_articles() -> list:
    return [
        # content-grounded: real body, real Evidence -- same headline as the
        # link-grounded article below (e.g. two retrieval attempts against
        # the same underlying story), so clustering has a genuine
        # same-development signal (title similarity) to go on, independent
        # of the `company` field (see test_company_is_not_the_publisher and
        # tests/events/test_link_grounded_clustering.py for why `company`
        # must never be the publisher).
        ExtractedArticle(
            url="https://www.ft.com/content/aaaa-aaaa", title=_SAME_DEVELOPMENT_TITLE,
            text="Barclays today announced a major AI infrastructure partnership with Nvidia "
                 "to expand its trading operations, the companies said. " * 3,
            published_at=D, status=ExtractionStatus.SUCCESS,
        ),
        # link-grounded: content unavailable, same development, same headline
        ExtractedArticle(
            url="https://www.ft.com/content/bbbb-bbbb", title=_SAME_DEVELOPMENT_TITLE,
            text=None, published_at=D, status=ExtractionStatus.FAILED,
            metadata={"reason": "http_error", "content_unavailable_reason": "HTTP_403"},
        ),
        # inaccessible AND generic -- must produce nothing
        ExtractedArticle(
            url="https://www.ft.com/content/cccc-cccc", title="Markets react to global events",
            text=None, published_at=D, status=ExtractionStatus.FAILED,
            metadata={"reason": "robots_disallowed", "content_unavailable_reason": "ROBOTS_BLOCKED"},
        ),
    ]


def _build_services(config, *, digest_type_period=(D - timedelta(hours=12), D + timedelta(hours=1))):
    # event_type/summary match the content-grounded article's real subject
    # (see _mixed_ft_articles) so clustering has a genuine same-development
    # signal (title + event-type match) to work with, independent of the
    # `company` field.
    financial_news_agent = FinancialNewsAgent(
        FakeLLMClient([_relevant_assessment(
            event_type=EventType.PARTNERSHIP,
            summary="Barclays today announced a major AI infrastructure partnership with Nvidia "
                    "to expand its trading operations, the companies said.",
        )]),
        config,
    )
    report_agent = ReportAgent(ReportingFakeLLMClient([make_narrative()]))
    retrieval_service = FakeRetrievalService(
        sources=[_ft_source()], articles_by_source={"Reuters": _mixed_ft_articles()},
    )
    return PipelineServices(
        config=config,
        retrieval_service=retrieval_service,
        normalization_service=real_normalization_service(),
        agents={AgentName.FINANCIAL_NEWS: financial_news_agent},
        event_service=EventService(config, clock=FixedClock(D)),
        report_agent=report_agent,
        digest_service=DigestService(config, clock=FixedClock(D)),
        clock=FixedClock(D),
    )


def test_ft_mixed_retrieval_produces_both_content_and_link_grounded_paths(config):
    """Test 9: 3 FT articles discovered (1 content-grounded, 1 substantive
    link-grounded, 1 generic-and-inaccessible). Expected: all 3 retained as
    Articles; exactly 2 ResearchItems (content + link); the generic one
    produces nothing; no fabricated evidence anywhere."""
    services = _build_services(config)
    period_start, period_end = D - timedelta(hours=12), D + timedelta(hours=1)

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.MORNING, period_start=period_start, period_end=period_end)
    )

    assert len(result["articles"]) == 3  # all 3 discovered articles retained, none dropped
    research_items = result["research_items"]
    assert len(research_items) == 2  # the generic/inaccessible one produced nothing

    by_grounding = {item.grounding_type: item for item in research_items}
    assert GroundingType.CONTENT in by_grounding
    assert GroundingType.LINK in by_grounding
    link_item = by_grounding[GroundingType.LINK]
    assert link_item.title == "Barclays signs major AI infrastructure partnership with Nvidia"
    assert link_item.evidence == []
    assert link_item.entities == []


def _link_only_services(config):
    """A single link-grounded-only FT article -- no content-grounded item
    in the same run, so clustering has nothing to merge it with (see
    `test_same_development_content_and_link_grounded_are_merged_not_duplicated`
    below for the deliberate merge case)."""
    link_only_articles = [
        ExtractedArticle(
            url="https://www.ft.com/content/bbbb-bbbb",
            title="Barclays signs major AI infrastructure partnership with Nvidia",
            text=None, published_at=D, status=ExtractionStatus.FAILED,
            metadata={"reason": "http_error", "content_unavailable_reason": "HTTP_403"},
        ),
    ]
    financial_news_agent = FinancialNewsAgent(FakeLLMClient([]), config)  # never called -- no content-grounded item
    retrieval_service = FakeRetrievalService(
        sources=[_ft_source()], articles_by_source={"Reuters": link_only_articles},
    )
    return PipelineServices(
        config=config, retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.FINANCIAL_NEWS: financial_news_agent}, event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([])),  # never called -- deterministic report path
        digest_service=DigestService(config, clock=FixedClock(D)), clock=FixedClock(D),
    )


def test_link_grounded_event_reaches_the_digest(config):
    """Test 2: a dated, relevant, link-grounded development must be able to
    reach the digest -- digest qualification is based on event_date, not
    on evidence/content availability."""
    services = _link_only_services(config)
    period_start, period_end = D - timedelta(hours=12), D + timedelta(hours=1)

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.MORNING, period_start=period_start, period_end=period_end)
    )

    events = result["events"]
    assert len(events) == 1
    assert events[0].grounding_type == GroundingType.LINK
    assert result["digest"] is not None
    assert events[0].id in result["digest"].events  # actually qualified into the digest


def test_link_grounded_report_never_calls_the_llm(config):
    """FakeLLMClient([]) (no scripted responses) is used for BOTH the
    financial-news agent and the report agent -- if the link-grounded path
    called either, it would raise "ran out of scripted responses" and this
    test would fail with that exception instead of passing."""
    services = _link_only_services(config)
    period_start, period_end = D - timedelta(hours=12), D + timedelta(hours=1)

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.MORNING, period_start=period_start, period_end=period_end)
    )

    assert len(result["reports"]) == 1
    link_report = result["reports"][0]
    assert "automated access to the article content was unavailable" in link_report.opening_context
    assert link_report.details.companies == []  # never fabricated


def test_same_headline_content_and_link_grounded_items_cluster_into_one_event(config):
    """Test 15 (REVISED again after the near-duplicate-title clustering
    override -- see `events.similarity.is_near_duplicate_title_same_day`,
    added to fix a confirmed real-run duplication: the same PR Newswire
    release syndicated under two URLs produced two separate Events purely
    because the weighted score landed just under threshold).

    Previously documented as an accepted trade-off: a content-grounded
    item and a link-grounded item with an IDENTICAL headline and the same
    publication date landed at a weighted score of ~0.54, just under the
    0.55 `CLUSTER_THRESHOLD` (a link-grounded item contributes no entity
    signal, having no body to extract from), so they stayed as two
    separate Events. The near-duplicate-title override now correctly
    recognizes this exact case -- an identical headline on the same day
    is overwhelming evidence of the same real-world development,
    regardless of which of the two items happens to have a retrievable
    body -- and merges them into ONE Event, resolving to CONTENT grounding
    (the richer of the two members) while still carrying both items'
    ids. This eliminates exactly the kind of same-story duplication the
    override exists to fix, rather than presenting a reader with two
    fragmented cards for one real development."""
    services = _build_services(config)
    period_start, period_end = D - timedelta(hours=12), D + timedelta(hours=1)

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.MORNING, period_start=period_start, period_end=period_end)
    )

    assert len(result["research_items"]) == 2  # both items were genuinely produced
    events = result["events"]
    assert len(events) == 1  # now correctly merged -- no more duplicate same-headline Events
    merged_event = events[0]
    assert merged_event.grounding_type == GroundingType.CONTENT  # the richer member wins
    assert set(merged_event.research_items) == {item.id for item in result["research_items"]}


def test_link_grounded_original_url_survives_to_the_digest(config):
    """URL-preservation trace: Article.url -> ResearchItem.url -> Event
    (via research_items) -> Report/Digest. A reader must always be able
    to reach the original article, since its content was never captured."""
    services = _link_only_services(config)
    period_start, period_end = D - timedelta(hours=12), D + timedelta(hours=1)
    original_url = "https://www.ft.com/content/bbbb-bbbb"

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.MORNING, period_start=period_start, period_end=period_end)
    )

    link_item = result["research_items"][0]
    assert link_item.url == original_url

    report = result["reports"][0]
    report_urls = [ref.url for ref in report.sources]
    assert original_url in report_urls  # the reader can always reach the original article


def _over_inclusion_articles() -> list:
    return [
        ExtractedArticle(
            url="https://www.ft.com/content/eeee-eeee", title="AI trends in 2026",
            text=None, published_at=D, status=ExtractionStatus.FAILED,
            metadata={"reason": "http_error", "content_unavailable_reason": "HTTP_403"},
        ),
        ExtractedArticle(
            url="https://www.ft.com/content/ffff-ffff", title="What AI means for businesses",
            text=None, published_at=D, status=ExtractionStatus.FAILED,
            metadata={"reason": "http_error", "content_unavailable_reason": "HTTP_403"},
        ),
        ExtractedArticle(
            url="https://www.ft.com/content/gggg-gggg", title="The future of technology",
            text=None, published_at=D, status=ExtractionStatus.FAILED,
            metadata={"reason": "http_error", "content_unavailable_reason": "HTTP_403"},
        ),
    ]


def test_generic_commentary_titles_never_become_link_grounded_items(config):
    """Test 8 / over-inclusion check: generic, evergreen-commentary titles
    (no concrete event-type phrase) must produce NOTHING, even though they
    are topically on-subject and would otherwise be routed to an agent."""
    financial_news_agent = FinancialNewsAgent(FakeLLMClient([]), config)
    retrieval_service = FakeRetrievalService(
        sources=[_ft_source()], articles_by_source={"Reuters": _over_inclusion_articles()},
    )
    services = PipelineServices(
        config=config, retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.FINANCIAL_NEWS: financial_news_agent}, event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([])), digest_service=DigestService(config, clock=FixedClock(D)),
        clock=FixedClock(D),
    )
    period_start, period_end = D - timedelta(hours=12), D + timedelta(hours=1)

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.MORNING, period_start=period_start, period_end=period_end)
    )

    assert len(result["articles"]) == 3  # discovery still retains all 3
    assert result["research_items"] == []  # but none qualify as a development
    assert result["events"] == []
    assert result["digest"].events == []


def test_ft_five_discovered_two_retrieved_three_inaccessible(config):
    """Part 7: FT-specific mocked scenario -- 5 articles discovered, 2 with
    retrieved content (content-grounded), 3 inaccessible (of which some
    are substantive enough to become link-grounded and some are generic)."""
    articles = [
        ExtractedArticle(
            url="https://www.ft.com/content/1111-1111", title="FT reports on AI deployment at a major bank",
            text="A major bank today announced a significant AI deployment across its operations. " * 3,
            published_at=D, status=ExtractionStatus.SUCCESS,
        ),
        ExtractedArticle(
            url="https://www.ft.com/content/2222-2222", title="FT reports on a new AI regulatory filing",
            text="Regulators today published new guidance on AI governance in financial services. " * 3,
            published_at=D, status=ExtractionStatus.SUCCESS,
        ),
        ExtractedArticle(
            url="https://www.ft.com/content/3333-3333",
            title="Barclays signs major AI infrastructure partnership with Nvidia",
            text=None, published_at=D, status=ExtractionStatus.FAILED,
            metadata={"reason": "http_error", "content_unavailable_reason": "HTTP_403"},
        ),
        ExtractedArticle(
            url="https://www.ft.com/content/4444-4444",
            title="HSBC announces acquisition of an AI compliance startup",
            text=None, published_at=D, status=ExtractionStatus.FAILED,
            metadata={"reason": "robots_disallowed", "content_unavailable_reason": "ROBOTS_BLOCKED"},
        ),
        ExtractedArticle(
            url="https://www.ft.com/content/5555-5555", title="Markets react to global events",
            text=None, published_at=D, status=ExtractionStatus.FAILED,
            metadata={"reason": "robots_disallowed", "content_unavailable_reason": "ROBOTS_BLOCKED"},
        ),
    ]
    financial_news_agent = FinancialNewsAgent(
        FakeLLMClient([_relevant_assessment(), _relevant_assessment()]), config,
    )
    retrieval_service = FakeRetrievalService(sources=[_ft_source()], articles_by_source={"Reuters": articles})
    services = PipelineServices(
        config=config, retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.FINANCIAL_NEWS: financial_news_agent}, event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative(), make_narrative()])),
        digest_service=DigestService(config, clock=FixedClock(D)), clock=FixedClock(D),
    )
    period_start, period_end = D - timedelta(hours=12), D + timedelta(hours=1)

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.MORNING, period_start=period_start, period_end=period_end)
    )

    assert len(result["articles"]) == 5  # all 5 discovered, none dropped
    research_items = result["research_items"]
    # 2 content-grounded + 2 substantive link-grounded (Barclays, HSBC);
    # the generic "Markets react to global events" produces nothing.
    assert len(research_items) == 4
    by_grounding = {}
    for item in research_items:
        by_grounding.setdefault(item.grounding_type, []).append(item)
    assert len(by_grounding[GroundingType.CONTENT]) == 2
    assert len(by_grounding[GroundingType.LINK]) == 2


def test_current_mode_excludes_link_grounded_item_with_unknown_publication_date(config):
    """Test 7 (strict upstream date-gate version): an inaccessible article
    with NO reliable publication date must never enter CURRENT merely
    because it was discovered recently. CURRENT's hard upstream date gate
    (`pipeline.nodes._apply_current_date_gate`) now drops an undated
    article BEFORE routing, so it never reaches the LINK-grounded builder,
    ResearchItem creation, or Event creation at all -- stricter than the
    previous behavior (which let a LINK-grounded item form with
    `publication_date=None` and relied on a later EventBuilder guard to
    keep it out of the digest)."""
    undated_articles = [
        ExtractedArticle(
            url="https://www.ft.com/content/dddd-dddd",
            title="Goldman Sachs announces new AI trading partnership with Anthropic",
            text=None, published_at=None, status=ExtractionStatus.FAILED,
            metadata={"reason": "http_error", "content_unavailable_reason": "HTTP_403"},
        ),
    ]
    financial_news_agent = FinancialNewsAgent(FakeLLMClient([]), config)
    retrieval_service = FakeRetrievalService(
        sources=[_ft_source()], articles_by_source={"Reuters": undated_articles},
    )
    services = PipelineServices(
        config=config, retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.FINANCIAL_NEWS: financial_news_agent}, event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([])), digest_service=DigestService(config, clock=FixedClock(D)),
        clock=FixedClock(D),
    )
    now = D
    period_start, period_end = now - timedelta(hours=24), now

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=period_start, period_end=period_end)
    )

    # The upstream date gate drops the undated article before routing --
    # zero research items, zero events, never even reaches the
    # LINK-grounded builder.
    assert result["research_items"] == []
    assert result["events"] == []
    assert result["digest"].events == []
    assert result["metrics"]["current_dropped_undated"] == 1
