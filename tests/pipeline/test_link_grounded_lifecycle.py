"""Fixture-based, end-to-end proof of the complete inaccessible-article
lifecycle through the real LangGraph: discovery -> Article retained ->
no Evidence (no clean_text/content_hash) -> LINK-grounded ResearchItem
(no Ollama call) -> Event -> deterministic Report (no Ollama call) ->
Digest, with the original article URL preserved at every stage.

No network, no Ollama, no production DB, no scheduler -- FakeRetrievalService
and scripted FakeLLMClients only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.agents.financial_news import FinancialNewsAgent
from research_assistant.agents.report import ReportAgent
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, GroundingType, SourcePriority, SourceType
from research_assistant.pipeline import PipelineServices, PipelineState, build_graph
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import ExtractedArticle, ExtractionStatus, SourceCategory, SourceDefinition

from tests.pipeline.conftest import FakeRetrievalService, real_normalization_service
from tests.pipeline.test_graph import FakeLLMClient
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient

D = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)

_ORIGINAL_URL = "https://www.ft.com/content/aaaa-aaaa"
_SUBSTANTIVE_TITLE = "Barclays signs major AI infrastructure partnership with Nvidia"


def _ft_source() -> SourceDefinition:
    # Generic real secondary source (no research_domain override) -- routes
    # to FinancialNewsAgent exactly like "Financial Times" used to before FT
    # gained its own AI/Technology specialist route (agents/router.py).
    return SourceDefinition(
        name="Reuters", domain="reuters.com", urls=["https://www.reuters.com/technology/artificial-intelligence/"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
    )


def _access_blocked_article(title: str, *, url: str = _ORIGINAL_URL, published_at=D) -> ExtractedArticle:
    """An article discovered (title/source/date known) but whose body could
    not be retrieved -- a 403-style access-blocked failure, never robots.txt
    bypass or anti-bot evasion (no retrieval behavior is exercised here at
    all -- this is a pre-built fixture, not a live fetch)."""
    return ExtractedArticle(
        url=url, title=title, text=None, published_at=published_at, status=ExtractionStatus.FAILED,
        metadata={"reason": "http_error", "content_unavailable_reason": "HTTP_403"},
    )


def _services(config, articles, *, llm_responses=(), report_responses=()):
    fake_news_llm = FakeLLMClient(list(llm_responses))
    fake_report_llm = ReportingFakeLLMClient(list(report_responses))
    financial_news_agent = FinancialNewsAgent(fake_news_llm, config)
    report_agent = ReportAgent(fake_report_llm)
    retrieval_service = FakeRetrievalService(sources=[_ft_source()], articles_by_source={"Reuters": articles})
    services = PipelineServices(
        config=config, retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.FINANCIAL_NEWS: financial_news_agent}, event_service=EventService(config, clock=FixedClock(D)),
        report_agent=report_agent, digest_service=DigestService(config, clock=FixedClock(D)), clock=FixedClock(D),
    )
    return services, fake_news_llm, fake_report_llm


def test_full_inaccessible_article_lifecycle_reaches_the_digest(config):
    """Points 1-12: a single trusted, dated, substantive, access-blocked
    article walks the entire LINK-grounded lifecycle through to the digest,
    with no Evidence and no Ollama call anywhere along the path."""
    services, fake_news_llm, fake_report_llm = _services(
        config, [_access_blocked_article(_SUBSTANTIVE_TITLE)],
    )
    period_start, period_end = D - timedelta(hours=12), D + timedelta(hours=1)

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.MORNING, period_start=period_start, period_end=period_end)
    )

    # 2. Article retained despite extraction failure.
    assert len(result["articles"]) == 1
    article = result["articles"][0]
    assert article.extraction_status == ExtractionStatus.FAILED
    assert article.clean_text is None
    assert article.content_hash is None

    # 3. No Evidence created (gated on clean_text/content_hash, both absent).
    assert result["evidence"] == []

    # 4/5. A LINK-grounded ResearchItem with honest, unfabricated fields.
    assert len(result["research_items"]) == 1
    item = result["research_items"][0]
    assert item.grounding_type == GroundingType.LINK
    assert item.company == ""  # no publisher leaked in as a fake subject
    assert item.publisher == "Reuters"
    assert item.url == _ORIGINAL_URL
    assert item.evidence == []
    assert item.entities == [] and item.people == [] and item.products == [] and item.technologies == []
    assert item.deal_value is None
    # Conservative relevance -- never HIGH on metadata alone.
    assert item.financial_services_relevance.level.value in ("low", "medium")
    assert item.nomura_relevance.level.value == "low"
    assert item.strategic_significance.level.value == "low"

    # 6. No Ollama/LLM call for the LINK path (financial-news agent is never
    # invoked for this item -- the router sends it straight to the
    # deterministic link_grounded builder).
    assert fake_news_llm.calls == []

    # 7/8. EventService builds an Event from the LINK item; it is not
    # downgraded or upgraded, carries the trusted date, and invents nothing.
    assert len(result["events"]) == 1
    event = result["events"][0]
    assert event.grounding_type == GroundingType.LINK
    assert event.event_date == D
    assert event.related_companies == []

    # 9/10. Deterministic report (no LLM call), transparent about what is
    # and isn't known, and carries the original URL.
    assert fake_report_llm.calls == []
    assert len(result["reports"]) == 1
    report = result["reports"][0]
    assert "automated access to the article content was unavailable" in report.opening_context
    assert "What is not known" in report.details.narrative
    assert report.details.companies == []
    report_urls = [ref.url for ref in report.sources]
    assert _ORIGINAL_URL in report_urls

    # 11/12. The event qualifies into the digest, with the URL still
    # reachable from it.
    assert result["digest"] is not None
    assert event.id in result["digest"].events
    assert event.primary_source.url == _ORIGINAL_URL


def test_lifecycle_does_not_depend_on_company_evidence_clean_text_or_llm_output(config):
    """Explicit negative-space proof: the LINK path produces a usable,
    digest-eligible result with company="" , zero Evidence, no clean_text,
    and zero LLM calls -- none of these are silently required anywhere
    downstream."""
    services, fake_news_llm, fake_report_llm = _services(
        config, [_access_blocked_article(_SUBSTANTIVE_TITLE)],
    )
    period_start, period_end = D - timedelta(hours=12), D + timedelta(hours=1)

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.MORNING, period_start=period_start, period_end=period_end)
    )

    assert result["research_items"][0].company == ""
    assert result["evidence"] == []
    assert result["articles"][0].clean_text is None
    assert fake_news_llm.calls == []
    assert fake_report_llm.calls == []
    assert result["digest"].events == [result["events"][0].id]  # still reaches the digest


def test_mixed_two_content_three_link_one_generic(config):
    """Point 15: 2 content-grounded + 3 inaccessible-but-relevant (LINK) +
    1 inaccessible-and-generic article in one run."""
    content_articles = [
        ExtractedArticle(
            url="https://www.ft.com/content/c1", title="FT reports on AI deployment at a major bank",
            text="A major bank today announced a significant AI deployment across its operations. " * 3,
            published_at=D, status=ExtractionStatus.SUCCESS,
        ),
        ExtractedArticle(
            url="https://www.ft.com/content/c2", title="FT reports on a new AI regulatory filing",
            text="Regulators today published new guidance on AI governance in financial services. " * 3,
            published_at=D, status=ExtractionStatus.SUCCESS,
        ),
    ]
    link_articles = [
        _access_blocked_article("Barclays signs major AI infrastructure partnership with Nvidia", url="https://www.ft.com/content/l1"),
        _access_blocked_article("HSBC announces acquisition of an AI compliance startup", url="https://www.ft.com/content/l2"),
        _access_blocked_article("Santander announces new AI funding round for its ventures arm", url="https://www.ft.com/content/l3"),
    ]
    generic_article = ExtractedArticle(
        url="https://www.ft.com/content/g1", title="Markets react to global events",
        text=None, published_at=D, status=ExtractionStatus.FAILED,
        metadata={"reason": "robots_disallowed", "content_unavailable_reason": "ROBOTS_BLOCKED"},
    )
    all_articles = content_articles + link_articles + [generic_article]

    from tests.reporting.conftest import make_narrative
    from tests.pipeline.test_link_grounded_integration import _relevant_assessment

    services, fake_news_llm, fake_report_llm = _services(
        config, all_articles,
        llm_responses=[_relevant_assessment(), _relevant_assessment()],
        report_responses=[make_narrative(), make_narrative()],
    )
    period_start, period_end = D - timedelta(hours=12), D + timedelta(hours=1)

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.MORNING, period_start=period_start, period_end=period_end)
    )

    assert len(result["articles"]) == 6  # all 6 discovered, none dropped
    research_items = result["research_items"]
    assert len(research_items) == 5  # 2 content + 3 link; the generic one produces nothing

    by_grounding = {}
    for item in research_items:
        by_grounding.setdefault(item.grounding_type, []).append(item)
    assert len(by_grounding[GroundingType.CONTENT]) == 2
    assert len(by_grounding[GroundingType.LINK]) == 3

    # No Evidence for the 3 LINK items (only the 2 content items may have it).
    link_article_ids = {item.article_id for item in by_grounding[GroundingType.LINK]}
    assert all(ev.article_id not in link_article_ids for ev in result["evidence"])

    # LINK items never triggered the financial-news agent beyond the 2
    # scripted content-path calls.
    assert len(fake_news_llm.calls) == 2

    # Eligible LINK events reach the digest with their original URLs intact.
    events = result["events"]
    link_events = [e for e in events if e.grounding_type == GroundingType.LINK]
    assert len(link_events) == 3
    digest_event_ids = set(result["digest"].events)
    for event in link_events:
        assert event.id in digest_event_ids
        assert event.primary_source.url in {"https://www.ft.com/content/l1", "https://www.ft.com/content/l2", "https://www.ft.com/content/l3"}
