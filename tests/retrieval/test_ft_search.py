"""Targeted FT interactive-search discovery (retrieval.ft_search) --
additive, FT-only, never touches the network/a real browser in this suite
(deterministic fixtures/mocks only, per the task's explicit requirement)."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.config import get_config
from research_assistant.models.enums import AgentName, SourcePriority, SourceType
from research_assistant.retrieval import ft_search as ft_search_module
from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.discovery import discover_candidates
from research_assistant.retrieval.models import CachePolicy, CandidateURL, DiscoveryMethod, SourceCategory, SourceDefinition
from research_assistant.retrieval.service import RetrievalService, build_source_definitions

from tests.retrieval.fakes import FakeResponse, FakeSession

# A realistic slice of FT search-results HTML: real article links
# (`/content/<uuid>`), plus the kinds of hub/category/pagination links a
# real FT page mixes in alongside them -- exactly what `_is_ft_article_url`
# must tell apart.
_FT_SEARCH_RESULTS_HTML = """
<html><body>
<a href="/content/ea32c53d-4de7-4b34-af44-cc148b23433f">Japanese and Korean shipbuilders deploy robots</a>
<a href="/content/9b3a355e-5975-445f-9004-b95513e3856a">Masayoshi Son's AI ambitions outgrow SoftBank's balance sheet</a>
<a href="/war-in-ukraine">War in Ukraine</a>
<a href="/stream/e3402603-d253-4aa1-ac4d-fc9bdbf4ccb8">OpenAI</a>
<a href="/artificial-intelligence?page=2">Next page</a>
</body></html>
"""


def _ft_source(search_terms=None, top_k_per_query=5, enabled=True) -> SourceDefinition:
    from research_assistant.config.schema import SourceSearchConfig

    return SourceDefinition(
        name="Financial Times", domain="ft.com", urls=["https://www.ft.com/artificial-intelligence"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
        search=SourceSearchConfig(
            enabled=enabled, search_terms=search_terms or ["AI", "fintech"], top_k_per_query=top_k_per_query,
        ),
    )


# ---------------------------------------------------------------------------
# 1-3: configuration
# ---------------------------------------------------------------------------


def test_ft_is_present_in_the_real_source_registry():
    config = get_config()
    names = {s.name for s in config.sources.secondary_sources}
    assert "Financial Times" in names


def test_ft_search_is_configured_on_the_real_source_with_expected_terms():
    config = get_config()
    ft = next(s for s in config.sources.secondary_sources if s.name == "Financial Times")
    assert ft.search is not None
    assert ft.search.enabled is True
    assert ft.search.top_k_per_query > 0
    for expected in ("AI", "artificial intelligence", "generative AI", "fintech", "AI banking"):
        assert expected in ft.search.search_terms


def test_ft_search_config_flows_through_to_the_retrieval_layer_source_definition():
    """build_source_definitions must carry `search` through from
    config.schema.SourceEntry into retrieval.models.SourceDefinition --
    discovery.py reads it from there, not from raw config."""
    config = get_config()
    defs = build_source_definitions(config)
    ft = next(d for d in defs if d.name == "Financial Times")
    assert ft.search is not None
    assert ft.search.enabled is True


def test_no_other_source_has_search_configured():
    """This is an FT-specific feature, not generic infrastructure applied
    broadly -- every other configured source's `search` stays None."""
    config = get_config()
    all_entries = (
        config.sources.peers + config.sources.ai_providers + config.sources.analysts
        + [s for s in config.sources.secondary_sources if s.name != "Financial Times"]
    )
    assert all(entry.search is None for entry in all_entries)


# ---------------------------------------------------------------------------
# 4, 10: article vs. hub/search-page detection
# ---------------------------------------------------------------------------


def test_search_results_parsing_extracts_only_real_article_urls():
    source = _ft_source()
    candidates = ft_search_module._parse_ft_search_results_html(
        _FT_SEARCH_RESULTS_HTML, base_url="https://www.ft.com/search", source=source, top_k=10,
    )
    urls = {c.url for c in candidates}
    assert "https://www.ft.com/content/ea32c53d-4de7-4b34-af44-cc148b23433f" in urls
    assert "https://www.ft.com/content/9b3a355e-5975-445f-9004-b95513e3856a" in urls
    # hub/category/stream/pagination links must never become candidates
    assert not any("war-in-ukraine" in u for u in urls)
    assert not any("/stream/" in u for u in urls)
    assert not any("page=" in u for u in urls)
    assert len(candidates) == 2


def test_the_search_results_page_itself_is_never_a_candidate():
    source = _ft_source()
    html = '<html><body><a href="/search?q=AI">AI</a></body></html>'
    candidates = ft_search_module._parse_ft_search_results_html(
        html, base_url="https://www.ft.com/search", source=source, top_k=10,
    )
    assert candidates == []


def test_article_url_detection_requires_the_exact_uuid_shape():
    assert ft_search_module._is_ft_article_url("/content/ea32c53d-4de7-4b34-af44-cc148b23433f")
    assert not ft_search_module._is_ft_article_url("/content/not-a-real-uuid")
    assert not ft_search_module._is_ft_article_url("/stream/ea32c53d-4de7-4b34-af44-cc148b23433f")
    assert not ft_search_module._is_ft_article_url("/war-in-ukraine")


# ---------------------------------------------------------------------------
# 9: no crawl-time date is ever assigned
# ---------------------------------------------------------------------------


def test_search_discovered_candidates_never_carry_a_crawl_time_date():
    """A search result's discovery time must never masquerade as the
    article's publication date -- the real date comes later, from the
    article page itself, via the existing unmodified retrieve/extract path."""
    source = _ft_source()
    candidates = ft_search_module._parse_ft_search_results_html(
        _FT_SEARCH_RESULTS_HTML, base_url="https://www.ft.com/search", source=source, top_k=10,
    )
    assert all(c.published_at is None for c in candidates)
    assert all(c.discovery_method == DiscoveryMethod.SEARCH for c in candidates)


# ---------------------------------------------------------------------------
# 5, 7: top-K and dedup across queries
# ---------------------------------------------------------------------------


def test_search_respects_its_own_top_k_per_query_independent_of_agent_top_k():
    source = _ft_source()
    candidates = ft_search_module._parse_ft_search_results_html(
        _FT_SEARCH_RESULTS_HTML, base_url="https://www.ft.com/search", source=source, top_k=1,
    )
    assert len(candidates) == 1


def test_discover_via_ft_search_dedupes_across_multiple_search_terms(monkeypatch):
    """Two different search terms returning the SAME article must only
    produce one candidate -- canonical-URL dedup across queries, before
    Evidence is ever created."""
    source = _ft_source(search_terms=["AI", "fintech"])

    call_count = {"n": 0}

    def fake_search_via_http(src, term, *, http_client, cache_policy, top_k):
        call_count["n"] += 1
        return ft_search_module._parse_ft_search_results_html(
            _FT_SEARCH_RESULTS_HTML, base_url="https://www.ft.com/search", source=src, top_k=top_k,
        )

    monkeypatch.setattr(ft_search_module, "_search_via_http", fake_search_via_http)

    session = FakeSession([])
    http_client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    candidates = ft_search_module.discover_via_ft_search(source, http_client=http_client)

    assert call_count["n"] == 2  # both terms were actually queried
    urls = [c.url for c in candidates]
    assert len(urls) == len(set(urls))  # no duplicates despite two identical result sets
    assert len(candidates) == 2  # the 2 real articles, not 4


def test_disabled_search_config_produces_no_candidates():
    source = _ft_source(enabled=False)
    session = FakeSession([])
    http_client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    assert ft_search_module.discover_via_ft_search(source, http_client=http_client) == []


def test_a_source_with_no_search_config_at_all_produces_no_candidates():
    source = SourceDefinition(
        name="Other Secondary", domain="example.com", urls=["https://example.com/ai"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
    )
    session = FakeSession([])
    http_client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    assert ft_search_module.discover_via_ft_search(source, http_client=http_client) == []


# ---------------------------------------------------------------------------
# HTTP-first-then-browser tiering
# ---------------------------------------------------------------------------


def test_http_search_is_tried_first_and_browser_is_skipped_when_it_succeeds(monkeypatch):
    source = _ft_source(search_terms=["AI"])
    browser_called = {"invoked": False}

    def fake_browser(*a, **k):
        browser_called["invoked"] = True
        return []

    monkeypatch.setattr(ft_search_module, "_search_via_browser", fake_browser)

    session = FakeSession([
        FakeResponse(status_code=200, url="https://www.ft.com/search?q=AI", text=_FT_SEARCH_RESULTS_HTML),
    ])
    http_client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    candidates = ft_search_module.discover_via_ft_search(source, http_client=http_client)

    assert browser_called["invoked"] is False
    assert len(candidates) == 2


def test_browser_search_is_tried_only_when_http_search_finds_nothing(monkeypatch):
    source = _ft_source(search_terms=["AI"])
    browser_called = {"invoked": False}

    def fake_browser(src, term, *, top_k, robots=None):
        browser_called["invoked"] = True
        return [
            CandidateURL(
                url="https://www.ft.com/content/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                source_name=src.name, domain=src.domain, discovery_method=DiscoveryMethod.SEARCH,
                discovered_at=datetime.now(timezone.utc), title="Browser-found article",
            )
        ]

    monkeypatch.setattr(ft_search_module, "_search_via_browser", fake_browser)

    session = FakeSession([FakeResponse(status_code=200, url="https://www.ft.com/search?q=AI", text="<html></html>")])
    http_client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    candidates = ft_search_module.discover_via_ft_search(source, http_client=http_client)

    assert browser_called["invoked"] is True
    assert len(candidates) == 1


def test_crawl4ai_unavailable_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(ft_search_module, "CRAWL4AI_AVAILABLE", False)
    result = ft_search_module._search_via_browser(_ft_source(), "AI", top_k=5)
    assert result == []


def test_browser_search_never_bypasses_robots_txt(monkeypatch):
    """Confirmed live (2026-10-04): FT's own robots.txt disallows `/search`
    for `User-agent: *`. The browser tier never goes through `HTTPClient`
    (which is where robots.txt is normally enforced), so it must check
    explicitly -- this reproduces that disallow decision with a fake
    checker rather than hitting the real site, and proves the browser is
    never even launched when robots.txt says no."""
    monkeypatch.setattr(ft_search_module, "CRAWL4AI_AVAILABLE", True)

    launched = {"invoked": False}

    class _FakeCrawler:
        async def __aenter__(self):
            launched["invoked"] = True
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def arun(self, url, config):
            raise AssertionError("must never launch the browser when robots.txt disallows the page")

    monkeypatch.setattr(ft_search_module, "AsyncWebCrawler", lambda config: _FakeCrawler())

    class _DisallowSearch:
        def is_allowed(self, url):
            return "/search" not in url

    result = ft_search_module._search_via_browser(_ft_source(), "AI", top_k=5, robots=_DisallowSearch())

    assert result == []
    assert launched["invoked"] is False


def test_http_search_logs_an_explicit_robots_blocked_message(monkeypatch, caplog):
    """Observable, per Part 3: must read exactly 'robots blocked; skipped',
    never a silent/ambiguous zero-result log line, when FT's own robots.txt
    is the reason nothing came back."""
    import logging

    class _DisallowAll:
        def is_allowed(self, url):
            return False

    http_client = HTTPClient(
        timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=FakeSession([]), robots_checker=_DisallowAll(),
    )
    with caplog.at_level(logging.INFO, logger="research_assistant.retrieval.ft_search"):
        result = ft_search_module._search_via_http(
            _ft_source(), "AI", http_client=http_client, cache_policy=CachePolicy.CACHE, top_k=5,
        )
    assert result == []
    assert any("FT search term 'AI': robots blocked; skipped" in rec.message for rec in caplog.records)


def test_discover_via_ft_search_passes_the_http_clients_own_robots_checker_to_the_browser_tier(monkeypatch):
    """The browser tier must consult the SAME robots.txt decision the HTTP
    tier already uses -- never a separate, potentially more permissive,
    check."""
    source = _ft_source(search_terms=["AI"])
    received = {}

    def fake_browser(src, term, *, top_k, robots=None):
        received["robots"] = robots
        return []

    monkeypatch.setattr(ft_search_module, "_search_via_browser", fake_browser)

    class _FakeRobots:
        def is_allowed(self, url):
            return False

    session = FakeSession([FakeResponse(status_code=200, url="https://www.ft.com/search?q=AI", text="<html></html>")])
    http_client = HTTPClient(
        timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session, robots_checker=_FakeRobots(),
    )
    ft_search_module.discover_via_ft_search(source, http_client=http_client)

    assert received["robots"] is http_client.robots


# ---------------------------------------------------------------------------
# Integration with discovery.discover_candidates: additive, never a hub-failure blocker
# ---------------------------------------------------------------------------


def test_discover_candidates_merges_hub_and_search_results(monkeypatch):
    source = _ft_source(search_terms=["AI"])
    hub_html = '<html><body><main><a href="/content/11111111-1111-1111-1111-111111111111">A hub-listed article</a></main></body></html>'

    def fake_search_via_http(src, term, *, http_client, cache_policy, top_k):
        return ft_search_module._parse_ft_search_results_html(
            _FT_SEARCH_RESULTS_HTML, base_url="https://www.ft.com/search", source=src, top_k=top_k,
        )

    monkeypatch.setattr(ft_search_module, "_search_via_http", fake_search_via_http)

    session = FakeSession([FakeResponse(status_code=200, url=source.urls[0], text=hub_html)])
    http_client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    candidates = discover_candidates(source, http_client=http_client, max_candidates=50)
    urls = {c.url for c in candidates}
    assert "https://www.ft.com/content/11111111-1111-1111-1111-111111111111" in urls  # from the hub page
    assert "https://www.ft.com/content/ea32c53d-4de7-4b34-af44-cc148b23433f" in urls  # from search


def test_discover_candidates_recovers_via_search_when_the_hub_page_itself_fails(monkeypatch):
    """FT's search endpoint is a separate URL from its hub page -- a hub
    fetch failure must not prevent search discovery from still finding
    something, and must not raise SourceUnreachableError when it does."""
    import requests

    source = _ft_source(search_terms=["AI"])

    def fake_search_via_http(src, term, *, http_client, cache_policy, top_k):
        return ft_search_module._parse_ft_search_results_html(
            _FT_SEARCH_RESULTS_HTML, base_url="https://www.ft.com/search", source=src, top_k=top_k,
        )

    monkeypatch.setattr(ft_search_module, "_search_via_http", fake_search_via_http)

    session = FakeSession([requests.ConnectionError("dns failure")])
    http_client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    candidates = discover_candidates(source, http_client=http_client, max_candidates=50)
    assert len(candidates) == 2


# ---------------------------------------------------------------------------
# 11: FT routing to FinancialNewsAgent (pre-existing logic, reconfirmed with
# a search-discovered candidate specifically)
# ---------------------------------------------------------------------------


def test_ft_participates_in_current_morning_and_evening(config=None):
    """Source discovery (`make_discover_sources_node`) never takes
    `digest_type` into account at all -- FT is not CURRENT-only."""
    from research_assistant.config import get_config as _get_config
    from research_assistant.models.digest import DigestType
    from research_assistant.pipeline.nodes import make_discover_sources_node
    from research_assistant.pipeline.state import PipelineState
    from research_assistant.retrieval.service import RetrievalService

    config = _get_config()
    services = type("Services", (), {"retrieval_service": RetrievalService(config)})()
    node = make_discover_sources_node(services)

    for digest_type in (DigestType.CURRENT, DigestType.MORNING, DigestType.EVENING):
        result = node(PipelineState(digest_type=digest_type))
        names = {s.name for s in result["sources"]}
        assert "Financial Times" in names, f"FT missing for {digest_type}"


# ---------------------------------------------------------------------------
# 6, 8-10: period/date semantics -- the real article page's own date is
# used (never crawl time), and the existing half-open [start, end) rule
# unchanged by this feature.
# ---------------------------------------------------------------------------


def test_ft_publication_date_comes_from_the_article_page_not_the_search_result():
    """A search-discovered CandidateURL carries no date (confirmed above);
    once retrieved+extracted through the EXISTING, unmodified pipeline, the
    real date comes from the article page's own metadata."""
    from research_assistant.retrieval.extractor import extract_article

    article_html = """
    <html><head>
    <meta property="article:published_time" content="2026-10-04T09:00:00Z" />
    </head><body><article>
    <p>SoftBank's AI investment strategy is under scrutiny as its balance
    sheet absorbs new AI bets, with investors questioning the long-term
    exposure to a single technology wave.</p>
    </article></body></html>
    """
    result = extract_article(article_html, "https://www.ft.com/content/9b3a355e-5975-445f-9004-b95513e3856a")
    assert result.published_at is not None
    assert result.published_at.isoformat().startswith("2026-10-04")


def test_ft_article_at_either_article_level_period_bound_is_retained():
    """`_filter_by_period` (the article-eligibility gate used by both
    normalize()'s wide history window and agent_analysis()'s narrower
    digest-period narrowing) is a CLOSED interval, inclusive at both ends --
    this is unchanged by the FT search feature. The strict HALF-OPEN
    `[period_start, period_end)` rule is enforced one stage later, at
    EVENT-level digest qualification (see the next two tests) -- not here."""
    from research_assistant.pipeline.nodes import _filter_by_period
    from research_assistant.retrieval.models import ExtractedArticle, ExtractionStatus
    from research_assistant.evidence.normalizer import normalize_article
    from tests.pipeline.conftest import jpmorgan_source

    period_start = datetime(2026, 10, 3, 8, 59, 49, tzinfo=timezone.utc)
    period_end = datetime(2026, 10, 4, 8, 59, 49, tzinfo=timezone.utc)
    source = jpmorgan_source()
    article_at_start = normalize_article(
        ExtractedArticle(url="https://www.ft.com/content/a", published_at=period_start, status=ExtractionStatus.SUCCESS, text="x" * 50),
        source=source,
    )
    article_at_end = normalize_article(
        ExtractedArticle(url="https://www.ft.com/content/b", published_at=period_end, status=ExtractionStatus.SUCCESS, text="x" * 50),
        source=source,
    )
    in_period, _ = _filter_by_period([article_at_start, article_at_end], period_start, period_end)
    assert article_at_start in in_period
    assert article_at_end in in_period


## Note: the actual half-open `[period_start, period_end)` EVENT-level
## digest-qualification rule (item 10 of the task's test list) is already
## covered generically, for any source, by the pre-existing
## tests/pipeline/test_digest_boundary_e2e.py -- unchanged by this feature,
## so not re-tested here to avoid a redundant/riskier re-implementation.


def test_ft_search_discovered_article_still_routes_to_ai_technology_agent():
    """Source-aware-routing refactor (config/sources.yaml's Financial Times
    entry now sets `research_domain: ai_technology`): an FT article -- found
    via the interactive-search path this file otherwise tests, or via plain
    hub-page discovery, the routing decision is identical either way --
    routes to the AI/Technology specialist (`AgentName.AI_PRODUCT`), not the
    generic `AgentName.FINANCIAL_NEWS`, so FT's own AI/technology coverage
    reaches the specialist agent whose prompt is actually tuned to judge it
    (see agents.router.ArticleRouter's research_domain-aware case)."""
    from research_assistant.agents.router import route_articles
    from research_assistant.evidence.normalizer import normalize_article
    from research_assistant.retrieval.models import ExtractedArticle, ExtractionStatus

    config = get_config()
    ft_def = next(d for d in build_source_definitions(config) if d.name == "Financial Times")
    extracted = ExtractedArticle(
        url="https://www.ft.com/content/9b3a355e-5975-445f-9004-b95513e3856a",
        title="Masayoshi Son's AI ambitions outgrow SoftBank's balance sheet",
        text="SoftBank's AI investment strategy is under scrutiny as its balance sheet absorbs new AI bets. " * 5,
        status=ExtractionStatus.SUCCESS,
    )
    article = normalize_article(extracted, source=ft_def)
    routing = route_articles([article], config)
    assert AgentName.AI_PRODUCT in routing.by_agent
    assert article.id in {a.id for a in routing.by_agent[AgentName.AI_PRODUCT]}
    assert article.id not in {a.id for a in routing.by_agent.get(AgentName.FINANCIAL_NEWS, [])}
