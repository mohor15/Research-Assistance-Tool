"""PR Newswire discovery fix (forensic finding, run `8704470a4a623d72`):
PR Newswire's entire site lives under "/news-releases/...", so the generic
keep-keyword filter in `discovery.py` matches BOTH a real individual
release and every one of its category/topic LISTING pages -- a live run
discovered 27 category pages and zero actual releases because of this.

This file proves the domain-scoped `_is_pr_newswire_release_url` filter
(release URLs end in "-<6+ digits>.html"; category/topic pages always end
in "/") correctly separates the two, using a listing-page fixture modeled
on the real cached PR Newswire AI topic page, plus the real JumpCloud
article (run `8704470a4a623d72`'s forensic subject) as the regression case.

No live Ollama/network anywhere -- FakeSession/FakeResponse only. The date-
extraction fixture below is a trimmed copy of the real cached JSON-LD this
article actually served (verified against `data/cache` during the forensic
investigation), not an invented one.
"""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.config import get_config
from research_assistant.evidence.normalizer import normalize_article
from research_assistant.models.article import DatePrecision
from research_assistant.models.enums import AgentName, SourcePriority, SourceType
from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.discovery import discover_candidates
from research_assistant.retrieval.extractor import extract_article
from research_assistant.retrieval.models import DiscoveryMethod, SourceCategory, SourceDefinition

from tests.retrieval.fakes import FakeResponse, FakeSession

JUMPCLOUD_URL = (
    "https://www.prnewswire.com/news-releases/"
    "new-jumpcloud-research-finds-ai-access-is-outpacing-enterprise-governance-302901426.html"
)
JUMPCLOUD_TITLE = "New JumpCloud Research Finds AI Access Is Outpacing Enterprise Governance"

# Modeled on the real cached PR Newswire AI topic page: a handful of real-
# shaped individual releases (including JumpCloud) mixed with real-shaped
# category/topic listing links (taken verbatim from that page's own hrefs).
AI_TOPIC_PAGE_HTML = f"""
<html><body><main>
  <a href="{JUMPCLOUD_URL}">{JUMPCLOUD_TITLE}</a>
  <a href="/news-releases/apollo-graphql-introduces-graphos-agent-services-giving-ai-agents-governed-access-to-enterprise-apis-and-systems-302900890.html">Apollo GraphQL Introduces GraphOS Agent Services</a>
  <a href="/news-releases/89-of-guests-want-ai-working-behind-the-scenes-not-running-the-front-desk-mews-finds-302901379.html">89% of Guests Want AI Working Behind the Scenes</a>
  <a href="/news-releases/latest-news-topics/artificial-intelligence/">All Artificial Intelligence News and Press Releases from PR Newswire</a>
  <a href="/news-releases/financial-services-latest-news/venture-capital-list/">All Venture Capital News and Press Releases from PR Newswire</a>
  <a href="/news-releases/business-technology-latest-news/financial-technology-list/">All Financial Technology News and Press Releases from PR Newswire</a>
  <a href="/news-releases/all-public-company-news/">All Public Company News</a>
</main></body></html>
"""


def _pr_newswire_source(url: str = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/") -> SourceDefinition:
    return SourceDefinition(
        name="PR Newswire", domain="prnewswire.com", urls=[url],
        source_type=SourceType.PRESS_DISTRIBUTION, priority=SourcePriority.MEDIUM,
        category=SourceCategory.SECONDARY,
    )


def _discover(html: str, configured_url: str):
    session = FakeSession([FakeResponse(status_code=200, url=configured_url, text=html)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    source = _pr_newswire_source(configured_url)
    return discover_candidates(source, http_client=client, max_candidates=50)


# ---------------------------------------------------------------------------
# 1/2/3. The JumpCloud link is discovered; the topic page itself and every
# category/index link are rejected.
# ---------------------------------------------------------------------------


def test_jumpcloud_release_is_discovered_from_the_ai_topic_listing_page():
    configured_url = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"
    candidates = _discover(AI_TOPIC_PAGE_HTML, configured_url)

    urls = {c.url for c in candidates}
    assert JUMPCLOUD_URL in urls
    assert all(c.discovery_method == DiscoveryMethod.HTML_LINKS for c in candidates)


def test_topic_page_itself_never_becomes_a_candidate():
    configured_url = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"
    candidates = _discover(AI_TOPIC_PAGE_HTML, configured_url)

    urls = {c.url for c in candidates}
    assert configured_url not in urls
    assert configured_url.rstrip("/") not in {u.rstrip("/") for u in urls}


def test_category_and_index_links_are_rejected():
    configured_url = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"
    candidates = _discover(AI_TOPIC_PAGE_HTML, configured_url)

    urls = {c.url for c in candidates}
    rejected_fragments = [
        "/news-releases/financial-services-latest-news/venture-capital-list/",
        "/news-releases/business-technology-latest-news/financial-technology-list/",
        "/news-releases/all-public-company-news/",
    ]
    for fragment in rejected_fragments:
        assert not any(fragment in u for u in urls), f"{fragment} should have been rejected"


def test_only_release_shaped_links_survive_and_the_rest_are_category_pages():
    """Exactly the 3 release-shaped links (JumpCloud + the 2 others) survive
    out of the page's 7 total links -- the remaining 4 are all category/
    topic/index pages, confirming the filter is doing real work, not just
    passing everything through."""
    configured_url = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"
    candidates = _discover(AI_TOPIC_PAGE_HTML, configured_url)

    assert len(candidates) == 3
    for c in candidates:
        assert c.url.endswith(".html")
        slug_before_html = c.url[: -len(".html")]
        numeric_suffix = slug_before_html.rsplit("-", 1)[-1]
        assert numeric_suffix.isdigit() and len(numeric_suffix) >= 6


# ---------------------------------------------------------------------------
# 4/5. The individual release URL feeds the normal retrieval/normalization
# path unchanged, and existing date extraction still recovers the correct
# date -- this discovery fix touches nothing downstream of discovery.
# ---------------------------------------------------------------------------


# Trimmed, verbatim copy of the real JSON-LD this exact URL served (confirmed
# against the real cached response during the forensic investigation) --
# not an invented fixture.
_JUMPCLOUD_REAL_HTML = f"""<html><head>
<title>{JUMPCLOUD_TITLE}</title>
<script type="application/ld+json">
        {{
            "@context": "https://schema.org",
            "@type": "NewsArticle",
            "headline": "{JUMPCLOUD_TITLE}",
            "datePublished": "2026-10-07T13:04:00-04:00",
            "dateModified": "2026-10-07T13:05:23-04:00",
            "publisher": {{"@type": "Organization", "name": "Cision PR Newswire"}}
        }}
    </script>
</head>
<body><article>
<h1>{JUMPCLOUD_TITLE}</h1>
<p>LOUISVILLE, Colo., Oct. 7, 2026 /PRNewswire/ -- JumpCloud Inc. today released a new study, Enterprise AI Access Is Outpacing Control. The report shows that organizations are giving AI agents access to systems faster than they can secure them.</p>
</article></body></html>"""


def test_the_discovered_jumpcloud_url_still_extracts_the_correct_date():
    """Proves the discovery fix changes nothing about what happens once an
    individual release URL is actually fetched -- the existing,
    unmodified-by-THIS-fix extraction/normalization path still recovers
    `publication_date=2026-10-07`.

    Precision updated (forensic finding, CURRENT run `e818e08adc2e4dd1`,
    the direct follow-up to this file's own `8704470a4a623d72` finding):
    this fixture's JSON-LD `datePublished` ("2026-10-07T13:04:00-04:00")
    carries a real time-of-day component -- the DATE_ONLY value this test
    previously asserted was itself the confirmed precision-loss bug (see
    `extractor.py`'s date-precedence fix), not a correct outcome. The real
    page's true publish instant (17:04:00 UTC) must now be preserved as
    EXACT, which is exactly what let this run's six real PR Newswire
    releases wrongly fail the zero-slack 24h wide-lookback filter."""
    extracted = extract_article(_JUMPCLOUD_REAL_HTML, JUMPCLOUD_URL)
    article = normalize_article(extracted, source=_pr_newswire_source())

    assert article.publication_date is not None
    assert article.publication_date == datetime(2026, 10, 7, 17, 4, 0, tzinfo=timezone.utc)
    assert article.publication_date_precision == DatePrecision.EXACT


# ---------------------------------------------------------------------------
# 6. research_domain=startup_funding is unchanged by this fix.
# ---------------------------------------------------------------------------


def test_pr_newswire_research_domain_still_routes_to_startup_agent():
    config = get_config()
    entry = next(s for s in config.sources.secondary_sources if s.name == "PR Newswire")
    assert entry.research_domain == "startup_funding"

    from research_assistant.agents.router import ArticleRouter

    extracted = extract_article(_JUMPCLOUD_REAL_HTML, JUMPCLOUD_URL)
    article = normalize_article(extracted, source=_pr_newswire_source())
    agents = ArticleRouter(config).route(article)
    assert AgentName.STARTUP in agents
    assert AgentName.FINANCIAL_NEWS not in agents


# ---------------------------------------------------------------------------
# Generality: the filter is a URL-shape rule, not a JumpCloud-specific one.
# ---------------------------------------------------------------------------


def test_filter_discovers_other_releases_not_just_jumpcloud():
    """Several representative PR Newswire release URLs, none sharing any
    text with JumpCloud, prove the rule is general -- an arbitrary AI
    release with the right URL shape is discovered regardless of slug."""
    configured_url = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"
    html = """
    <html><body><main>
      <a href="/news-releases/example-company-launches-new-ai-platform-302912345.html">Example Company Launches New AI Platform</a>
      <a href="/news-releases/acme-ai-raises-series-b-funding-302912346.html">Acme AI Raises Series B Funding</a>
      <a href="/news-releases/hyperscale-data-sets-november-12-as-investor-day-to-discuss-michigan-ai-expansion-302897945.html">Hyperscale Data Sets Investor Day</a>
    </main></body></html>
    """
    candidates = _discover(html, configured_url)
    urls = {c.url for c in candidates}
    assert "https://www.prnewswire.com/news-releases/example-company-launches-new-ai-platform-302912345.html" in urls
    assert "https://www.prnewswire.com/news-releases/acme-ai-raises-series-b-funding-302912346.html" in urls
    assert "https://www.prnewswire.com/news-releases/hyperscale-data-sets-november-12-as-investor-day-to-discuss-michigan-ai-expansion-302897945.html" in urls
    assert len(candidates) == 3


def test_filter_rejects_category_pages_even_without_jumpcloud_present():
    """Negative fixture: a listing page with ONLY category/index links (no
    JumpCloud, no real release at all) must discover zero candidates --
    exactly reproducing the bug this fix addresses (27 category pages, 0
    real releases) in isolation."""
    configured_url = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"
    html = """
    <html><body><main>
      <a href="/news-releases/financial-services-latest-news/venture-capital-list/">All Venture Capital News and Press Releases from PR Newswire</a>
      <a href="/news-releases/automotive-transportation-latest-news/">All Automotive & Transportation News</a>
      <a href="/news-releases/business-technology-latest-news/semiconductors-list/">All Semiconductors News and Press Releases from PR Newswire</a>
      <a href="/news-releases/all-public-company-news/">All Public Company News</a>
    </main></body></html>
    """
    candidates = _discover(html, configured_url)
    assert candidates == []


def test_non_prnewswire_sources_are_unaffected_by_this_filter():
    """The domain-scoped filter must never apply to any other source --
    reuses the existing generic discovery test's own fixture/source shape."""
    from research_assistant.models.enums import SourcePriority as _SP
    from research_assistant.models.enums import SourceType as _ST

    other_source = SourceDefinition(
        name="Example Bank", domain="example.com",
        urls=["https://example.com/newsroom"],
        source_type=_ST.PRIMARY, priority=_SP.HIGH, category=SourceCategory.PEER,
    )
    html = """
    <html><body><main>
      <a href="/news/2025/09/ai-copilot-launch">Bank launches AI copilot</a>
    </main></body></html>
    """
    session = FakeSession([FakeResponse(status_code=200, url="https://example.com/newsroom", text=html)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    candidates = discover_candidates(other_source, http_client=client, max_candidates=50)

    assert len(candidates) == 1
    assert candidates[0].url == "https://example.com/news/2025/09/ai-copilot-launch"


# ---------------------------------------------------------------------------
# Pagination recall fix: a live check of the real configured page found it
# paginated (`pagesize=25`, `<link rel="next" href=".../?page=2">`), with
# page 2 holding 25 entirely distinct releases (zero overlap with page 1).
# Discovery previously only ever fetched page 1 -- missing the majority of
# same-day releases regardless of relevance/triage/Qwen behavior.
# ---------------------------------------------------------------------------

_PAGE_2_URL = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/?page=2"
_PAGE_3_URL = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/?page=3"

_PAGE_1_WITH_NEXT_LINK_HTML = f"""
<html><head>
<link rel="canonical" href="https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"/>
<link rel="next" href="{_PAGE_2_URL}"/>
</head><body><main>
  <a href="{JUMPCLOUD_URL}">{JUMPCLOUD_TITLE}</a>
</main></body></html>
"""

_PAGE_2_HTML = f"""
<html><head>
<link rel="next" href="{_PAGE_3_URL}"/>
</head><body><main>
  <a href="/news-releases/architect-financial-technologies-launches-liquid-inference-302900862.html">Architect Financial Technologies Launches Liquid Inference</a>
</main></body></html>
"""

_PAGE_3_URL_WITH_NEXT = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/?page=4"

_PAGE_3_HTML = f"""
<html><head>
<link rel="next" href="{_PAGE_3_URL_WITH_NEXT}"/>
</head><body><main>
  <a href="/news-releases/this-is-a-third-page-release-302900999.html">A Third Page Release</a>
</main></body></html>
"""


def test_pagination_follows_rel_next_and_discovers_page_2_releases():
    configured_url = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"
    session = FakeSession([
        FakeResponse(status_code=200, url=configured_url, text=_PAGE_1_WITH_NEXT_LINK_HTML),
        FakeResponse(status_code=200, url=_PAGE_2_URL, text=_PAGE_2_HTML),
        FakeResponse(status_code=200, url=_PAGE_3_URL, text=_PAGE_3_HTML),
    ])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    source = _pr_newswire_source(configured_url)

    candidates = discover_candidates(source, http_client=client, max_candidates=50)

    urls = {c.url for c in candidates}
    assert JUMPCLOUD_URL in urls  # page 1
    assert "https://www.prnewswire.com/news-releases/architect-financial-technologies-launches-liquid-inference-302900862.html" in urls  # page 2
    assert len(session.calls) == 3


def test_pagination_is_bounded_and_never_fetches_an_unbounded_number_of_pages():
    """`_PR_NEWSWIRE_MAX_PAGES` (3) caps traversal at 3 total pages -- a 4th
    page (which page 3's own HTML advertises via `rel="next"`) must never
    be fetched."""
    configured_url = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"
    session = FakeSession([
        FakeResponse(status_code=200, url=configured_url, text=_PAGE_1_WITH_NEXT_LINK_HTML),
        FakeResponse(status_code=200, url=_PAGE_2_URL, text=_PAGE_2_HTML),
        FakeResponse(status_code=200, url=_PAGE_3_URL, text=_PAGE_3_HTML),
    ])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    source = _pr_newswire_source(configured_url)

    candidates = discover_candidates(source, http_client=client, max_candidates=50)

    # Exactly 3 HTTP calls total (pages 1-3) -- page 4 is never fetched,
    # even though page 3's own HTML advertises a further `rel="next"`.
    assert len(session.calls) == 3
    urls = {c.url for c in candidates}
    assert "https://www.prnewswire.com/news-releases/this-is-a-third-page-release-302900999.html" in urls


def test_pagination_stops_cleanly_when_a_later_page_fails_to_fetch():
    configured_url = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"
    session = FakeSession([
        FakeResponse(status_code=200, url=configured_url, text=_PAGE_1_WITH_NEXT_LINK_HTML),
        FakeResponse(status_code=500, url=_PAGE_2_URL, text=""),
    ])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    source = _pr_newswire_source(configured_url)

    candidates = discover_candidates(source, http_client=client, max_candidates=50)

    urls = {c.url for c in candidates}
    assert JUMPCLOUD_URL in urls  # page 1 content still returned


def test_pagination_never_applies_to_non_prnewswire_sources():
    """A source that happens to also expose a `rel="next"` link must never
    have it followed -- this is scoped to `source.domain == "prnewswire.com"`
    only."""
    from research_assistant.models.enums import SourcePriority as _SP
    from research_assistant.models.enums import SourceType as _ST

    other_source = SourceDefinition(
        name="Example Bank", domain="example.com",
        urls=["https://example.com/newsroom"],
        source_type=_ST.PRIMARY, priority=_SP.HIGH, category=SourceCategory.PEER,
    )
    html = """
    <html><head><link rel="next" href="https://example.com/newsroom?page=2"/></head>
    <body><main><a href="/news/2025/09/ai-copilot-launch">Bank launches AI copilot</a></main></body></html>
    """
    session = FakeSession([FakeResponse(status_code=200, url="https://example.com/newsroom", text=html)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    candidates = discover_candidates(other_source, http_client=client, max_candidates=50)

    assert len(session.calls) == 1  # no second page ever fetched
    assert len(candidates) == 1


# ---------------------------------------------------------------------------
# Observability fix (forensic finding, CURRENT run `31fa192e4d7a7175`):
# `max_candidates_per_source` truncation was previously silent -- a source
# whose own page(s) listed MORE on-topic links than the cap discarded the
# weakest-ranked excess with nothing to show it happened, making "under-
# covered" and "cap genuinely too small" indistinguishable without a live
# re-fetch.
# ---------------------------------------------------------------------------


def test_truncation_beyond_max_candidates_is_logged_visibly(caplog):
    configured_url = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"
    links = "\n".join(
        f'<a href="/news-releases/example-ai-release-{i}-30291{i:04d}.html">Example AI Release {i}</a>'
        for i in range(5)
    )
    html = f"<html><body><main>{links}</main></body></html>"
    session = FakeSession([FakeResponse(status_code=200, url=configured_url, text=html)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    source = _pr_newswire_source(configured_url)

    import logging

    with caplog.at_level(logging.WARNING, logger="research_assistant.retrieval.discovery"):
        candidates = discover_candidates(source, http_client=client, max_candidates=3)

    assert len(candidates) == 3  # truncation still happens, unchanged
    assert any("truncating to max_candidates_per_source=3" in r.message for r in caplog.records)
    assert any("2 discarded this run" in r.message for r in caplog.records)


def test_no_truncation_warning_when_under_the_cap(caplog):
    configured_url = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"
    html = (
        '<html><body><main>'
        '<a href="/news-releases/example-ai-release-302910001.html">Example AI Release</a>'
        '</main></body></html>'
    )
    session = FakeSession([FakeResponse(status_code=200, url=configured_url, text=html)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    source = _pr_newswire_source(configured_url)

    import logging

    with caplog.at_level(logging.WARNING, logger="research_assistant.retrieval.discovery"):
        discover_candidates(source, http_client=client, max_candidates=50)

    assert not any("truncating to max_candidates_per_source" in r.message for r in caplog.records)
