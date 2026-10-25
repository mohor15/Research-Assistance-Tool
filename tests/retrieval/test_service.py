"""End-to-end: source -> discovery -> candidate -> retrieval -> extraction.

Uses the real Phase 0 config (`get_config()`) wired to a scripted
`HTTPClient`/`FakeSession` so no test here touches the network, and verifies
the composition in `RetrievalService`, including robots.txt enforcement and
browser-fallback gating.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest
import requests

import research_assistant.retrieval.service as service_module
from research_assistant.config import get_config
from research_assistant.retrieval.cache import PageCache
from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.exceptions import SourceUnreachableError
from research_assistant.retrieval.models import (
    CandidateURL,
    DiscoveryMethod,
    ExtractionStatus,
    RetrievalFailure,
    RetrievalMethod,
    RetrievedPage,
    SourceCategory,
)
from research_assistant.retrieval.rate_limiter import DomainRateLimiter
from research_assistant.retrieval.robots import RobotsChecker
from research_assistant.retrieval.service import RetrievalService

from tests.retrieval.fakes import FakeResponse, FakeSession

NEWSROOM_HTML = """
<html><head><meta property="og:site_name" content="JPMorgan Chase"/></head>
<body>
<main>
  <a href="/newsroom/ai-strategy-update">JPMorgan outlines AI strategy update</a>
</main>
</body></html>
"""

ARTICLE_HTML = """
<html><head>
<meta property="og:title" content="JPMorgan outlines AI strategy update"/>
<meta property="article:published_time" content="2025-09-10T12:00:00Z"/>
<meta property="og:site_name" content="JPMorgan Chase"/>
</head><body><article>
<p>JPMorgan Chase today outlined an update to its enterprise AI strategy,
describing new internal tooling for research analysts and expanded use of
large language models across risk and compliance workflows.</p>
<p>The bank said the initiative builds on previous investments in AI
infrastructure and will be rolled out gradually across its global offices
over the coming quarters, with a focus on measurable productivity gains.</p>
</article></body></html>
"""


def _no_robots_checker() -> RobotsChecker:
    checker = RobotsChecker(enabled=False, user_agent="TestBot")
    return checker


def _build_service(session: FakeSession, *, browser_fallback: bool = False, tmp_path) -> RetrievalService:
    config = get_config()
    config = config.model_copy(deep=True)
    config.settings.retrieval.browser_fallback = browser_fallback

    http_client = HTTPClient(
        timeout_seconds=5,
        max_retries=0,
        retry_delay_seconds=0,
        session=session,
        robots_checker=_no_robots_checker(),
        cache=PageCache(cache_dir=tmp_path, enabled=False),
    )
    return RetrievalService(config, http_client=http_client, robots_checker=_no_robots_checker())


def test_retrieve_source_end_to_end_produces_extracted_articles(tmp_path):
    session = FakeSession([
        FakeResponse(status_code=200, url="https://www.jpmorganchase.com/newsroom", text=NEWSROOM_HTML),
        FakeResponse(
            status_code=200,
            url="https://www.jpmorganchase.com/newsroom/ai-strategy-update",
            text=ARTICLE_HTML,
        ),
    ])
    service = _build_service(session, tmp_path=tmp_path)

    jpmorgan = next(s for s in service.list_sources(SourceCategory.PEER) if s.name == "JPMorgan Chase")
    articles = service.retrieve_source(jpmorgan)

    assert len(articles) == 1
    article = articles[0]
    assert article.status == ExtractionStatus.SUCCESS
    assert article.title == "JPMorgan outlines AI strategy update"
    assert "enterprise AI strategy" in article.text
    assert article.publisher  # backfilled from source name if extraction found none


def test_retrieve_source_rejects_a_pdf_asset_instead_of_passing_binary_garbage_through(tmp_path):
    """Regression for the NVIDIA gallery-file bug (Phase 10): a discovered
    candidate that resolves to a PDF asset must be rejected by the
    Content-Type guard in the extractor, end-to-end through the real
    RetrievalService -- never silently turned into "extracted" evidence."""
    session = FakeSession([
        FakeResponse(status_code=200, url="https://www.jpmorganchase.com/newsroom", text=NEWSROOM_HTML),
        FakeResponse(
            status_code=200,
            url="https://www.jpmorganchase.com/newsroom/ai-strategy-update",
            text="%PDF-1.7\n%\xe2\xe3\xcf\xd3\nbinary pdf bytes here",
            headers={"Content-Type": "application/pdf"},
        ),
    ])
    service = _build_service(session, tmp_path=tmp_path)

    jpmorgan = next(s for s in service.list_sources(SourceCategory.PEER) if s.name == "JPMorgan Chase")
    articles = service.retrieve_source(jpmorgan)

    assert len(articles) == 1
    assert articles[0].status == ExtractionStatus.FAILED
    assert articles[0].metadata["reason"] == "unsupported_content_type"


def test_discover_candidates_rescues_a_late_on_topic_link_using_real_topic_vocabulary(tmp_path):
    """Phase 13D regression (the real BCG finding, reproduced generically):
    `RetrievalService.discover_candidates` must thread the project's real
    `config/topics.yaml` vocabulary through to the discovery-time ranking,
    so a hub page listing more links than `max_candidates_per_source` does
    not silently lose a genuinely on-topic publication that happens to
    appear late in the page's raw link order."""
    off_topic_links = "\n".join(
        f'<a href="/industries/practice-{i}">Industry practice page {i}</a>' for i in range(60)
    )
    hub_html = f"""
    <html><body><main>
      {off_topic_links}
      <a href="/publications/the-formula-for-agentic-ai-value">The Formula for Agentic AI Value</a>
    </main></body></html>
    """
    session = FakeSession([FakeResponse(status_code=200, url="https://www.jpmorganchase.com/newsroom", text=hub_html)])
    service = _build_service(session, tmp_path=tmp_path)

    jpmorgan = next(s for s in service.list_sources(SourceCategory.PEER) if s.name == "JPMorgan Chase")
    candidates = service.discover_candidates(jpmorgan)

    assert any("agentic-ai-value" in c.url for c in candidates)


def test_list_sources_covers_all_scrapable_categories():
    config = get_config()
    service = RetrievalService(config, http_client=HTTPClient(timeout_seconds=1, max_retries=0, retry_delay_seconds=0, session=FakeSession([])))

    categories = {s.category for s in service.list_sources()}
    assert categories == {
        SourceCategory.PEER,
        SourceCategory.AI_PROVIDER,
        SourceCategory.ANALYST,
        SourceCategory.SECONDARY,
    }


def test_robots_disallowed_page_is_never_retrieved(tmp_path):
    config = get_config()
    config = config.model_copy(deep=True)

    session = FakeSession([])  # if robots blocks correctly, no HTTP call should ever be attempted
    robots = RobotsChecker(enabled=True, user_agent="TestBot")

    class _AlwaysDisallow:
        def is_allowed(self, url):
            return False

    http_client = HTTPClient(
        timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session, robots_checker=_AlwaysDisallow()
    )
    service = RetrievalService(config, http_client=http_client, robots_checker=_AlwaysDisallow())

    source = next(s for s in service.list_sources(SourceCategory.PEER) if s.name == "JPMorgan Chase")

    with pytest.raises(SourceUnreachableError, match="robots_disallowed"):
        service.discover_candidates(source)

    assert session.calls == []  # discovery itself never got HTML because the source page was blocked -- robots.txt was never bypassed


def test_browser_fallback_is_not_invoked_when_disabled(monkeypatch, tmp_path):
    called = {"invoked": False}

    def _fake_fetch_with_browser(*a, **k):
        called["invoked"] = True
        raise AssertionError("browser fallback should not be invoked when browser_fallback=false")

    monkeypatch.setattr(service_module, "fetch_with_browser", _fake_fetch_with_browser)
    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)

    session = FakeSession([FakeResponse(status_code=500, url="https://example.com/fails", text="")])
    service = _build_service(session, browser_fallback=False, tmp_path=tmp_path)

    from research_assistant.retrieval.models import CandidateURL, DiscoveryMethod
    from datetime import datetime, timezone

    candidate = CandidateURL(
        url="https://example.com/fails",
        source_name="Test",
        domain="example.com",
        discovery_method=DiscoveryMethod.HTML_LINKS,
        discovered_at=datetime.now(timezone.utc),
    )
    page = service.retrieve(candidate)

    assert called["invoked"] is False
    assert page.success is False


def test_browser_fallback_is_skipped_for_a_definitive_404(monkeypatch, tmp_path):
    """A page that returns a genuine 404 cannot be recovered by rendering --
    the resource doesn't exist regardless of client. Bounds the cost of a
    bad/stale candidate: no 20s browser render is wasted on it."""
    called = {"invoked": False}

    def _fake_fetch_with_browser(*a, **k):
        called["invoked"] = True
        raise AssertionError("browser fallback should not be attempted for a definitive 404")

    monkeypatch.setattr(service_module, "fetch_with_browser", _fake_fetch_with_browser)
    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)

    session = FakeSession([FakeResponse(status_code=404, url="https://example.com/gone", text="")])
    service = _build_service(session, browser_fallback=True, tmp_path=tmp_path)

    candidate = CandidateURL(
        url="https://example.com/gone", source_name="Test", domain="example.com",
        discovery_method=DiscoveryMethod.HTML_LINKS, discovered_at=datetime.now(timezone.utc),
    )
    page = service.retrieve(candidate)

    assert called["invoked"] is False
    assert page.success is False
    assert page.failure.reason == "http_error"


def test_browser_fallback_still_works_for_a_legitimate_js_heavy_research_page(monkeypatch, tmp_path):
    """Confirms the 404-exclusion above is narrow: a timeout on an otherwise
    legitimate research URL still gets a browser-rendering attempt."""
    rendered_page = RetrievedPage(
        url="https://example.com/news/ai-launch",
        final_url="https://example.com/news/ai-launch",
        status_code=200,
        retrieved_at=datetime.now(timezone.utc),
        method=RetrievalMethod.BROWSER,
        html="<html>rendered</html>",
        success=True,
    )

    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)
    monkeypatch.setattr(service_module, "fetch_with_browser", lambda *a, **k: rendered_page)

    session = FakeSession([requests.Timeout("slow")])
    service = _build_service(session, browser_fallback=True, tmp_path=tmp_path)

    candidate = CandidateURL(
        url="https://example.com/news/ai-launch", source_name="Test", domain="example.com",
        discovery_method=DiscoveryMethod.HTML_LINKS, discovered_at=datetime.now(timezone.utc),
    )
    page = service.retrieve(candidate)

    assert page.method == RetrievalMethod.BROWSER
    assert page.success is True


# ---------------------------------------------------------------------------
# Browser-rendered DISCOVERY fallback (Phase 1 fix for Lloyds Banking Group):
# a source's discovery page that returns 200 OK but yields ZERO candidate
# links (a client-side-rendered shell) gets one bounded browser-rendering
# retry -- never triggered merely because a page returned fewer candidates
# than hoped, and never for a page that already found something.
# ---------------------------------------------------------------------------

_THIN_SHELL_HTML = '<html><body><a href="/careers">Careers</a></body></html>'  # mirrors the real Lloyds page
_RENDERED_NEWSROOM_HTML = """
<html><body><main>
  <a href="/media/press-releases/2026/lloyds-ai-initiative">Lloyds launches AI initiative</a>
</main></body></html>
"""


def _source_with_one_url(url: str, *, domain: Optional[str] = None) -> "object":
    from urllib.parse import urlparse

    from research_assistant.models.enums import SourcePriority, SourceType
    from research_assistant.retrieval.models import SourceDefinition

    return SourceDefinition(
        name="Lloyds Banking Group", domain=domain or urlparse(url).netloc, urls=[url],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )


def test_discovery_falls_back_to_browser_when_the_page_yields_zero_candidates(monkeypatch, tmp_path):
    url = "https://www.lloydsbankinggroup.com/media/press-releases.html"
    rendered_page = RetrievedPage(
        url=url, final_url=url, status_code=200, retrieved_at=datetime.now(timezone.utc),
        method=RetrievalMethod.BROWSER, html=_RENDERED_NEWSROOM_HTML, success=True,
    )
    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)
    monkeypatch.setattr(service_module, "fetch_with_browser", lambda *a, **k: rendered_page)

    session = FakeSession([FakeResponse(status_code=200, url=url, text=_THIN_SHELL_HTML)])
    service = _build_service(session, browser_fallback=True, tmp_path=tmp_path)

    candidates = service.discover_candidates(_source_with_one_url(url))

    assert len(candidates) == 1
    assert "lloyds-ai-initiative" in candidates[0].url


def test_discovery_browser_fallback_does_not_trigger_when_candidates_were_already_found(monkeypatch, tmp_path):
    """Must never fire merely because a page is HTTP 200 -- only when
    discovery found literally nothing."""
    url = "https://example.com/newsroom"
    called = {"invoked": False}

    def _fake_fetch_with_browser(*a, **k):
        called["invoked"] = True
        raise AssertionError("browser-rendered discovery should not be attempted when candidates already exist")

    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)
    monkeypatch.setattr(service_module, "fetch_with_browser", _fake_fetch_with_browser)

    session = FakeSession([FakeResponse(status_code=200, url=url, text=NEWSROOM_HTML)])
    service = _build_service(session, browser_fallback=True, tmp_path=tmp_path)

    candidates = service.discover_candidates(_source_with_one_url(url))

    assert called["invoked"] is False
    assert len(candidates) == 1


_YAHOO_NAV_ONLY_HTML = """
<html><body><main>
  <a href="/topic/tech/">Tech</a>
  <a href="/topic/artificial-intelligence/">Artificial Intelligence</a>
  <a href="/markets/stocks/">Stocks</a>
</main></body></html>
"""

_YAHOO_RENDERED_HTML = """
<html><body><main>
  <a href="/topic/tech/">Tech</a>
  <a href="/markets/stocks/articles/tsmc-posts-record-profit-ai-chips-104536579.html">TSMC posts record profit amid AI chip demand</a>
</main></body></html>
"""


def test_yahoo_finance_discovery_is_forced_through_the_browser_even_when_nav_candidates_exist(monkeypatch, tmp_path):
    """Confirmed real regression: Yahoo Finance's plain HTML is a React nav
    shell that always yields SOME "candidates" (pure navigation, never a
    real article), so the existing "zero candidates" browser-fallback
    trigger never fires. `_YAHOO_FINANCE_DOMAIN` must force the browser path
    unconditionally for this one domain and use ITS results, not the nav
    noise from the plain-HTTP pass."""
    url = "https://finance.yahoo.com/topic/artificial-intelligence/"
    rendered_page = RetrievedPage(
        url=url, final_url=url, status_code=200, retrieved_at=datetime.now(timezone.utc),
        method=RetrievalMethod.BROWSER, html=_YAHOO_RENDERED_HTML, success=True,
    )
    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)
    monkeypatch.setattr(service_module, "fetch_with_browser", lambda *a, **k: rendered_page)

    session = FakeSession([FakeResponse(status_code=200, url=url, text=_YAHOO_NAV_ONLY_HTML)])
    service = _build_service(session, browser_fallback=True, tmp_path=tmp_path)
    candidates = service.discover_candidates(_source_with_one_url(url))

    urls = {c.url for c in candidates}
    assert "https://finance.yahoo.com/markets/stocks/articles/tsmc-posts-record-profit-ai-chips-104536579.html" in urls
    assert "https://finance.yahoo.com/topic/artificial-intelligence/" not in urls  # nav noise discarded


def test_non_yahoo_source_is_unaffected_by_the_yahoo_browser_override(monkeypatch, tmp_path):
    """The forced-browser behavior is scoped to finance.yahoo.com only --
    a different domain with the exact same nav-only HTML must still rely on
    the existing zero-candidates fallback semantics unchanged."""
    url = "https://example.com/newsroom"
    called = {"invoked": False}

    def _fake_fetch_with_browser(*a, **k):
        called["invoked"] = True
        raise AssertionError("browser-rendered discovery should not be attempted for a non-Yahoo source with existing candidates")

    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)
    monkeypatch.setattr(service_module, "fetch_with_browser", _fake_fetch_with_browser)

    session = FakeSession([FakeResponse(status_code=200, url=url, text=_YAHOO_NAV_ONLY_HTML)])
    service = _build_service(session, browser_fallback=True, tmp_path=tmp_path)

    candidates = service.discover_candidates(_source_with_one_url(url))

    assert called["invoked"] is False
    assert len(candidates) >= 1


def test_discovery_browser_fallback_is_not_invoked_when_disabled(monkeypatch, tmp_path):
    url = "https://www.lloydsbankinggroup.com/media/press-releases.html"
    called = {"invoked": False}

    def _fake_fetch_with_browser(*a, **k):
        called["invoked"] = True
        raise AssertionError("should not be invoked when browser_fallback=false")

    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)
    monkeypatch.setattr(service_module, "fetch_with_browser", _fake_fetch_with_browser)

    session = FakeSession([FakeResponse(status_code=200, url=url, text=_THIN_SHELL_HTML)])
    service = _build_service(session, browser_fallback=False, tmp_path=tmp_path)

    candidates = service.discover_candidates(_source_with_one_url(url))

    assert called["invoked"] is False
    assert candidates == []


def test_browser_rendered_discovery_never_bypasses_robots_txt(monkeypatch, tmp_path):
    """Unit-level, isolated from the "how did we get here" question: the
    browser-rendered discovery retry re-checks robots.txt for the exact
    same URL before rendering it, exactly like the per-candidate browser
    fallback already does -- it must never render a disallowed page."""
    url = "https://www.lloydsbankinggroup.com/media/press-releases.html"
    called = {"invoked": False}

    def _fake_fetch_with_browser(*a, **k):
        called["invoked"] = True
        raise AssertionError("must never render a robots-disallowed page")

    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)
    monkeypatch.setattr(service_module, "fetch_with_browser", _fake_fetch_with_browser)

    config = get_config()
    config = config.model_copy(deep=True)
    config.settings.retrieval.browser_fallback = True

    class _AlwaysDisallow:
        def is_allowed(self, url):
            return False

    service = RetrievalService(
        config, http_client=HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=FakeSession([])),
        robots_checker=_AlwaysDisallow(),
    )

    candidates, failures = service._try_browser_rendered_discovery(_source_with_one_url(url), max_candidates=50)

    assert called["invoked"] is False
    assert candidates == []
    assert failures == []


def test_discovery_recovers_via_browser_when_http_fetch_raises_source_unreachable(monkeypatch, tmp_path):
    """Bug fix (Part A2): previously SourceUnreachableError propagated
    straight out of discover_candidates() and the browser-fallback branch
    was never reached at all -- a source whose plain HTTP fetch failed
    outright (TLS error, timeout, connection reset, ...) never got a
    browser retry, even with browser_fallback=true. Mirrors the real
    Reuters/Bloomberg-class failure: the HTTP client never gets a response
    at all, not merely zero candidates."""
    url = "https://example.com/newsroom"
    rendered_page = RetrievedPage(
        url=url, final_url=url, status_code=200, retrieved_at=datetime.now(timezone.utc),
        method=RetrievalMethod.BROWSER, html=_RENDERED_NEWSROOM_HTML, success=True,
    )
    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)
    monkeypatch.setattr(service_module, "fetch_with_browser", lambda *a, **k: rendered_page)

    session = FakeSession([requests.ConnectionError("dns failure")])
    service = _build_service(session, browser_fallback=True, tmp_path=tmp_path)

    candidates = service.discover_candidates(_source_with_one_url(url))

    assert len(candidates) == 1
    assert "lloyds-ai-initiative" in candidates[0].url


def test_discovery_re_raises_source_unreachable_when_browser_fallback_also_fails(monkeypatch, tmp_path):
    """The re-raise path of the Part A2 fix: when the browser fallback is
    attempted but ALSO recovers nothing, the original SourceUnreachableError
    still propagates (so pipeline.nodes.retrieve()'s existing diagnostic
    behavior for a genuinely-unreachable source is unchanged) -- it must
    never be silently swallowed into an empty candidate list."""
    url = "https://example.com/newsroom"

    monkeypatch.setattr(service_module, "is_browser_available", lambda: False)

    session = FakeSession([requests.ConnectionError("dns failure")])
    service = _build_service(session, browser_fallback=True, tmp_path=tmp_path)

    with pytest.raises(SourceUnreachableError, match="connection_error"):
        service.discover_candidates(_source_with_one_url(url))


def test_browser_fallback_is_invoked_when_http_fails_and_enabled(monkeypatch, tmp_path):
    from datetime import datetime, timezone
    from research_assistant.retrieval.models import CandidateURL, DiscoveryMethod

    rendered_page = RetrievedPage(
        url="https://example.com/needs-js",
        final_url="https://example.com/needs-js",
        status_code=200,
        retrieved_at=datetime.now(timezone.utc),
        method=RetrievalMethod.BROWSER,
        html="<html>rendered</html>",
        success=True,
    )

    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)
    monkeypatch.setattr(service_module, "fetch_with_browser", lambda *a, **k: rendered_page)

    session = FakeSession([FakeResponse(status_code=500, url="https://example.com/needs-js", text="")])
    service = _build_service(session, browser_fallback=True, tmp_path=tmp_path)

    candidate = CandidateURL(
        url="https://example.com/needs-js",
        source_name="Test",
        domain="example.com",
        discovery_method=DiscoveryMethod.HTML_LINKS,
        discovered_at=datetime.now(timezone.utc),
    )
    page = service.retrieve(candidate)

    assert page.method == RetrievalMethod.BROWSER
    assert page.success is True


def test_matching_401_from_http_and_browser_is_reclassified_as_anti_bot_access_blocked(monkeypatch, tmp_path):
    """Confirms the Reuters/Barron's-class case: the HTTP client AND a real
    rendered browser independently got the same 401 for this exact URL --
    that is a confirmed access restriction, not a transient failure, and
    must be reported as such rather than left as a bare "http_error"."""
    url = "https://example.com/blocked"
    browser_page = RetrievedPage(
        url=url, final_url=url, status_code=401, retrieved_at=datetime.now(timezone.utc),
        method=RetrievalMethod.BROWSER, html="<html>blocked</html>", success=False,
        failure=RetrievalFailure(reason="http_error", detail="401"),
    )
    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)
    monkeypatch.setattr(service_module, "fetch_with_browser", lambda *a, **k: browser_page)

    session = FakeSession([FakeResponse(status_code=401, url=url, text="")])
    service = _build_service(session, browser_fallback=True, tmp_path=tmp_path)

    candidate = CandidateURL(
        url=url, source_name="Test", domain="example.com",
        discovery_method=DiscoveryMethod.HTML_LINKS, discovered_at=datetime.now(timezone.utc),
    )
    page = service.retrieve(candidate)

    assert page.success is False
    assert page.failure.reason == "anti_bot_access_blocked"


def test_a_401_with_no_browser_corroboration_is_not_reclassified_as_anti_bot(monkeypatch, tmp_path):
    """Narrow: if the browser fallback isn't even attempted (e.g. Playwright
    unavailable), a bare HTTP 401 stays a bare http_error -- it's never
    *assumed* to be anti-bot just because it's a 401."""
    url = "https://example.com/blocked"
    monkeypatch.setattr(service_module, "is_browser_available", lambda: False)

    session = FakeSession([FakeResponse(status_code=401, url=url, text="")])
    service = _build_service(session, browser_fallback=True, tmp_path=tmp_path)

    candidate = CandidateURL(
        url=url, source_name="Test", domain="example.com",
        discovery_method=DiscoveryMethod.HTML_LINKS, discovered_at=datetime.now(timezone.utc),
    )
    page = service.retrieve(candidate)

    assert page.success is False
    assert page.failure.reason == "http_error"


def test_browser_fallback_is_skipped_for_an_rss_url(monkeypatch, tmp_path):
    """Phase 9 requirement: a feed URL that fails to fetch must never
    trigger a browser render attempt -- rendering XML in a JS-capable
    browser cannot produce usable article HTML."""
    called = {"invoked": False}

    def _fake_fetch_with_browser(*a, **k):
        called["invoked"] = True
        raise AssertionError("browser fallback should not be attempted for an RSS/XML URL")

    monkeypatch.setattr(service_module, "fetch_with_browser", _fake_fetch_with_browser)
    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)

    url = "https://example.com/feed.xml"
    session = FakeSession([requests.ConnectionError("dns failure")])
    service = _build_service(session, browser_fallback=True, tmp_path=tmp_path)

    candidate = CandidateURL(
        url=url, source_name="Test", domain="example.com",
        discovery_method=DiscoveryMethod.HTML_LINKS, discovered_at=datetime.now(timezone.utc),
    )
    page = service.retrieve(candidate)

    assert called["invoked"] is False
    assert page.success is False


def test_browser_fallback_is_skipped_for_a_pdf_url(monkeypatch, tmp_path):
    called = {"invoked": False}

    def _fake_fetch_with_browser(*a, **k):
        called["invoked"] = True
        raise AssertionError("browser fallback should not be attempted for a PDF URL")

    monkeypatch.setattr(service_module, "fetch_with_browser", _fake_fetch_with_browser)
    monkeypatch.setattr(service_module, "is_browser_available", lambda: True)

    url = "https://example.com/report.pdf"
    session = FakeSession([requests.ConnectionError("dns failure")])
    service = _build_service(session, browser_fallback=True, tmp_path=tmp_path)

    candidate = CandidateURL(
        url=url, source_name="Test", domain="example.com",
        discovery_method=DiscoveryMethod.HTML_LINKS, discovered_at=datetime.now(timezone.utc),
    )
    page = service.retrieve(candidate)

    assert called["invoked"] is False
    assert page.success is False


def test_feed_summary_backfills_text_when_the_article_page_itself_is_blocked(tmp_path):
    """Part A3/WSJ case: the article page 401s (confirmed anti-bot/paywall
    block, never bypassed), but the source's own RSS feed already publicly
    syndicated a summary for this exact URL -- using that summary as
    degraded evidence is legitimate (it's not fetched from the blocked
    page) and still has to clear the normal evidence gates downstream."""
    from research_assistant.retrieval.models import ExtractedArticle

    candidate = CandidateURL(
        url="https://example.com/articles/a", source_name="WSJ", domain="wsj.com",
        discovery_method=DiscoveryMethod.RSS, discovered_at=datetime.now(timezone.utc),
        title="Feed title", feed_summary="A short feed-provided summary of the article.",
    )
    extracted = ExtractedArticle(url=candidate.url, status=ExtractionStatus.FAILED, metadata={"reason": "http_error"})
    source = _source_with_one_url("https://example.com/articles/a", domain="wsj.com")

    backfilled = service_module._backfill_from_candidate(extracted, candidate, source)

    assert backfilled.text == "A short feed-provided summary of the article."
    assert backfilled.status == ExtractionStatus.PARTIAL
    assert backfilled.metadata["text_source"] == "feed_summary"


def test_discover_candidates_propagates_cache_policy_to_the_http_client(monkeypatch, tmp_path):
    """RetrievalService.discover_candidates/retrieve must actually thread
    `cache_policy` down to HTTPClient.get(), not just accept the parameter
    -- a stale cached hub page must be bypassed end-to-end through the real
    service, not only at the bare HTTPClient layer."""
    from research_assistant.retrieval.cache import PageCache
    from research_assistant.retrieval.models import CachePolicy, RetrievalMethod

    url = "https://example.com/newsroom"
    source = _source_with_one_url(url)

    cache = PageCache(cache_dir=tmp_path, enabled=True)
    cache.set(url, RetrievedPage(
        url=url, final_url=url, status_code=200, retrieved_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
        method=RetrievalMethod.HTTP, html=_THIN_SHELL_HTML, success=True,
    ))

    session = FakeSession([FakeResponse(status_code=200, url=url, text=NEWSROOM_HTML)])
    http_client = HTTPClient(
        timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session,
        robots_checker=_no_robots_checker(), cache=cache,
    )
    config = get_config()
    config = config.model_copy(deep=True)
    service = RetrievalService(config, http_client=http_client, robots_checker=_no_robots_checker())

    candidates = service.discover_candidates(source, cache_policy=CachePolicy.FRESH)

    assert len(session.calls) == 1  # the stale cache entry was bypassed -- a live fetch happened
    assert any("ai-strategy-update" in c.url for c in candidates)  # content came from the FRESH page, not the stale shell


def test_feed_summary_never_overrides_a_successful_extraction():
    from research_assistant.retrieval.models import ExtractedArticle

    candidate = CandidateURL(
        url="https://example.com/articles/a", source_name="WSJ", domain="wsj.com",
        discovery_method=DiscoveryMethod.RSS, discovered_at=datetime.now(timezone.utc),
        feed_summary="A short feed-provided summary.",
    )
    extracted = ExtractedArticle(
        url=candidate.url, status=ExtractionStatus.SUCCESS, text="The real, fully extracted article body.",
    )
    source = _source_with_one_url("https://example.com/articles/a", domain="wsj.com")

    backfilled = service_module._backfill_from_candidate(extracted, candidate, source)

    assert backfilled.text == "The real, fully extracted article body."
    assert backfilled.status == ExtractionStatus.SUCCESS
