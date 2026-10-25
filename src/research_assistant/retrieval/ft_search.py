"""Targeted interactive-search discovery for Financial Times ONLY.

FT's configured discovery URL (`https://www.ft.com/artificial-intelligence`)
is a hub/topic page: ranked and truncated like any other source's hub page
(see `discovery.discover_candidates`), it can miss relevant current articles
that exist on FT but never surface on that one page. This module adds an
ADDITIVE discovery path -- FT's own search feature -- never a replacement
for the existing hub-page discovery, and never used for any other source
(see `discovery.discover_candidates`'s `source.search` gate, which only
Financial Times configures).

Two tiers, in order, mirroring the project's existing HTTP-first-then-
browser pattern (`retrieval.service.RetrievalService._try_browser_fallback`):

1. Plain HTTP GET to FT's own `?q=` search endpoint -- confirmed via live
   inspection (2026-10-04) to be server-rendered (real `<a href="/content/
   ...">` result links are present in the raw HTML, no JS required). This
   is the normal case and never touches a browser.
2. Only if that yields nothing: a genuine Crawl4AI/Playwright-driven
   interaction with FT's own search UI (types into the search input,
   submits it, waits for results to render) -- for resilience if FT ever
   makes the plain query-string endpoint JS-only. Optional dependency,
   exactly like `retrieval.browser`'s Playwright fallback: if Crawl4AI/
   Playwright isn't installed, this tier simply contributes nothing.

Never treats the search page itself, or any FT tag/category/pagination
link, as an article -- see `_FT_ARTICLE_PATH_RE` and `_is_ft_article_url`.
Both tiers converge on the exact same candidate-parsing function, so the
"what counts as an article" rule is defined in exactly one place.

IMPORTANT, confirmed via live inspection (2026-10-04): FT's own
`robots.txt` disallows `/search` for `User-agent: *`. Both tiers above
correctly respect this (the HTTP tier via the existing `HTTPClient`/
`RobotsChecker`; the browser tier via an explicit check -- see
`_search_via_browser`), which means this module is currently a
policy-compliant no-op against the real site: it never fetches `/search`
at all. This is intentional and tested, not a bug -- this project never
bypasses robots.txt regardless of which fetch mechanism is used. The
module is left in place (rather than removed) so that if FT's robots.txt
policy ever changes, this starts working with no code change.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional, Set
from urllib.parse import quote, urljoin, urlparse

if TYPE_CHECKING:
    from research_assistant.retrieval.robots import RobotsChecker

from bs4 import BeautifulSoup

from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.models import CachePolicy, CandidateURL, DiscoveryMethod, SourceDefinition

logger = logging.getLogger(__name__)

FT_SEARCH_URL_TEMPLATE = "https://www.ft.com/search?q={query}"

# FT's own URL taxonomy reliably marks a real, individual article with a
# `/content/<uuid>` path (confirmed via extensive live inspection across
# dozens of FT pages) -- its tag/category pages (`/war-in-ukraine`,
# `/stream/<uuid>`), pagination links (`?page=N`), and other hub shapes
# never use this exact form. This is the ONLY signal used to tell a search
# RESULT (an article) apart from a search/hub/index page -- never anchor
# text, never position on the page.
_FT_ARTICLE_PATH_RE = re.compile(r"^/content/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig  # type: ignore

    CRAWL4AI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when crawl4ai is absent
    AsyncWebCrawler = None  # keeps the name patchable/importable even when crawl4ai isn't installed
    BrowserConfig = None
    CrawlerRunConfig = None
    CRAWL4AI_AVAILABLE = False


def is_crawl4ai_available() -> bool:
    return CRAWL4AI_AVAILABLE


def _is_ft_article_url(path: str) -> bool:
    return bool(_FT_ARTICLE_PATH_RE.match(path.split("?")[0]))


def _parse_ft_search_results_html(html: str, *, base_url: str, source: SourceDefinition, top_k: int) -> List[CandidateURL]:
    """Shared by both tiers (HTTP and browser) -- the exact same rule for
    "what is a real article result" applies no matter which one produced
    the HTML. Never assigns a publication date here: a search result's
    crawl/render time is not the article's publication date (see module
    docstring and `retrieval.service`'s existing per-candidate retrieval,
    which extracts the real date from the article page itself, unchanged)."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as exc:
        logger.warning("Could not parse FT search results HTML: %s", exc)
        return []

    now = datetime.now(timezone.utc)
    candidates: List[CandidateURL] = []
    seen: Set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href:
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if not _is_ft_article_url(parsed.path):
            continue  # a tag/category/pagination/other hub link -- never a candidate

        key = absolute.split("?")[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)

        title = anchor.get_text(strip=True) or None
        candidates.append(
            CandidateURL(
                url=absolute,
                source_name=source.name,
                domain=source.domain,
                discovery_method=DiscoveryMethod.SEARCH,
                discovered_at=now,
                title=title,
            )
        )
        if len(candidates) >= top_k:
            break

    return candidates


def _search_via_http(
    source: SourceDefinition, term: str, *, http_client: HTTPClient, cache_policy: CachePolicy, top_k: int
) -> List[CandidateURL]:
    url = FT_SEARCH_URL_TEMPLATE.format(query=quote(term))
    page = http_client.get(url, cache_policy=cache_policy)
    if not page.success or not page.html:
        if page.failure and page.failure.reason == "robots_disallowed":
            # Observable, not a mysterious zero-result search: FT's own
            # robots.txt currently disallows /search for User-agent: * --
            # confirmed live 2026-10-04. Never bypassed; this log line is
            # the explicit record that the block was detected and honored.
            logger.info("FT search term %r: robots blocked; skipped", term)
        else:
            logger.info("FT search via HTTP found nothing for %r (%s)", term, page.failure)
        return []
    return _parse_ft_search_results_html(page.html, base_url=page.final_url, source=source, top_k=top_k)


_FT_SEARCH_PAGE_URL = "https://www.ft.com/search"


def _search_via_browser(
    source: SourceDefinition, term: str, *, top_k: int, robots: Optional["RobotsChecker"] = None
) -> List[CandidateURL]:
    """Only reached when the plain HTTP search endpoint yielded nothing for
    this term (see `discover_via_ft_search`). Drives FT's own search UI
    through Crawl4AI/Playwright: loads FT's homepage-level search entry
    point, locates the search input, types the term, submits it, and waits
    for real article links to render -- never scrapes the FT homepage/AI
    hub itself as if it were search results. Best-effort: any failure
    (selector not found, navigation error, timeout) yields an empty list,
    exactly like every other optional browser fallback in this project --
    never raises, never aborts the retrieval run.

    Unlike the HTTP tier, this never goes through `HTTPClient.get()`, so it
    would otherwise never consult robots.txt at all -- checked explicitly
    here instead, using the SAME `RobotsChecker` the HTTP tier's
    `http_client` already carries (see `discover_via_ft_search`), so using
    a browser for this one tier can never become a way to bypass robots.txt
    "by accident" through a different code path. CONFIRMED LIVE
    (2026-10-04): FT's own robots.txt disallows `/search` for `User-agent:
    *` -- this correctly makes this tier (and the HTTP tier above) a no-op
    against the real site. Left in place, rather than removed, because a
    future change to FT's robots.txt would make it work with no code
    change, and the behavior -- and why it is currently inert -- is now
    explicit and tested rather than silently broken."""
    if not CRAWL4AI_AVAILABLE:
        return []
    if robots is not None and not robots.is_allowed(_FT_SEARCH_PAGE_URL):
        logger.info("FT search term %r: robots blocked; skipped", term)
        return []

    js_code = """
    const input = document.querySelector('input[type="search"], input[name="q"], [data-trackable*="search"] input');
    if (input) {
        input.focus();
        input.value = %r;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        const form = input.closest('form');
        if (form) {
            form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
        } else {
            input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
        }
    }
    """ % term

    async def _run() -> str:
        browser_config = BrowserConfig(headless=True)
        run_config = CrawlerRunConfig(
            js_code=js_code,
            wait_for="css:a[href*='/content/']",
            wait_for_timeout=15000,
            page_timeout=20000,
        )
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=_FT_SEARCH_PAGE_URL, config=run_config)
            return result.html or ""

    try:
        html = asyncio.run(_run())
    except Exception as exc:
        logger.warning("FT browser search failed for %r: %s", term, exc)
        return []

    if not html:
        return []
    return _parse_ft_search_results_html(html, base_url=_FT_SEARCH_PAGE_URL, source=source, top_k=top_k)


def discover_via_ft_search(
    source: SourceDefinition, *, http_client: HTTPClient, cache_policy: CachePolicy = CachePolicy.CACHE,
) -> List[CandidateURL]:
    """Entry point used by `discovery.discover_candidates` -- additive to
    whatever the source's normal hub-page discovery already found, gated
    entirely on `source.search` being configured (today, only Financial
    Times). Runs each configured search term (HTTP first, browser only if
    HTTP found nothing for that term), deduplicating results by canonical
    URL across every term before returning -- a term appearing in multiple
    queries' results never produces duplicate candidates."""
    search = source.search
    if search is None or not search.enabled or not search.search_terms:
        return []

    candidates: List[CandidateURL] = []
    seen: Set[str] = set()
    for term in search.search_terms:
        term_results = _search_via_http(
            source, term, http_client=http_client, cache_policy=cache_policy, top_k=search.top_k_per_query,
        )
        if not term_results:
            term_results = _search_via_browser(
                source, term, top_k=search.top_k_per_query, robots=http_client.robots,
            )

        for candidate in term_results:
            key = candidate.url.split("?")[0].rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

    logger.info("FT search discovery: %d unique candidate(s) across %d term(s)", len(candidates), len(search.search_terms))
    return candidates
