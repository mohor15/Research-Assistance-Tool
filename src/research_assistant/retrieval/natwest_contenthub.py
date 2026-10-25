"""Targeted content-hub API discovery for NatWest Group ONLY, scoped to its
"AI & Data" press-release theme -- the section explicitly requested for
this source.

NatWest's press-releases hub page (`https://www.natwestgroup.com/news-and-
insights/news-room/press-releases.html`) never contains article links in
its initial HTML, nor even after a plain browser render -- confirmed via
live inspection (2026-10-14): the static HTML has zero press-release links
in it, a `?tags=...` query string has no effect, and a Playwright render
with `wait_until="domcontentloaded"`/`"networkidle"` plus an extra wait
still shows "No articles have been found." The real article list loads
only once the filter widget's "Apply" control is used, which fires a POST
to an AEM content-hub JSON endpoint -- confirmed via live network-request
capture (accept cookies, open the filter panel, click Apply) to be:

  POST https://www.natwestgroup.com/content/natwestgroup_com/en_uk/
       natwestgroup/news-and-insights/news-room/press-releases/
       jcr:content/root/responsivegrid/filter_copy_copy.contenthub.filter.json

with a JSON body whose `filters` list carries one entry per selected theme
tag (`{"firstleveltags": "natwestgroup-content-hub/themes/<tag>"}`). This
module calls that same confirmed endpoint directly with only the "Ai & data"
theme tag selected, bypassing the need to simulate the filter UI
interaction in a browser. No authentication/CSRF token is required --
confirmed via a plain unauthenticated POST returning real results
(`"success": true`, 22 real press releases, including the live example
used to confirm this: "NatWest improves AI speech recognition for UK
banking customers with NVIDIA Nemotron").

This is the first NON-HTML-scraping, site-specific discovery adapter in
this project besides `retrieval.ft_search` (Financial Times' search
endpoint) -- a deliberate, narrow exception to the "generic extraction,
never per-publisher special-casing" rule, justified the same way FT's is:
there is no generic way to discover this content, and the endpoint/payload
shape is fully confirmed rather than guessed. If NatWest ever changes this
widget, this module simply stops returning anything (handled the same as
every other empty-discovery case -- never raises), with no risk of
silently drifting onto the wrong content."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional
from urllib.parse import urljoin

import requests

if TYPE_CHECKING:
    from research_assistant.retrieval.robots import RobotsChecker

from research_assistant.retrieval.dates import parse_date
from research_assistant.retrieval.models import CandidateURL, DiscoveryMethod, SourceDefinition

logger = logging.getLogger(__name__)

NATWEST_DOMAIN = "natwestgroup.com"

_BASE_URL = "https://www.natwestgroup.com"
_CONTENT_HUB_ENDPOINT = (
    "https://www.natwestgroup.com/content/natwestgroup_com/en_uk/natwestgroup/"
    "news-and-insights/news-room/press-releases/jcr:content/root/responsivegrid/"
    "filter_copy_copy.contenthub.filter.json"
)
_SEARCHPATH = "/content/natwestgroup_com/en_uk/natwestgroup/news-and-insights"
_AI_DATA_THEME_TAG = "natwestgroup-content-hub/themes/ai-&-data"
_PRESS_RELEASE_CONTENT_TYPE_TAG = "natwestgroup-content-hub/content-type/press-release"

# Each article's `url` field is an AEM *internal* JCR content path (e.g.
# "/content/natwestgroup_com/en_uk/natwestgroup/news-and-insights/news-room/
# press-releases/2026/..."), which 404s if fetched directly -- confirmed
# live. The real public page drops this content-root prefix and has a
# ".html" extension (confirmed live on multiple real articles: stripping
# this exact prefix and appending ".html" 200s and lands on the real page).
_CONTENT_ROOT_PREFIX = "/content/natwestgroup_com/en_uk/natwestgroup"

_REQUEST_TIMEOUT_SECONDS = 20.0
# Bounded safety net, mirroring `discovery._PR_NEWSWIRE_MAX_PAGES`'s own
# bounded-pagination pattern: the real site currently returns its entire
# "AI & Data" result set (22 articles) in a single call, so this only ever
# matters if that volume grows well past one page in the future.
_MAX_REQUESTS = 5


def _payload(start_index: int) -> Dict:
    return {
        "searchpath": _SEARCHPATH,
        "filters": [{"firstleveltags": _AI_DATA_THEME_TAG}],
        "isHubPageFilter": False,
        "isDefaultArticles": True,
        "startIndex": start_index,
        "brand": "NATWESTGROUP_COM",
        "defaultcontenttypetags": _PRESS_RELEASE_CONTENT_TYPE_TAG,
    }


def _public_url(content_path: str) -> str:
    """Map an AEM internal content path to the real public page. See
    `_CONTENT_ROOT_PREFIX` for why this is necessary."""
    public_path = content_path
    if public_path.startswith(_CONTENT_ROOT_PREFIX):
        public_path = public_path[len(_CONTENT_ROOT_PREFIX):]
    if not public_path.endswith(".html"):
        public_path += ".html"
    return urljoin(_BASE_URL, public_path)


def _candidate_from_article(article: Dict, *, source: SourceDefinition, now: datetime) -> Optional[CandidateURL]:
    relative_url = article.get("url")
    if not relative_url:
        return None
    return CandidateURL(
        url=_public_url(relative_url),
        source_name=source.name,
        domain=source.domain,
        discovery_method=DiscoveryMethod.CONTENT_HUB_API,
        discovered_at=now,
        title=article.get("title") or None,
        published_at=parse_date(article.get("date")),
    )


def discover_via_natwest_contenthub(
    source: SourceDefinition, *, robots: Optional["RobotsChecker"] = None,
    timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
) -> List[CandidateURL]:
    """Entry point used by `discovery.discover_candidates` -- additive to
    whatever the source's normal hub-page discovery already found (today,
    nothing real; see module docstring). Gated entirely on
    `source.domain == NATWEST_DOMAIN` (see that call site), so this never
    fires for any other source. Best-effort: any failure (network error,
    non-200, non-JSON, unexpected shape) yields an empty list, exactly
    like every other optional/additive discovery path in this project --
    never raises, never aborts the retrieval run."""
    if robots is not None and not robots.is_allowed(_CONTENT_HUB_ENDPOINT):
        logger.info("NatWest content-hub API: robots.txt disallows %s; skipped", _CONTENT_HUB_ENDPOINT)
        return []

    now = datetime.now(timezone.utc)
    candidates: List[CandidateURL] = []
    seen = set()
    start_index = 0
    total_articles: Optional[int] = None

    for _ in range(_MAX_REQUESTS):
        try:
            response = requests.post(
                _CONTENT_HUB_ENDPOINT, json=_payload(start_index), timeout=timeout_seconds,
                headers={"User-Agent": "Mozilla/5.0 (compatible; research-assistant/1.0)"},
            )
        except requests.RequestException as exc:
            logger.warning("NatWest content-hub API request failed: %s", exc)
            break

        if response.status_code != 200:
            logger.warning("NatWest content-hub API returned HTTP %s", response.status_code)
            break

        try:
            data = response.json()
        except ValueError:
            logger.warning("NatWest content-hub API returned non-JSON content")
            break

        if not data.get("success"):
            break

        response_object = data.get("responseObject") or {}
        if total_articles is None:
            total_articles = response_object.get("totalArticles") or 0

        new_count = 0
        for page_entry in response_object.get("paginationList") or []:
            for article in page_entry.get("articles") or []:
                candidate = _candidate_from_article(article, source=source, now=now)
                if candidate is None:
                    continue
                key = candidate.url.split("?")[0].rstrip("/")
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
                new_count += 1

        if new_count == 0 or len(candidates) >= total_articles:
            break
        start_index = len(candidates)

    logger.info("NatWest content-hub API discovery: %d unique AI & Data candidate(s)", len(candidates))
    return candidates
