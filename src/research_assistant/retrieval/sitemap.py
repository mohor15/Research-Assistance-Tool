"""Phase 13F: sitemap.xml-based candidate discovery -- a second, independent
discovery path alongside the existing RSS/Atom-feed and HTML-link scanning
in `discovery.py` (module docstring there: "Prefers RSS/Atom... falls back
to scanning a fetched source page for same-domain, article-shaped links").

Many research/analyst publishers (McKinsey, BCG, Deloitte, KPMG, PwC-style
sites) do not expose a conventional RSS feed at their configured hub URL,
but do publish a standard `/sitemap.xml` (optionally a `<sitemapindex>` of
nested per-section sitemaps) -- this is public site-structure metadata, not
a bypass of any access control, and parsing it is exactly the kind of
"systematic exploitation of the source's own public information
architecture" (as opposed to an article-specific hardcoded URL) this phase
asks for.

This module is pure parsing + one bounded, best-effort HTTP fetch sequence;
it never raises on a missing/malformed sitemap (a 404 or garbled XML simply
yields zero entries, exactly like `feeds.py`'s "a bad feed... yields zero
candidates rather than aborting the retrieval run"), and it is generic --
nothing here references a specific publisher, URL, or article title.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Optional, Set
from urllib.parse import urlparse

from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.dates import parse_date
from research_assistant.retrieval.models import CachePolicy, CandidateURL, DiscoveryMethod

logger = logging.getLogger(__name__)

_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

# Bounded so a very large publisher sitemap (tens of thousands of URLs
# across an entire site, not just its research section) can never turn one
# source into an unbounded crawl -- this is a scavenging aid, not a
# replacement for the existing topical ranking + max_candidates truncation
# that still applies after this module hands its entries back.
_MAX_NESTED_SITEMAPS = 5
_MAX_URLS_PER_SITEMAP = 2000
_MAX_TOTAL_SITEMAP_CANDIDATES = 500

_CANDIDATE_SITEMAP_PATHS = ("/sitemap.xml", "/sitemap_index.xml", "/sitemap_news.xml")


def discover_via_sitemap(
    *, domain: str, source_name: str, http_client: HTTPClient, source_netlocs: Set[str],
    cache_policy: CachePolicy = CachePolicy.CACHE,
) -> List[CandidateURL]:
    """Probe the conventional sitemap location(s) for `domain` and return
    whatever same-site URLs they list, as `CandidateURL`s. Best-effort and
    silent on failure -- a source with no sitemap (or one that 404s, or
    isn't valid XML) simply contributes zero entries here, exactly as if
    this function had never been called; the existing feed/HTML-link
    discovery paths are completely unaffected either way."""
    for path in _CANDIDATE_SITEMAP_PATHS:
        url = f"https://{domain}{path}"
        page = http_client.get(url, cache_policy=cache_policy)
        if not page.success or not page.html:
            continue
        entries = _parse_sitemap_content(
            page.html, http_client=http_client, source_name=source_name, domain=domain,
            source_netlocs=source_netlocs, depth=0, cache_policy=cache_policy,
        )
        if entries:
            logger.info("Sitemap discovery found %d candidate(s) for %s via %s", len(entries), source_name, url)
            return entries[:_MAX_TOTAL_SITEMAP_CANDIDATES]
    return []


def _parse_sitemap_content(
    xml_text: str, *, http_client: HTTPClient, source_name: str, domain: str,
    source_netlocs: Set[str], depth: int, cache_policy: CachePolicy = CachePolicy.CACHE,
) -> List[CandidateURL]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.info("Malformed sitemap XML for %s: %s", source_name, exc)
        return []

    tag = root.tag.lower()
    now = datetime.now(timezone.utc)

    if tag.endswith("sitemapindex"):
        if depth >= _MAX_NESTED_SITEMAPS:
            return []
        entries: List[CandidateURL] = []
        for sitemap_el in root.findall(f"{_SITEMAP_NS}sitemap")[:_MAX_NESTED_SITEMAPS]:
            loc = _loc_text(sitemap_el)
            if not loc:
                continue
            page = http_client.get(loc, cache_policy=cache_policy)
            if not page.success or not page.html:
                continue
            entries.extend(
                _parse_sitemap_content(
                    page.html, http_client=http_client, source_name=source_name, domain=domain,
                    source_netlocs=source_netlocs, depth=depth + 1, cache_policy=cache_policy,
                )
            )
            if len(entries) >= _MAX_TOTAL_SITEMAP_CANDIDATES:
                break
        return entries

    if tag.endswith("urlset"):
        entries = []
        for url_el in root.findall(f"{_SITEMAP_NS}url")[:_MAX_URLS_PER_SITEMAP]:
            loc = _loc_text(url_el)
            if not loc:
                continue
            parsed = urlparse(loc)
            if parsed.netloc not in source_netlocs and domain not in parsed.netloc:
                continue  # never follow a sitemap off the configured source's own site
            lastmod_el = url_el.find(f"{_SITEMAP_NS}lastmod")
            lastmod = lastmod_el.text.strip() if lastmod_el is not None and lastmod_el.text else None
            entries.append(
                CandidateURL(
                    url=loc,
                    source_name=source_name,
                    domain=domain,
                    discovery_method=DiscoveryMethod.SITEMAP,
                    discovered_at=now,
                    title=None,
                    published_at=parse_date(lastmod),
                )
            )
        return entries

    logger.info("Unrecognized sitemap root tag <%s> for %s", root.tag, source_name)
    return []


def _loc_text(el: ET.Element) -> Optional[str]:
    loc = el.find(f"{_SITEMAP_NS}loc")
    if loc is None or not loc.text:
        return None
    return loc.text.strip()
