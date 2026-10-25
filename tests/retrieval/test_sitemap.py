"""Phase 13F ULTIMATUM 3: sitemap.xml-based scavenging discovery, additive
to the existing RSS/Atom + HTML-link discovery paths. Generic and
publisher-agnostic -- no test here references a real company/URL by name
beyond the synthetic `example.com`/`analyst.example` fixtures."""

from __future__ import annotations

from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.discovery import discover_candidates
from research_assistant.retrieval.models import DiscoveryMethod, SourceCategory, SourceDefinition
from research_assistant.retrieval.sitemap import discover_via_sitemap

from tests.retrieval.fakes import FakeResponse, FakeSession

URLSET_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://analyst.example/insights/the-economics-of-agentic-ai</loc><lastmod>2026-09-15</lastmod></url>
  <url><loc>https://analyst.example/insights/ai-operating-models</loc><lastmod>2026-09-20</lastmod></url>
  <url><loc>https://analyst.example/careers/open-roles</loc></url>
  <url><loc>https://otherdomain.com/not-ours</loc></url>
</urlset>
"""

SITEMAP_INDEX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://analyst.example/sitemap-insights.xml</loc></sitemap>
  <sitemap><loc>https://analyst.example/sitemap-legal.xml</loc></sitemap>
</sitemapindex>
"""

NESTED_URLSET_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://analyst.example/insights/enterprise-ai-adoption-survey</loc></url>
</urlset>
"""


def _analyst_source(urls=("https://analyst.example/insights",)) -> SourceDefinition:
    return SourceDefinition(
        name="Example Analyst", domain="analyst.example", urls=list(urls),
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.ANALYST,
    )


def test_a_plain_urlset_sitemap_yields_same_domain_entries_only():
    session = FakeSession([FakeResponse(status_code=200, url="https://analyst.example/sitemap.xml", text=URLSET_XML)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    entries = discover_via_sitemap(
        domain="analyst.example", source_name="Example Analyst", http_client=client,
        source_netlocs={"analyst.example"},
    )
    urls = {e.url for e in entries}

    assert "https://analyst.example/insights/the-economics-of-agentic-ai" in urls
    assert "https://otherdomain.com/not-ours" not in urls
    assert all(e.discovery_method == DiscoveryMethod.SITEMAP for e in entries)


def test_a_sitemap_index_is_followed_into_its_nested_sitemaps():
    session = FakeSession([
        FakeResponse(status_code=200, url="https://analyst.example/sitemap.xml", text=SITEMAP_INDEX_XML),
        FakeResponse(status_code=200, url="https://analyst.example/sitemap-insights.xml", text=NESTED_URLSET_XML),
        FakeResponse(status_code=200, url="https://analyst.example/sitemap-legal.xml", text="<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'></urlset>"),
    ])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    entries = discover_via_sitemap(
        domain="analyst.example", source_name="Example Analyst", http_client=client,
        source_netlocs={"analyst.example"},
    )

    assert any("enterprise-ai-adoption-survey" in e.url for e in entries)


def test_a_missing_sitemap_yields_zero_entries_without_raising():
    session = FakeSession([])  # both /sitemap.xml and /sitemap_index.xml probes 404 via FakeSession's default
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    entries = discover_via_sitemap(
        domain="analyst.example", source_name="Example Analyst", http_client=client,
        source_netlocs={"analyst.example"},
    )

    assert entries == []


def test_malformed_sitemap_xml_yields_zero_entries_without_raising():
    session = FakeSession([FakeResponse(status_code=200, url="https://analyst.example/sitemap.xml", text="not valid xml <<<")])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    entries = discover_via_sitemap(
        domain="analyst.example", source_name="Example Analyst", http_client=client,
        source_netlocs={"analyst.example"},
    )

    assert entries == []


# ---------------------------------------------------------------------------
# Integration: discover_candidates() for an ANALYST-category source merges
# sitemap-derived candidates in additively, after its normal HTML-link pass.
# ---------------------------------------------------------------------------

THIN_HUB_HTML = """
<html><body><main>
  <a href="/insights/quarterly-roundup">Quarterly Roundup</a>
</main></body></html>
"""


def test_sitemap_scavenging_supplements_a_thin_analyst_hub_page():
    session = FakeSession([
        FakeResponse(status_code=200, url="https://analyst.example/insights", text=THIN_HUB_HTML),
        FakeResponse(status_code=200, url="https://analyst.example/sitemap.xml", text=URLSET_XML),
    ])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    candidates = discover_candidates(_analyst_source(), http_client=client, max_candidates=50)
    urls = {c.url for c in candidates}

    assert "https://analyst.example/insights/quarterly-roundup" in urls  # from the HTML page
    assert "https://analyst.example/insights/the-economics-of-agentic-ai" in urls  # from the sitemap
    assert "https://analyst.example/careers/open-roles" not in urls  # sitemap entries still pass the relevance filter


def test_sitemap_scavenging_is_not_attempted_for_a_non_analyst_source():
    """Scoped to ANALYST-category sources (see discovery.py rationale) --
    a peer-bank source with the same thin hub page must NOT trigger an
    extra sitemap probe."""
    peer_source = SourceDefinition(
        name="Example Bank", domain="analyst.example", urls=["https://analyst.example/insights"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )
    session = FakeSession([FakeResponse(status_code=200, url="https://analyst.example/insights", text=THIN_HUB_HTML)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    discover_candidates(peer_source, http_client=client, max_candidates=50)

    assert len(session.calls) == 1  # no sitemap probe issued


# ---------------------------------------------------------------------------
# CNBC coverage-gap fix (confirmed real, 2026-10-15): CNBC's configured hub
# page is a single, unpaginated "latest stories" list with no RSS/search
# fallback. A real, in-window CNBC article was live-confirmed never linked
# from it across multiple crawls, while CNBC's own robots.txt-advertised
# `sitemap_news.xml` (a rolling ~48h site-wide news sitemap) did carry it.
# Sitemap scavenging is now also attempted for `domain == "cnbc.com"`
# specifically (additive to the existing ANALYST-category gate, never
# replacing it), and `_CANDIDATE_SITEMAP_PATHS` tries `/sitemap_news.xml` as
# a fallback after the two conventional paths.
# ---------------------------------------------------------------------------

CNBC_NEWS_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://www.cnbc.com/2026/10/12/meta-muse-wonder-ai-model-agent.html</loc></url>
  <url><loc>https://www.cnbc.com/2026/10/15/cafe-fuel-economy-v8-trucks.html</loc></url>
</urlset>
"""


def _cnbc_source() -> SourceDefinition:
    return SourceDefinition(
        name="CNBC", domain="cnbc.com", urls=["https://www.cnbc.com/ai-artificial-intelligence/"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
    )


def test_sitemap_scavenging_recovers_a_cnbc_article_missing_from_the_ai_hub_page():
    session = FakeSession([
        FakeResponse(status_code=200, url="https://www.cnbc.com/ai-artificial-intelligence/", text=THIN_HUB_HTML),
        FakeResponse(status_code=404, url="https://cnbc.com/sitemap.xml", text=""),
        FakeResponse(status_code=404, url="https://cnbc.com/sitemap_index.xml", text=""),
        FakeResponse(status_code=200, url="https://cnbc.com/sitemap_news.xml", text=CNBC_NEWS_SITEMAP_XML),
    ])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    candidates = discover_candidates(_cnbc_source(), http_client=client, max_candidates=50)
    urls = {c.url for c in candidates}

    assert "https://www.cnbc.com/2026/10/12/meta-muse-wonder-ai-model-agent.html" in urls


def test_sitemap_scavenging_is_not_attempted_for_a_secondary_source_that_is_not_cnbc():
    """A non-CNBC secondary source (e.g. Reuters) must NOT trigger the
    extra sitemap probe -- this is domain-gated to cnbc.com specifically,
    not opened up to the whole SECONDARY category."""
    reuters_source = SourceDefinition(
        name="Reuters", domain="analyst.example", urls=["https://analyst.example/insights"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
    )
    session = FakeSession([FakeResponse(status_code=200, url="https://analyst.example/insights", text=THIN_HUB_HTML)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    discover_candidates(reuters_source, http_client=client, max_candidates=50)

    assert len(session.calls) == 1  # no sitemap probe issued
