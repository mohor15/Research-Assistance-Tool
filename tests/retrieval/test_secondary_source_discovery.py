"""Regression: secondary sources previously had no `urls:` configured at
all, so `discover_candidates()`'s `for source_url in source.urls:` loop
silently never executed for any of them -- an empty-list no-op, not a
crash, which is why it went unnoticed. Now that each secondary source has
a real, verified discovery-entry-point URL, this proves the discovery loop
actually traverses it and produces candidates, using the REAL loaded
`config/sources.yaml` registry (not a synthetic fixture).
"""

from __future__ import annotations

from research_assistant.config import get_config
from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.discovery import discover_candidates
from research_assistant.retrieval.models import DiscoveryMethod, SourceCategory, SourceDefinition
from research_assistant.retrieval.service import build_source_definitions

from tests.retrieval.fakes import FakeResponse, FakeSession

SECONDARY_PAGE_HTML = """
<html><body><main>
  <a href="/news/2026/09/ai-copilot-bank-launch">Bank launches AI copilot for advisors</a>
  <a href="/news/2026/09/second-ai-story">Second AI story</a>
</main></body></html>
"""

# PR Newswire's own domain requires the release's numeric-ID URL shape (see
# retrieval.discovery's `_is_pr_newswire_release_url`) -- the generic
# `SECONDARY_PAGE_HTML` fixture's `/news/...` links are deliberately NOT
# shaped like a real release and would now be correctly rejected for this
# one domain, so PR Newswire needs its own realistic fixture here.
PR_NEWSWIRE_PAGE_HTML = """
<html><body><main>
  <a href="/news-releases/example-company-launches-new-ai-platform-302912345.html">Example Company Launches New AI Platform</a>
  <a href="/news-releases/acme-ai-raises-series-b-funding-302912346.html">Acme AI Raises Series B Funding</a>
  <a href="/news-releases/latest-news-topics/artificial-intelligence/">All Artificial Intelligence News and Press Releases from PR Newswire</a>
  <a href="/news-releases/financial-services-latest-news/venture-capital-list/">All Venture Capital News and Press Releases from PR Newswire</a>
</main></body></html>
"""

# Financial Times requires the `/content/<uuid>` article-shape (see
# retrieval.discovery's `_is_ft_article_url` FT-specific override, and
# retrieval.ft_search's module docstring, which already proved this is the
# ONLY real individual-article URL shape FT ever uses) -- the generic
# `SECONDARY_PAGE_HTML` fixture's `/news/...` links are deliberately NOT
# shaped like a real FT article and would now be correctly rejected, so FT
# needs its own realistic fixture here, exactly like PR Newswire above.
FT_PAGE_HTML = """
<html><body><main>
  <a href="/content/8d948115-5843-4419-967e-f79e65718f55">A real FT article about AI</a>
  <a href="/content/6ff0232c-f38a-4c70-8ca9-b0eed437f09b">A second real FT article</a>
  <a href="/war-in-ukraine">War in Ukraine</a>
  <a href="/the-big-read">The Big Read</a>
</main></body></html>
"""


def _secondary_source_definition(name: str) -> SourceDefinition:
    config = get_config()
    entry = next(s for s in config.sources.secondary_sources if s.name == name)
    return SourceDefinition(
        name=entry.name, domain=entry.domain, urls=entry.urls,
        source_type=entry.source_type, priority=entry.priority, category=SourceCategory.SECONDARY,
    )


def test_every_secondary_source_definition_has_at_least_one_url_from_real_config():
    config = get_config()
    defs = build_source_definitions(config)
    secondary_defs = [d for d in defs if d.category == SourceCategory.SECONDARY]
    assert len(secondary_defs) == 11
    for d in secondary_defs:
        assert len(d.urls) >= 1, f"{d.name} has no configured urls"


def test_discovery_actually_fetches_the_configured_secondary_source_url():
    """Before the fix, this call would have hit `FakeSession ran out of
    scripted responses` never -- it would have returned [] with ZERO HTTP
    calls at all, since the urls list was empty. Now it must issue exactly
    one GET to the real configured URL and return real candidates."""
    source = _secondary_source_definition("TechCrunch")
    configured_url = source.urls[0]

    session = FakeSession([FakeResponse(status_code=200, url=configured_url, text=SECONDARY_PAGE_HTML)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    candidates = discover_candidates(source, http_client=client, max_candidates=50)

    assert len(session.calls) == 1
    assert session.calls[0]["url"] == configured_url
    assert len(candidates) == 2
    assert all(c.discovery_method == DiscoveryMethod.HTML_LINKS for c in candidates)


def test_pr_newswire_discovery_is_traversed_and_stays_press_distribution():
    source = _secondary_source_definition("PR Newswire")
    configured_url = source.urls[0]

    session = FakeSession([FakeResponse(status_code=200, url=configured_url, text=PR_NEWSWIRE_PAGE_HTML)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    candidates = discover_candidates(source, http_client=client, max_candidates=50)

    assert len(session.calls) == 1
    # Only the 2 release-shaped links -- the topic/category links are
    # rejected regardless of their "news"/"press-release" keep-keyword hits.
    assert len(candidates) == 2
    assert all(c.url.endswith(".html") and c.url.rsplit("-", 1)[-1][:-5].isdigit() for c in candidates)
    assert source.source_type.value == "press_distribution"


def test_all_nine_secondary_sources_are_individually_traversable():
    """Most secondary sources have exactly one configured URL; WSJ has two
    (its main AI page plus its official RSS feed -- see config/sources.yaml)
    since its main page is anti-bot-blocked for automated clients, so this
    scripts one response per configured URL rather than assuming exactly
    one."""
    config = get_config()
    for entry in config.sources.secondary_sources:
        source = SourceDefinition(
            name=entry.name, domain=entry.domain, urls=entry.urls,
            source_type=entry.source_type, priority=entry.priority, category=SourceCategory.SECONDARY,
        )
        if entry.domain == "prnewswire.com":
            page_html = PR_NEWSWIRE_PAGE_HTML
        elif entry.domain == "ft.com":
            page_html = FT_PAGE_HTML
        else:
            page_html = SECONDARY_PAGE_HTML
        responses = [FakeResponse(status_code=200, url=u, text=page_html) for u in source.urls]
        session = FakeSession(responses)
        client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

        candidates = discover_candidates(source, http_client=client, max_candidates=50)

        # CNBC coverage-gap fix: CNBC also triggers 3 additive sitemap-
        # scavenging probes (/sitemap.xml, /sitemap_index.xml,
        # /sitemap_news.xml) after its configured hub-page URL(s) -- all
        # unscripted here, so FakeSession auto-404s them (contributing zero
        # candidates), but they still count as traversal calls.
        expected_calls = len(source.urls) + (3 if entry.domain == "cnbc.com" else 0)
        assert len(session.calls) == expected_calls, f"{entry.name} was not fully traversed"
        # At least the source's own-domain URL(s) must yield candidates; a
        # second URL on a different host (e.g. WSJ's feeds.a.dj.com RSS
        # endpoint) is correctly scoped out of plain HTML-link discovery by
        # the existing domain check -- this only proves every configured
        # URL was actually fetched, not that every one yields links.
        assert len(candidates) >= 2, f"{entry.name} produced no candidates"
