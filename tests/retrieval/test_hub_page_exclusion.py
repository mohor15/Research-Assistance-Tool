"""A source's own configured hub/index page must never be promoted to a
retrievable candidate via its own self-referencing link.

Real-world regression: BCG's configured insights hub
(https://www.bcg.com/capabilities/artificial-intelligence/insights) links
back to itself (a relative self-link, seen in the real page's own nav/
breadcrumb markup) with anchor text that also happens to contain
"artificial-intelligence", a keep-keyword -- so without this exclusion, the
hub page itself becomes a discovered candidate, gets retrieved and
extracted, and ends up as a single "Article" whose content is the entire
hub page rather than one discrete news item. The desired architecture:
SOURCE/HUB PAGE -> discovery only; INDIVIDUAL ARTICLE PAGE -> retrieval +
extraction + evidence.
"""

from __future__ import annotations

from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.discovery import discover_candidates
from research_assistant.retrieval.models import SourceCategory, SourceDefinition

from tests.retrieval.fakes import FakeResponse, FakeSession

# Mirrors the real bcg.com page's confirmed markup: both a relative and an
# absolute self-link back to the exact configured hub URL, alongside one
# genuine article link.
HUB_HTML_WITH_SELF_LINKS = """
<html><body>
  <nav>
    <a href="/capabilities/artificial-intelligence/insights">AI Insights</a>
  </nav>
  <main>
    <a href="https://www.bcg.com/capabilities/artificial-intelligence/insights">Artificial Intelligence Insights</a>
    <a href="/capabilities/artificial-intelligence/insights/generative-ai-equity-research">
      How Generative AI Is Reshaping Equity Research
    </a>
  </main>
</body></html>
"""


def _bcg_source() -> SourceDefinition:
    return SourceDefinition(
        name="BCG", domain="bcg.com",
        urls=["https://www.bcg.com/capabilities/artificial-intelligence/insights"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.ANALYST,
    )


def _client_returning(html: str) -> HTTPClient:
    session = FakeSession([
        FakeResponse(status_code=200, url="https://www.bcg.com/capabilities/artificial-intelligence/insights", text=html)
    ])
    return HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)


def test_hub_page_self_link_relative_and_absolute_variants_are_excluded():
    candidates = discover_candidates(_bcg_source(), http_client=_client_returning(HUB_HTML_WITH_SELF_LINKS), max_candidates=50)
    urls = {c.url for c in candidates}

    assert "https://www.bcg.com/capabilities/artificial-intelligence/insights" not in urls


def test_genuine_sub_article_discovered_on_the_hub_page_is_still_kept():
    candidates = discover_candidates(_bcg_source(), http_client=_client_returning(HUB_HTML_WITH_SELF_LINKS), max_candidates=50)
    urls = {c.url for c in candidates}

    assert "https://www.bcg.com/capabilities/artificial-intelligence/insights/generative-ai-equity-research" in urls


def test_hub_page_with_only_self_links_yields_no_candidates():
    html = """
    <html><body>
      <a href="/capabilities/artificial-intelligence/insights">AI Insights</a>
      <a href="https://www.bcg.com/capabilities/artificial-intelligence/insights">Artificial Intelligence Insights</a>
      <a href="https://www.bcg.com/capabilities/artificial-intelligence/insights/">With trailing slash</a>
    </body></html>
    """
    candidates = discover_candidates(_bcg_source(), http_client=_client_returning(html), max_candidates=50)

    assert candidates == []


def test_a_source_with_multiple_configured_urls_excludes_self_links_to_either():
    source = SourceDefinition(
        name="Example", domain="example.com",
        urls=["https://example.com/news", "https://example.com/insights"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.ANALYST,
    )
    html = """
    <html><body>
      <a href="https://example.com/insights">Insights hub (cross-link from the news hub)</a>
      <a href="https://example.com/news/real-article">A genuine news article</a>
    </body></html>
    """
    session = FakeSession([
        FakeResponse(status_code=200, url="https://example.com/news", text=html),
        FakeResponse(status_code=200, url="https://example.com/insights", text="<html><body></body></html>"),
    ])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    candidates = discover_candidates(source, http_client=client, max_candidates=50)
    urls = {c.url for c in candidates}

    assert "https://example.com/insights" not in urls
    assert "https://example.com/news/real-article" in urls
