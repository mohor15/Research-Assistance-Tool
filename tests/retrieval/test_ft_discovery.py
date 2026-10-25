"""Financial Times topic/section-page exclusion (forensic finding, CURRENT
run `e818e08adc2e4dd1`): FT's own topic/section/navigation pages
("/war-in-ukraine", "/middle-east-north-africa", "/the-big-read") were
discovered as candidates from the configured AI/Technology hub pages and
assigned a fabricated "today" publication date, wasting 5 of that run's 7
triage slots on non-articles.

Reuses the exact `/content/<uuid>` article-shape rule
`retrieval.ft_search` already proved, via extensive live inspection, is
the ONLY shape FT's own individual articles ever use (see that module's
own docstring and `_FT_ARTICLE_PATH_RE`) -- never a blanket ft.com
exclusion, and never keyed to a specific topic/section slug name.

No live Ollama/network anywhere -- FakeSession/FakeResponse only.
"""

from __future__ import annotations

from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.discovery import discover_candidates
from research_assistant.retrieval.models import DiscoveryMethod, SourceCategory, SourceDefinition

from tests.retrieval.fakes import FakeResponse, FakeSession

# The exact three topic/section pages named in the forensic report, plus
# two genuine `/content/<uuid>` articles -- taken verbatim from the real
# cached HTML this run's own artifacts served (see the forensic report's
# "PR Newswire Example-by-Example Stage Trace" section for the equivalent
# PR Newswire evidence; these three FT paths and the two real article
# UUIDs are drawn directly from this run's own `pipeline_runs.diagnostics_
# json`).
FT_HUB_PAGE_HTML = """
<html><body><main>
  <a href="/content/8d948115-5843-4419-967e-f79e65718f55">The danger of pessimism fatigue</a>
  <a href="/content/6ff0232c-f38a-4c70-8ca9-b0eed437f09b">Napa Valley's hangover</a>
  <a href="/war-in-ukraine">War in Ukraine</a>
  <a href="/middle-east-north-africa">Middle East &amp; North Africa</a>
  <a href="/the-big-read">The Big Read</a>
  <a href="/artificial-intelligence">Artificial intelligence</a>
  <a href="/technology">Technology</a>
</main></body></html>
"""


def _ft_source(url: str = "https://www.ft.com/artificial-intelligence") -> SourceDefinition:
    return SourceDefinition(
        name="Financial Times", domain="ft.com", urls=[url],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH,
        category=SourceCategory.SECONDARY,
    )


def _discover(html: str, configured_url: str):
    session = FakeSession([FakeResponse(status_code=200, url=configured_url, text=html)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    source = _ft_source(configured_url)
    return discover_candidates(source, http_client=client, max_candidates=50)


def test_the_three_named_topic_section_pages_are_rejected():
    configured_url = "https://www.ft.com/artificial-intelligence"
    candidates = _discover(FT_HUB_PAGE_HTML, configured_url)

    urls = {c.url for c in candidates}
    for rejected_path in ("/war-in-ukraine", "/middle-east-north-africa", "/the-big-read"):
        assert not any(u.endswith(rejected_path) for u in urls), f"{rejected_path} should have been rejected"


def test_the_configured_hub_pages_themselves_are_never_candidates():
    configured_url = "https://www.ft.com/artificial-intelligence"
    candidates = _discover(FT_HUB_PAGE_HTML, configured_url)

    urls = {c.url for c in candidates}
    assert not any(u.endswith("/artificial-intelligence") or u.endswith("/technology") for u in urls)


def test_genuine_content_uuid_articles_are_kept():
    configured_url = "https://www.ft.com/artificial-intelligence"
    candidates = _discover(FT_HUB_PAGE_HTML, configured_url)

    urls = {c.url for c in candidates}
    assert "https://www.ft.com/content/8d948115-5843-4419-967e-f79e65718f55" in urls
    assert "https://www.ft.com/content/6ff0232c-f38a-4c70-8ca9-b0eed437f09b" in urls


def test_only_the_two_genuine_articles_survive_out_of_seven_links():
    configured_url = "https://www.ft.com/artificial-intelligence"
    candidates = _discover(FT_HUB_PAGE_HTML, configured_url)

    assert len(candidates) == 2
    assert all(c.discovery_method == DiscoveryMethod.HTML_LINKS for c in candidates)


def test_an_article_is_never_rejected_merely_for_an_old_looking_or_broad_topic_title():
    """Negative control (explicit task requirement): the filter is driven
    purely by URL shape, never by title wording -- a genuine `/content/
    <uuid>` article with a broad-sounding or old-looking title must still
    be kept."""
    html = """
    <html><body><main>
      <a href="/content/1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d">A Decade-Old Debate About Markets Resurfaces</a>
    </main></body></html>
    """
    candidates = _discover(html, "https://www.ft.com/artificial-intelligence")
    urls = {c.url for c in candidates}
    assert "https://www.ft.com/content/1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d" in urls


def test_no_blanket_ft_domain_exclusion_every_content_link_is_evaluated_on_its_own_shape():
    """Negative control: this is a shape rule, never a domain-level
    on/off switch -- an ft.com page with ONLY genuine articles must still
    discover all of them."""
    html = """
    <html><body><main>
      <a href="/content/11111111-1111-1111-1111-111111111111">First real article</a>
      <a href="/content/22222222-2222-2222-2222-222222222222">Second real article</a>
      <a href="/content/33333333-3333-3333-3333-333333333333">Third real article</a>
    </main></body></html>
    """
    candidates = _discover(html, "https://www.ft.com/artificial-intelligence")
    assert len(candidates) == 3


def test_non_ft_sources_are_unaffected_by_this_filter():
    other_source = SourceDefinition(
        name="Example Bank", domain="example.com",
        urls=["https://example.com/newsroom"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )
    html = """
    <html><body><main>
      <a href="/war-in-ukraine">War in Ukraine (not an FT path, must not be rejected by the FT rule)</a>
    </main></body></html>
    """
    session = FakeSession([FakeResponse(status_code=200, url="https://example.com/newsroom", text=html)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    candidates = discover_candidates(other_source, http_client=client, max_candidates=50)

    assert len(candidates) == 1
    assert candidates[0].url == "https://example.com/war-in-ukraine"
