"""Discovery: relative links, duplicate URLs, navigation filtering, candidate limits."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.discovery import discover_candidates
from research_assistant.retrieval.exceptions import SourceUnreachableError
from research_assistant.retrieval.models import DiscoveryMethod, SourceCategory, SourceDefinition

from tests.retrieval.fakes import FakeResponse, FakeSession

NEWSROOM_HTML = """
<html>
<body>
  <nav>
    <a href="/privacy">Privacy Policy</a>
    <a href="/careers">Careers</a>
    <a href="/contact-us">Contact us</a>
  </nav>
  <main>
    <a href="/news/2025/09/ai-copilot-launch">Bank launches AI copilot</a>
    <a href="/news/2025/09/ai-copilot-launch">Bank launches AI copilot (duplicate link)</a>
    <a href="/news/2025/09/ai-copilot-launch#comments">Bank launches AI copilot (with fragment)</a>
    <a href="https://otherdomain.com/news/unrelated">An unrelated story on another domain</a>
    <a href="javascript:void(0)">Expand menu</a>
    <a href="mailto:press@example.com">Email press</a>
    <a href="/news/2025/09/second-story">Second real story</a>
    <a href="/news/2025/09/third-story">Third real story</a>
  </main>
</body>
</html>
"""


def _source(urls) -> SourceDefinition:
    return SourceDefinition(
        name="Example Bank",
        domain="example.com",
        urls=urls,
        source_type=SourceType.PRIMARY,
        priority=SourcePriority.HIGH,
        category=SourceCategory.PEER,
    )


def _http_client_returning(html: str) -> HTTPClient:
    session = FakeSession([FakeResponse(status_code=200, url="https://example.com/newsroom", text=html)])
    return HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)


def test_relative_links_are_resolved_to_absolute_urls():
    client = _http_client_returning(NEWSROOM_HTML)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    urls = {c.url for c in candidates}
    assert "https://example.com/news/2025/09/ai-copilot-launch" in urls
    assert all(c.discovery_method == DiscoveryMethod.HTML_LINKS for c in candidates)


def test_duplicate_urls_including_fragment_variants_are_deduplicated():
    client = _http_client_returning(NEWSROOM_HTML)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    urls = [c.url for c in candidates]
    ai_copilot_hits = [u for u in urls if "ai-copilot-launch" in u]
    assert len(ai_copilot_hits) == 1


def test_navigation_and_off_domain_links_are_excluded():
    client = _http_client_returning(NEWSROOM_HTML)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    urls = {c.url for c in candidates}
    assert not any("privacy" in u for u in urls)
    assert not any("careers" in u for u in urls)
    assert not any("contact-us" in u for u in urls)
    assert not any("otherdomain.com" in u for u in urls)


def test_candidate_limit_is_respected():
    client = _http_client_returning(NEWSROOM_HTML)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=2)

    assert len(candidates) == 2


def _hub_page_with_a_late_on_topic_link(off_topic_count: int) -> str:
    """A hub page with many generic, keyword-free links followed by ONE
    genuinely on-topic link near the very end -- reproduces the real-world
    shape of a research firm's insights page (hundreds of generic links,
    with the actually-relevant publication appearing well past any
    reasonable document-order cutoff). Generic/synthetic -- no company name
    or real URL is hardcoded here. Deliberately NOT under "/industries/" or
    "/capabilities/" (a separate, narrower rule now rejects a SHORT slug
    directly under those two specific roots regardless of ranking/
    truncation -- seede.g. `_is_short_taxonomy_hub_page` -- so this fixture
    uses an unrelated generic path to keep testing the ranking/truncation
    interaction this test is actually about, independently of that rule)."""
    off_topic_links = "\n".join(
        f'<a href="/resources/practice-{i}">Resource page {i}</a>' for i in range(off_topic_count)
    )
    return f"""
    <html><body><main>
      {off_topic_links}
      <a href="/publications/the-formula-for-agentic-ai-value">The Formula for Agentic AI Value</a>
    </main></body></html>
    """


def test_without_topical_ranking_a_late_on_topic_link_is_silently_lost():
    """The real-world regression reproduced generically: a page listing more
    links than `max_candidates`, where the one genuinely on-topic link
    appears after the cutoff position, in raw document order."""
    html = _hub_page_with_a_late_on_topic_link(off_topic_count=60)
    client = _http_client_returning(html)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    assert not any("agentic-ai-value" in c.url for c in candidates)


def test_topical_ranking_rescues_a_late_on_topic_link_from_truncation():
    """Same page, same cutoff -- but with the existing topics vocabulary
    supplied, the on-topic link survives the cutoff regardless of its raw
    document position."""
    html = _hub_page_with_a_late_on_topic_link(off_topic_count=60)
    client = _http_client_returning(html)
    candidates = discover_candidates(
        _source(["https://example.com/newsroom"]), http_client=client, max_candidates=50,
        topic_keywords=["agentic ai"],
    )

    assert any("agentic-ai-value" in c.url for c in candidates)
    assert len(candidates) == 50  # the cap itself is unchanged, only what fills it


def test_topical_ranking_is_a_no_op_when_no_keyword_anywhere_matches():
    """Backward compatibility: when nothing in a page matches the topic
    vocabulary at all (the all-zero-score case), stable-sort ties preserve
    the original document order -- identical behavior to no ranking."""
    unranked = discover_candidates(
        _source(["https://example.com/newsroom"]), http_client=_http_client_returning(NEWSROOM_HTML), max_candidates=50,
    )
    ranked = discover_candidates(
        _source(["https://example.com/newsroom"]), http_client=_http_client_returning(NEWSROOM_HTML), max_candidates=50,
        topic_keywords=["a keyword phrase that appears nowhere on this page"],
    )

    assert [c.url for c in unranked] == [c.url for c in ranked]


def test_a_source_with_every_url_unfetchable_raises_source_unreachable():
    """Phase 1 fix: a whole-source fetch failure must be a detectable
    signal (previously this silently returned [], indistinguishable from
    "fetched fine, found nothing" -- see SourceUnreachableError)."""
    session = FakeSession([FakeResponse(status_code=500, url="https://example.com/newsroom", text="")])
    # max_retries=0 so this resolves quickly to a clean failure
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    with pytest.raises(SourceUnreachableError, match="http_error"):
        discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)


def test_a_source_with_only_some_urls_unfetchable_does_not_raise():
    """Only a TOTAL failure (every configured URL) raises -- a source with
    multiple URLs where at least one succeeds still returns whatever
    candidates it found, exactly as before."""
    html = '<html><body><main><a href="/news/story">A real story</a></main></body></html>'
    session = FakeSession([
        FakeResponse(status_code=500, url="https://example.com/newsroom-a", text=""),
        FakeResponse(status_code=200, url="https://example.com/newsroom-b", text=html),
    ])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    candidates = discover_candidates(
        _source(["https://example.com/newsroom-a", "https://example.com/newsroom-b"]),
        http_client=client, max_candidates=50,
    )

    assert len(candidates) == 1


def test_feed_source_is_routed_to_feed_parsing_not_link_scraping():
    rss = """<?xml version="1.0"?><rss version="2.0"><channel>
        <item><title>A</title><link>https://example.com/a</link></item>
        </channel></rss>"""
    session = FakeSession([
        FakeResponse(status_code=200, url="https://example.com/feed.xml", text=rss, headers={"Content-Type": "application/rss+xml"})
    ])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    candidates = discover_candidates(_source(["https://example.com/feed.xml"]), http_client=client, max_candidates=50)

    assert len(candidates) == 1
    assert candidates[0].discovery_method == DiscoveryMethod.RSS


# ---------------------------------------------------------------------------
# Coverage audit findings (real production data): a multinational corporate
# site's locale/country selector menu and a research firm's own dated
# "/publications/" content type.
# ---------------------------------------------------------------------------

_LOCALE_SELECTOR_HUB_HTML = """
<html><body><main>
  <a href="/us/en/home.html">United States</a>
  <a href="/au/en/home.html">Australia</a>
  <a href="/at/de/home.html">Austria (Deutsch)</a>
  <a href="/bd/en/home.html">Bangladesh</a>
  <a href="/publications/2026/real-ai-report">A genuine dated AI publication</a>
</main></body></html>
"""


def test_locale_selector_homepages_are_never_candidates():
    """Real production finding: every one of 11 persisted "articles" for
    an analyst source was actually a country-selector homepage link
    (`/xx/en/home.html`) from the configured hub page's own locale menu --
    zero genuine content ever reached analysis. This must never happen
    again, for any source sharing the same URL shape."""
    client = _http_client_returning(_LOCALE_SELECTOR_HUB_HTML)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    urls = {c.url for c in candidates}
    assert not any(u.endswith("/home.html") for u in urls)
    assert "https://example.com/publications/2026/real-ai-report" in urls


def test_a_real_looking_publication_slug_is_not_mistaken_for_a_locale_homepage():
    """Negative control: the locale-homepage shape is narrow (two short
    codes then literally "home") -- it must never reject an ordinary
    multi-word publication slug just because it happens to start with a
    short-looking segment."""
    html = '<html><body><a href="/ai/the-future-of-agentic-ai-in-banking">A real publication</a></body></html>'
    client = _http_client_returning(html)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    urls = {c.url for c in candidates}
    assert "https://example.com/ai/the-future-of-agentic-ai-in-banking" in urls


def test_publications_path_segment_is_a_keep_keyword():
    """Real production finding: BCG's one genuine dated publication found
    in this project's entire history lived at "/publications/<year>/<slug>",
    a content-type word the keep-keyword vocabulary never covered -- it
    ranked no higher than the dozens of generic "/capabilities/..." /
    "/industries/..." taxonomy pages the same hub page links to."""
    html = (
        '<html><body>'
        '<a href="/capabilities/artificial-intelligence">AI capability hub (out of scope, no keyword)</a>'
        '<a href="/publications/2026/how-agentic-ai-helps-banks">A real dated publication</a>'
        '</body></html>'
    )
    client = _http_client_returning(html)
    candidates = discover_candidates(
        _source(["https://example.com/newsroom"]), http_client=client, max_candidates=50,
        topic_keywords=["agentic ai"],
    )

    urls = {c.url for c in candidates}
    assert "https://example.com/publications/2026/how-agentic-ai-helps-banks" in urls


def test_short_practice_area_taxonomy_pages_are_excluded_despite_ai_keyword():
    """Real production finding: a research firm's AI-insights hub page
    links to dozens of generic "/capabilities/<topic>"/"/industries/<sector>"
    category pages that all contain the "ai"/"artificial-intelligence"
    keep-keyword (e.g. "/capabilities/artificial-intelligence/ai-agents"),
    so they crowded out real "/publications/..." releases once ranked and
    truncated -- confirmed real: only 13 of 40 real BCG publications
    survived a 50-candidate cutoff before this fix, because 36 of the 50
    slots were consumed by this exact category-page shape."""
    html = """
    <html><body>
      <a href="/capabilities/artificial-intelligence/ai-agents">AI Agents capability</a>
      <a href="/industries/financial-institutions/banking">Banking industry</a>
      <a href="/capabilities/risk-management-and-compliance/credit-risk-management">Credit risk capability</a>
      <a href="/capabilities/artificial-intelligence">Artificial Intelligence (bare root)</a>
      <a href="/industries">Industries (bare root)</a>
    </body></html>
    """
    client = _http_client_returning(html)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    urls = {c.url for c in candidates}
    assert not any("/capabilities/" in u or "/industries/" in u or u.endswith("/capabilities") for u in urls)


def test_a_deep_article_under_capabilities_with_a_real_content_keyword_is_still_kept():
    """Negative control (real regression this fix must never reintroduce):
    a genuine BCG article can live arbitrarily deep under "/capabilities/"
    -- the confirmed real case is ".../capabilities/artificial-intelligence/
    insights/generative-ai-equity-research", which must survive because
    "insights" is a real content-type keyword, not because of path depth
    or slug length."""
    html = (
        '<html><body><a href="/capabilities/artificial-intelligence/insights/'
        'generative-ai-equity-research">How Generative AI Is Reshaping Equity Research</a></body></html>'
    )
    client = _http_client_returning(html)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    urls = {c.url for c in candidates}
    assert "https://example.com/capabilities/artificial-intelligence/insights/generative-ai-equity-research" in urls


def test_a_long_headline_slug_under_capabilities_is_not_rejected_by_depth_or_length_alone():
    """Negative control: the taxonomy-hub rule only ever rejects a SHORT
    slug (<=4 hyphenated tokens) directly under "/capabilities/"/
    "/industries/" -- a longer, headline-style slug (the shape every
    confirmed real publication/sub-topic page actually has) must survive
    even with no other keep-keyword present, so this can only ever remove
    confirmed category noise, never a long real-looking slug."""
    html = (
        '<html><body><a href="/capabilities/international-business/'
        'go-to-market-strategy-for-emerging-markets">Go-to-market strategy</a></body></html>'
    )
    client = _http_client_returning(html)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    urls = {c.url for c in candidates}
    assert "https://example.com/capabilities/international-business/go-to-market-strategy-for-emerging-markets" in urls


# ---------------------------------------------------------------------------
# Evergreen/navigation-page reduction (forensic finding, CURRENT run
# `e818e08adc2e4dd1`): a real HSBC AI-news run discovered "/who-we-are",
# "/who-we-are/esg-and-responsible-business/tax-transparency", and
# "/who-we-are/our-markets" as candidates -- none a dated article -- and a
# real Deutsche Bank run discovered "/what-we-do/..." pages the same way.
# A real OpenAI run also discovered its own "/news/rss.xml" feed-file link
# (embedded as a plain anchor on an ordinary HTML page) as if it were an
# article. All three contributed to 194 of 201 period-eligible candidates
# in that run having no extractable publication date at all.
# ---------------------------------------------------------------------------


def test_who_we_are_corporate_overview_pages_are_rejected():
    html = """
    <html><body>
      <a href="/who-we-are">Who we are</a>
      <a href="/who-we-are/esg-and-responsible-business/tax-transparency">Tax transparency</a>
      <a href="/who-we-are/our-markets">Our markets</a>
    </body></html>
    """
    client = _http_client_returning(html)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    assert candidates == []


def test_what_we_do_corporate_overview_pages_are_rejected():
    html = """
    <html><body>
      <a href="/what-we-do">What we do</a>
      <a href="/what-we-do/products-and-services/client-logins">Client logins</a>
    </body></html>
    """
    client = _http_client_returning(html)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    assert candidates == []


def test_a_real_article_is_kept_despite_containing_markets_ai_or_research():
    """Negative control (the task's own explicit requirement): a genuine
    dated article must never be rejected merely for containing a generic
    word like "markets", "ai", or "research" -- only the specific, narrow
    corporate-overview-section phrases above ("who-we-are", "what-we-do")
    are rejected, never a broader word they happen to share a root with."""
    html = """
    <html><body>
      <a href="/news/2026/09/our-markets-ai-research-report">Our Markets AI Research Report launched</a>
      <a href="/insights/the-future-of-ai-in-banking-markets">The future of AI in banking markets</a>
    </body></html>
    """
    client = _http_client_returning(html)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    urls = {c.url for c in candidates}
    assert "https://example.com/news/2026/09/our-markets-ai-research-report" in urls
    assert "https://example.com/insights/the-future-of-ai-in-banking-markets" in urls


def test_rss_xml_feed_link_embedded_in_an_html_page_is_rejected():
    """Real OpenAI finding: the page's own "news" keep-keyword (present in
    the URL path) must never rescue a bare feed-file link -- this is
    checked by URL extension before any keyword logic runs."""
    html = """
    <html><body>
      <a href="/news/rss.xml">RSS</a>
      <a href="/news/index.atom">Atom feed</a>
    </body></html>
    """
    client = _http_client_returning(html)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    assert candidates == []


def test_a_real_news_article_next_to_a_feed_link_is_still_kept():
    html = """
    <html><body>
      <a href="/news/rss.xml">RSS</a>
      <a href="/news/2026/09/ai-copilot-launch">Bank launches AI copilot</a>
    </body></html>
    """
    client = _http_client_returning(html)
    candidates = discover_candidates(_source(["https://example.com/newsroom"]), http_client=client, max_candidates=50)

    urls = {c.url for c in candidates}
    assert "https://example.com/news/2026/09/ai-copilot-launch" in urls
    assert not any(u.endswith(".xml") or u.endswith(".atom") for u in urls)
