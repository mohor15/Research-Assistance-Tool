"""Discovery-time URL relevance filtering.

Candidate discovery should reject obvious navigation/account/marketing/
corporate-directory/reference pages before expensive retrieval or browser
fallback, while remaining conservative: plausible research/news/article
paths and genuinely ambiguous paths are both kept. This filter only applies
to the HTML-link-scraping fallback — RSS/Atom-discovered entries are never
run through it.
"""

from __future__ import annotations

from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.discovery import _is_relevant_candidate, discover_candidates
from research_assistant.retrieval.models import DiscoveryMethod, SourceCategory, SourceDefinition

from tests.retrieval.fakes import FakeResponse, FakeSession

# ---------------------------------------------------------------------------
# Unit-level: `_is_relevant_candidate` against representative real-world URLs
# ---------------------------------------------------------------------------

REJECT_URLS = [
    "https://www.microsoft.com/en-us/store/b/sale?icid=DSM_TopNavDeals",
    "https://www.microsoft.com/microsoft-365",
    "https://newsroom.bankofamerica.com/content/newsroom/executive-bios",
    "https://newsroom.bankofamerica.com/content/newsroom/journalist-resources",
    "https://newsroom.bankofamerica.com/content/newsroom/prime-rate-information",
    "https://newsroom.bankofamerica.com/content/newsroom/company-overview",
    "https://newsroom.bankofamerica.com/content/newsroom/awards",
    "https://newsroom.bankofamerica.com/content/newsroom/company-overview/bank-of-america-fast-facts",
]

KEEP_URLS = [
    "https://www.microsoft.com/en-us/ai",
    "https://newsroom.bankofamerica.com/content/newsroom/press-releases/example",
    "https://www.jpmorganchase.com/newsroom/news",
    "https://www.db.com/what-we-do/research/example",
    "https://openai.com/news/2025/09/gpt-launch-announcement",  # representative AI announcement URL
]


def _path(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).path


def test_representative_navigation_marketing_and_directory_urls_are_rejected():
    for url in REJECT_URLS:
        assert _is_relevant_candidate(_path(url), None) is False, url


def test_representative_research_news_and_ai_urls_are_kept():
    for url in KEEP_URLS:
        assert _is_relevant_candidate(_path(url), None) is True, url


def test_ambiguous_multi_segment_urls_with_no_keyword_hit_are_kept_not_filtered():
    # No reject or keep keyword anywhere, and not a bare single-segment path.
    assert _is_relevant_candidate("/2025/q3-update", None) is True


def test_bare_short_product_page_is_rescued_by_clearly_relevant_anchor_text():
    # "/microsoft-365" alone reads as a product page, but if the link's own
    # anchor text is unambiguous news/announcement language, it is kept.
    assert _is_relevant_candidate("/microsoft-365", None) is False
    assert _is_relevant_candidate("/microsoft-365", "Microsoft 365 Copilot research announcement") is True


def test_reject_keyword_wins_even_when_a_keep_keyword_also_appears_in_the_path():
    # ".../newsroom/executive-bios" contains "newsroom" (a keep keyword) but
    # must still be rejected — reject has final precedence.
    assert _is_relevant_candidate("/content/newsroom/executive-bios", None) is False


def test_keyword_matching_does_not_false_positive_on_substrings():
    # "newsroom" must not spuriously match the keyword "news".
    # (Confirmed indirectly: a bare "/newsroom" path is kept via the "newsroom"
    # keep keyword itself, not a partial match of "news".)
    assert _is_relevant_candidate("/newsroom", None) is True
    # "press-kit" (reject) must not be confused with "press-release" (keep).
    assert _is_relevant_candidate("/press-kit", None) is False
    assert _is_relevant_candidate("/press-release/example", None) is True


# ---------------------------------------------------------------------------
# Source-aware scope: out-of-scope, keyword-free paths are rejected; keep
# keywords and in-scope ambiguous paths are unaffected.
# ---------------------------------------------------------------------------


def test_out_of_scope_generic_product_pages_are_rejected():
    # Real-world regression: Microsoft AI is registered at blogs.microsoft.com,
    # but its newsroom page cross-links deep into www.microsoft.com's
    # unrelated product catalog (Outlook, OneDrive), which has no keep
    # keyword anywhere in the path.
    assert _is_relevant_candidate(
        "/en-us/microsoft-365/outlook/email-and-calendar-software-microsoft-outlook", None, in_scope=False
    ) is False
    assert _is_relevant_candidate(
        "/en-us/microsoft-365/onedrive/online-cloud-storage", None, in_scope=False
    ) is False


def test_out_of_scope_path_is_still_kept_when_a_keep_keyword_matches():
    # A keep-keyword hit always wins, regardless of scope.
    assert _is_relevant_candidate("/en-us/ai", None, in_scope=False) is True


def test_in_scope_ambiguous_path_is_kept_but_out_of_scope_ambiguous_path_is_rejected():
    ambiguous_path = "/2025/quarterly-roundup"
    assert _is_relevant_candidate(ambiguous_path, None, in_scope=True) is True
    assert _is_relevant_candidate(ambiguous_path, None, in_scope=False) is False


def test_discovery_rejects_out_of_scope_product_pages_but_keeps_in_scope_and_keyword_matches():
    html = """
    <html><body><main>
      <a href="https://blogs.microsoft.com/ai/news/copilot-update">Copilot update</a>
      <a href="https://blogs.microsoft.com/ai/2025/quarterly-roundup">Quarterly roundup</a>
      <a href="https://www.microsoft.com/en-us/microsoft-365/outlook/email-and-calendar-software-microsoft-outlook">Outlook</a>
      <a href="https://www.microsoft.com/en-us/microsoft-365/onedrive/online-cloud-storage">OneDrive</a>
      <a href="https://www.microsoft.com/en-us/ai">Microsoft AI</a>
    </main></body></html>
    """
    session = FakeSession([FakeResponse(status_code=200, url="https://blogs.microsoft.com/ai/", text=html)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    source = SourceDefinition(
        name="Microsoft AI", domain="microsoft.com", urls=["https://blogs.microsoft.com/ai/"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.AI_PROVIDER,
    )

    candidates = discover_candidates(source, http_client=client, max_candidates=50)
    urls = {c.url for c in candidates}

    assert "https://blogs.microsoft.com/ai/news/copilot-update" in urls
    assert "https://blogs.microsoft.com/ai/2025/quarterly-roundup" in urls  # in-scope, ambiguous, kept
    assert "https://www.microsoft.com/en-us/ai" in urls  # out-of-scope but keep-keyword hit
    assert not any("outlook" in u for u in urls)
    assert not any("onedrive" in u for u in urls)


# ---------------------------------------------------------------------------
# Integration: discover_candidates end-to-end with a realistic mixed page
# ---------------------------------------------------------------------------

MIXED_NEWSROOM_HTML = """
<html>
<body>
  <nav>
    <a href="/privacy">Privacy Policy</a>
    <a href="/careers">Careers</a>
  </nav>
  <main>
    <a href="/content/newsroom/press-releases/ai-copilot-launch">Bank launches AI copilot</a>
    <a href="/content/newsroom/press-releases/ai-copilot-launch">Duplicate link, same story</a>
    <a href="/content/newsroom/executive-bios">Executive bios</a>
    <a href="/content/newsroom/journalist-resources">Journalist resources</a>
    <a href="/content/newsroom/prime-rate-information">Prime rate information</a>
    <a href="/content/newsroom/company-overview">Company overview</a>
    <a href="/content/newsroom/awards">Awards</a>
    <a href="https://otherdomain.com/news/unrelated">Unrelated story off-domain</a>
    <a href="/content/newsroom/press-releases/second-story">Second real story</a>
  </main>
</body>
</html>
"""


def _source() -> SourceDefinition:
    return SourceDefinition(
        name="Example Bank",
        domain="example.com",
        urls=["https://example.com/newsroom"],
        source_type=SourceType.PRIMARY,
        priority=SourcePriority.HIGH,
        category=SourceCategory.PEER,
    )


def _client_returning(html: str) -> HTTPClient:
    session = FakeSession([FakeResponse(status_code=200, url="https://example.com/newsroom", text=html)])
    return HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)


def test_discovery_rejects_directory_and_utility_pages_but_keeps_press_releases():
    candidates = discover_candidates(_source(), http_client=_client_returning(MIXED_NEWSROOM_HTML), max_candidates=50)
    urls = {c.url for c in candidates}

    assert "https://example.com/content/newsroom/press-releases/ai-copilot-launch" in urls
    assert "https://example.com/content/newsroom/press-releases/second-story" in urls
    assert not any("executive-bios" in u for u in urls)
    assert not any("journalist-resources" in u for u in urls)
    assert not any("prime-rate-information" in u for u in urls)
    assert not any("company-overview" in u for u in urls)
    assert not any("awards" in u for u in urls)
    assert not any("privacy" in u for u in urls)
    assert not any("careers" in u for u in urls)


def test_discovery_deduplication_still_works_alongside_relevance_filtering():
    candidates = discover_candidates(_source(), http_client=_client_returning(MIXED_NEWSROOM_HTML), max_candidates=50)
    urls = [c.url for c in candidates]
    hits = [u for u in urls if "ai-copilot-launch" in u]
    assert len(hits) == 1


def test_discovery_same_domain_restriction_still_works_alongside_relevance_filtering():
    candidates = discover_candidates(_source(), http_client=_client_returning(MIXED_NEWSROOM_HTML), max_candidates=50)
    urls = {c.url for c in candidates}
    assert not any("otherdomain.com" in u for u in urls)


# ---------------------------------------------------------------------------
# Phase 12 EOD hardening: bare index/hub/archive roots, and PDF/media
# asset URLs, must never reach the LLM analysis stage as if they were
# specific articles (real Phase 10/11 regressions: NVIDIA's bare "/news"
# archive, OpenAI's bare "/research" and "/research/index", a NVIDIA PDF
# gallery asset, and a NVIDIA homepage tagline containing the bare word
# "AI" rescuing an otherwise-bare product page).
# ---------------------------------------------------------------------------


def test_bare_index_and_archive_roots_are_rejected():
    for path in ("/news", "/research", "/research/index", "/news/research", "/blog", "/press", "/archive"):
        assert _is_relevant_candidate(path, None) is False, path
        assert _is_relevant_candidate(path, None, in_scope=True) is False, path


def test_bare_index_root_rejection_does_not_affect_a_sources_own_newsroom_entry():
    # "/newsroom" alone is a source's own declared entry point, not a
    # generic cross-site section label -- unaffected by the new bare-index
    # check (deliberately excluded from `_HUB_INDEX_SEGMENTS`).
    assert _is_relevant_candidate("/newsroom", None) is True


def test_a_long_headline_slug_under_an_index_root_is_still_kept():
    # The bare-root check only fires when EVERY segment is a generic
    # index word -- a real headline nested under one is unaffected.
    assert _is_relevant_candidate("/blog/dsx-ready-ai-factories-power-cooling", None) is True
    assert _is_relevant_candidate("/index/introducing-gpt-6-1-sol", None) is True


def test_pdf_and_media_asset_urls_are_rejected_by_path_extension():
    # A direct file-extension path is rejected at discovery time, before
    # any fetch. (The NVIDIA gallery case from Phase 10/11 signals its file
    # type via a query parameter, not the path extension -- that case is
    # instead caught by the Content-Type guard in `retrieval.extractor`,
    # already covered by its own regression tests.)
    for path in (
        "/assets/photo.jpg",
        "/news/report.pdf",  # even under an otherwise-keep-keyword path
    ):
        assert _is_relevant_candidate(path, None) is False, path


def test_generic_ai_tagline_anchor_text_no_longer_rescues_a_bare_homepage_path():
    # Real regression: nvidia.com's homepage ("/en-us") anchor text is
    # "World Leader in AI Computing" -- the bare word "ai" inside a generic
    # marketing tagline must not be enough to treat the homepage as an
    # article.
    assert _is_relevant_candidate("/en-us", "World Leader in AI Computing") is False
    # A genuinely substantive anchor (containing a stronger signal than
    # just "ai") still rescues a bare path, exactly as before.
    assert _is_relevant_candidate("/microsoft-365", "Microsoft 365 Copilot research announcement") is True


def test_rss_atom_candidates_bypass_the_relevance_filter_entirely():
    # A feed entry whose URL would be rejected by the path filter (no
    # news/research keyword, single short segment) must still survive,
    # because feed-discovered entries are never run through it.
    rss = """<?xml version="1.0"?><rss version="2.0"><channel>
        <item><title>Some update</title><link>https://example.com/microsoft-365</link></item>
        </channel></rss>"""
    session = FakeSession([
        FakeResponse(status_code=200, url="https://example.com/feed.xml", text=rss, headers={"Content-Type": "application/rss+xml"})
    ])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    candidates = discover_candidates(
        SourceDefinition(
            name="Example Bank", domain="example.com", urls=["https://example.com/feed.xml"],
            source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
        ),
        http_client=client,
        max_candidates=50,
    )

    assert len(candidates) == 1
    assert candidates[0].discovery_method == DiscoveryMethod.RSS
    assert candidates[0].url == "https://example.com/microsoft-365"
