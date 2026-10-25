"""Candidate discovery: turn a configured source into a list of `CandidateURL`.

Prefers RSS/Atom (via `feeds.py`) and falls back to scanning a fetched
source page for same-domain, article-shaped links. The source registry
(`config/sources.yaml`) defines the entire research surface — this module
never follows links off the configured domain, so it cannot turn into a
general-purpose web crawler.

The HTML-link fallback additionally runs a cheap, deterministic relevance
filter (`_is_relevant_candidate`) before a link becomes a `CandidateURL`, so
that obvious navigation/marketing/corporate-directory pages never reach the
expensive retrieval/browser-fallback stage. This is a discovery-time
concern only — it never touches article extraction, scoring, or anything
downstream of `CandidateURL`. RSS/Atom-discovered entries are not run
through this filter: a feed is already a curated, publisher-declared list
of content, not a raw link scrape.

The filter is also source-aware: a link is "in scope" when it stays on the
same netloc as one of the source's own configured URLs (e.g. a source
registered at `blogs.microsoft.com/ai/` treats other `blogs.microsoft.com`
links as in-scope). An in-scope, keyword-free path is still kept
conservatively (ambiguous content on the source's own declared research
section is not worth losing), but an *out-of-scope* link (one that wanders
onto a different subdomain/section of the same domain, e.g. the main
`www.microsoft.com` marketing site) with no keep-keyword hit is rejected —
this is what keeps a research-focused source from also crawling that
domain's entire unrelated product catalog. No hardcoded per-company product
list is used anywhere; scope is derived purely from the registry's own
`source.urls`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Set
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup

from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.exceptions import SourceUnreachableError
from research_assistant.retrieval.feeds import looks_like_feed, parse_feed
from research_assistant.retrieval.ft_search import _is_ft_article_url, discover_via_ft_search
from research_assistant.retrieval.natwest_contenthub import NATWEST_DOMAIN, discover_via_natwest_contenthub
from research_assistant.retrieval.models import (
    CachePolicy,
    CandidateURL,
    DiscoveryMethod,
    RetrievalFailure,
    RetrievedPage,
    SourceCategory,
    SourceDefinition,
)
from research_assistant.retrieval.sitemap import discover_via_sitemap

logger = logging.getLogger(__name__)

# Path segments/phrases that mark a link as boilerplate, navigation, account,
# marketing/commerce, corporate-directory, or reference/utility content
# rather than research/news/announcement content. Phrases are matched as
# whole hyphen-delimited word sequences (see `_contains_phrase`), so e.g.
# "press-kit" never spuriously matches "press-release".
_REJECT_KEYWORDS: Set[str] = {
    # navigation / account / legal / boilerplate (retained from the original filter)
    "privacy", "privacy-policy", "terms", "terms-of-use", "terms-of-service",
    "cookie", "cookies", "cookie-policy", "sitemap", "login", "signin", "sign-in",
    "signup", "sign-up", "subscribe", "unsubscribe", "rss", "feed", "search",
    "account", "customer-service", "support", "faq", "faqs", "glossary", "directory",
    # careers / people / corporate-directory
    "careers", "jobs", "job", "executive-bios", "leadership", "board-of-directors",
    "media-contacts", "journalist-resources", "press-kit",
    # marketing / commerce
    "store", "shop", "sale", "deals", "pricing", "buy", "cart", "checkout",
    "coupons", "promotions", "gift-cards",
    # generic corporate "about"/directory pages
    "about", "about-us", "contact", "contact-us", "accessibility", "legal",
    "overview", "company-overview", "fast-facts", "awards", "history", "our-story",
    "locations", "branch-locator", "find-a-branch",
    # calculators / reference / rate-information utility pages
    "calculator", "calculators", "prime-rate-information", "rate-information",
}

# Path segments/phrases that plausibly indicate research/news/announcement
# content. A hit here always keeps the URL, even on an otherwise-ambiguous
# shallow path (see `_looks_like_bare_product_page`).
_KEEP_KEYWORDS: Set[str] = {
    "news", "newsroom", "press-release", "press-releases", "article", "articles",
    "research", "insights", "insight", "blog", "announcement", "announcements",
    "ai", "artificial-intelligence", "partnership", "partnerships",
    "acquisition", "acquisitions", "funding", "investment",
    "model-release", "product-launch", "media-release", "media-releases",
    # Coverage audit finding (BCG): BCG's own individual, dated publications
    # live at "/publications/<year>/<slug>" -- a genuinely distinct, dated
    # content type this registry's sources use (confirmed real: "/bcg.com/
    # publications/2026/how-agentic-ai-will-industrialize-financial-scams")
    # that the keyword set never covered, so it ranked no higher than the
    # dozens of generic "/capabilities/..."/"/industries/..." taxonomy pages
    # the same hub page also links to.
    "publications", "publication",
}

# A narrower subset used only to rescue a bare/short product-style path via
# its own anchor text (see `_looks_like_bare_product_page`). "ai" and
# "artificial-intelligence" are deliberately excluded here: they are
# extremely common in generic marketing taglines (e.g. a homepage's "World
# Leader in AI Computing" anchor) and are too weak a signal, alone, to
# rescue what otherwise reads as a bare brand/product page. They remain
# full keep-keywords for a direct URL-path match (above), where they are
# far more specific (e.g. "/en-us/ai" as a dedicated AI section).
_STRONG_KEEP_KEYWORDS: Set[str] = _KEEP_KEYWORDS - {"ai", "artificial-intelligence"}

# Bare "index root" words: when a path's segments are made ENTIRELY of these
# (e.g. "/news", "/research", "/research/index", "/news/research",
# "/blog"), it reads as a listing/archive/category page rather than a
# specific article -- real articles observed across this registry instead
# carry a long, headline-like slug (e.g. "/index/introducing-gpt-6-1-sol",
# "/blog/dsx-ready-ai-factories-power-cooling"). Deliberately NOT including
# "newsroom" or any of the multi-word press/announcement terms: those stay
# governed by the existing keep-keyword/ambiguous-path rules, since a bare
# "/newsroom" is a source's own declared entry point, not a generic
# cross-site section label.
_HUB_INDEX_SEGMENTS: Set[str] = {"news", "research", "insights", "press", "media", "articles", "blog", "index", "archive"}

# Coverage audit finding (KPMG): a multinational corporate site's country/
# language selector menu links to one homepage per locale, shaped as two
# short (2-3 letter) codes followed by a literal "home" segment (confirmed
# real and NOT a slug coincidence: KPMG's own AI-insights hub page --
# kpmg.com/us/en/insights-by-topic/ai-insights.html -- links to 148 of
# these, e.g. "/au/en/home.html", "/at/de/home.html", "/bd/en/home.html" --
# every one of which was a persisted "article" in this project's real
# database, and NONE was ever a real AI-insights article; the genuine
# content links on the same page never got a chance to be analyzed because
# the candidate pool was dominated by this noise). General by URL shape
# only -- never gated to kpmg.com -- since the same two-short-codes-then-
# "home" shape is extremely unlikely to occur in any real article slug on
# any domain.
_LOCALE_HOMEPAGE_RE = re.compile(r"^/[a-z]{2,3}/[a-z]{2,3}/home(?:\.html?)?/?$", re.IGNORECASE)

# Coverage audit finding (BCG): a research/consulting firm's own practice-
# area taxonomy ("capabilities", "industries") crowds out its genuine dated
# publications -- confirmed real: BCG's AI-insights hub page links to 156
# "/capabilities/<topic>" / "/industries/<sector>" category pages alongside
# only 40 real "/publications/<year>/<slug>" releases, and because these
# category pages also contain generic AI keywords ("artificial-intelligence",
# "ai-agents", ...) they rank just as high as -- and in bulk outnumber --
# the real publications once `_topical_rank_key` truncates to
# `max_candidates`. Deliberately narrow and evidence-backed:
#   - only fires on this exact small, generic root-segment vocabulary
#     (confirmed noise on BCG; both words are generic consulting-industry
#     taxonomy terms, not a brand name), never on an unrelated path;
#   - only rejects a SHORT slug directly under that root (<=4 hyphenated
#     tokens, e.g. "artificial-intelligence", "ai-agents", "financial-
#     institutions/banking") -- every confirmed-noise example observed was
#     this shape. A longer, headline-style slug (5+ tokens, the shape every
#     confirmed real BCG publication/sub-topic page actually has) is left
#     alone, so this can only ever remove confirmed category noise, never
#     risk dropping a real article merely because it happens to live one
#     level under one of these two root words.
#   - "who-we-are"/"what-we-do" (coverage audit finding, CURRENT run
#     `e818e08adc2e4dd1`): two further, extremely common corporate-
#     overview root section names -- confirmed real noise on HSBC's
#     "/who-we-are/..." hub (e.g. "/who-we-are", ".../esg-and-responsible-
#     business/tax-transparency", ".../our-markets", none a dated article)
#     and Deutsche Bank's "/what-we-do/..." hub. Deliberately placed here
#     (never in the unconditional `_REJECT_KEYWORDS` set above) because a
#     real, dated Deutsche Bank research publication genuinely lives at
#     "/what-we-do/research/<slug>" -- an unconditional reject would have
#     discarded it outright; this taxonomy-root mechanism's existing
#     strong-keep-keyword exemption (`_is_short_taxonomy_hub_page`) already
#     protects exactly that case, the same way it already protects a real
#     BCG publication nested under "/capabilities/...".
_TAXONOMY_ROOT_SEGMENTS: Set[str] = {"capabilities", "industries", "who-we-are", "what-we-do"}
_MAX_TAXONOMY_SLUG_TOKENS = 4


def _is_short_taxonomy_hub_page(segments: List[str]) -> bool:
    """A genuine article CAN live arbitrarily deep under one of these two
    root words (confirmed real: BCG's own
    ".../capabilities/artificial-intelligence/insights/generative-ai-equity-
    research") -- the distinguishing signal is never depth or slug length
    alone, but whether a real content-type word (`_STRONG_KEEP_KEYWORDS`,
    the same "ai"/"artificial-intelligence"-excluded set used elsewhere for
    exactly this "too weak alone" reason) appears anywhere in the path.
    Every confirmed-noise example found in production had no such word."""
    if not segments or segments[0] not in _TAXONOMY_ROOT_SEGMENTS:
        return False
    if _contains_phrase("-".join(segments), _STRONG_KEEP_KEYWORDS):
        return False
    if len(segments) == 1:
        return True  # bare "/capabilities" or "/industries" root
    return all(len(seg.split("-")) <= _MAX_TAXONOMY_SLUG_TOKENS for seg in segments[1:])


# Non-HTML binary assets (PDF/image/media) must never become a candidate at
# all -- mirrors the content-type guard in `retrieval.extractor`, applied
# here purely from the URL so an obvious binary asset never costs a fetch.
_BINARY_URL_EXTENSIONS = (
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp",
    ".mp4", ".mp3", ".wav", ".mov", ".zip",
)

# Coverage audit finding (CURRENT run `e818e08adc2e4dd1`): a feed FILE link
# embedded as a plain `<a href="...rss.xml">` anchor inside an ordinary HTML
# page (e.g. OpenAI's own newsroom page linking to its own
# "/news/rss.xml") is never itself an article -- it is a feed endpoint, the
# same kind of thing `discover_candidates`'s own `looks_like_feed`/
# `parse_feed` branch already handles correctly when the FETCHED PAGE
# itself is a feed. That existing branch never runs here, because the
# anchor's own containing page is ordinary HTML, not a feed -- so without
# this check, the keep-keyword "news" (present earlier in the same path,
# e.g. "/news/rss.xml") lets the bare feed-file link itself through as if
# it were a real article. Checked purely by URL extension, before any
# keyword logic, exactly like the existing binary-asset check above --
# generic, never scoped to one domain.
_FEED_URL_EXTENSIONS = (".xml", ".rss", ".atom")

# PR Newswire discovery fix (forensic finding, run 8704470a4a623d72): PR
# Newswire's entire site lives under "/news-releases/...", so the generic
# keep-keyword filter above ("news", "press-release(s)") matches BOTH a
# real individual release (e.g. "/news-releases/apollo-graphql-introduces-
# graphos-agent-services-...-302900890.html") and every one of its category/
# topic LISTING pages (e.g. "/news-releases/financial-services-latest-news/
# venture-capital-list/", "/news-releases/latest-news-topics/artificial-
# intelligence/") -- the keyword filter alone cannot tell them apart, which
# is why a run can discover 27 category pages and zero actual releases.
# Real PR Newswire release URLs end in a long numeric release ID directly
# before ".html" (confirmed against 25 real releases in a live-cached PR
# Newswire AI listing page, with zero false positives/negatives across its
# 284 total discovered links -- every category/topic page instead ends in
# "/", never in "-<digits>.html"). This is a general URL-SHAPE rule, not a
# slug-specific one: it recognizes any release by this numeric-ID suffix,
# not by matching JumpCloud (or any other) title/slug text.
_PR_NEWSWIRE_RELEASE_URL_RE = re.compile(r"-\d{6,}\.html$", re.IGNORECASE)


def _is_pr_newswire_release_url(path: str) -> bool:
    """True only for an individual PR Newswire press-release URL (ends in
    "-<6+ digits>.html") -- never a category/topic/listing page (which
    always ends in "/" on this site). Scoped to `source.domain ==
    "prnewswire.com"` by the one caller in `_discover_links_from_html`;
    never applied to any other source."""
    return bool(_PR_NEWSWIRE_RELEASE_URL_RE.search(path))


# PR Newswire recall fix: the configured AI topic page is paginated
# (`pagesize=25`, `<link rel="next" href=".../?page=2">`) and a live check
# confirmed each page holds a fully DISTINCT set of ~25 individual releases
# (zero overlap between page 1 and page 2) -- discovery previously only
# ever fetched page 1, structurally missing the large majority of same-day
# releases regardless of relevance/triage/Qwen behavior. Bounded (not
# unbounded crawling) and scoped to this one domain only, exactly like the
# existing `source.search`/FT mechanism.
_PR_NEWSWIRE_NEXT_PAGE_RE = re.compile(
    r"""<link\s+rel=["']next["']\s+href=["']([^"']+)["']""", re.IGNORECASE,
)
_PR_NEWSWIRE_MAX_PAGES = 3


def _pr_newswire_next_page_url(html: str, base_url: str) -> Optional[str]:
    match = _PR_NEWSWIRE_NEXT_PAGE_RE.search(html)
    if match is None:
        return None
    return urljoin(base_url, match.group(1))


def discover_candidates_from_page(page: RetrievedPage, source: SourceDefinition) -> List[CandidateURL]:
    """Parse an already-fetched `RetrievedPage` (feed or HTML) for candidate
    links — the same feed-detection, relevance filtering, and self-link
    exclusion `discover_candidates()` applies to a plain HTTP fetch. Exposed
    so `RetrievalService` can re-parse a *browser-rendered* fetch of a
    source's discovery page with identical logic (see
    `RetrievalService._try_browser_rendered_discovery`), instead of
    duplicating this parsing."""
    source_netlocs = {urlparse(u).netloc for u in source.urls}
    source_urls_canonical = {_canonical(u) for u in source.urls}

    if looks_like_feed(page.html, page.content_type):
        entries = parse_feed(page.html, source_name=source.name, domain=source.domain)
        logger.info("Parsed %d feed entries for %s", len(entries), source.name)
    else:
        entries = _discover_links_from_html(
            page.html, base_url=page.final_url, source=source,
            source_netlocs=source_netlocs, source_urls_canonical=source_urls_canonical,
        )
        logger.info("Discovered %d candidate links for %s", len(entries), source.name)
    return entries


def _topical_rank_key(candidate: CandidateURL, topic_keywords: List[str]) -> int:
    """Count of configured AI/financial-services topic phrases (the same
    `config/topics.yaml` vocabulary used elsewhere in the pipeline, not a
    new/invented one) appearing in the candidate's anchor text or URL path.
    Hyphens/underscores in the URL are normalized to spaces so a
    space-separated topic phrase (e.g. "agentic AI") still matches a
    hyphenated path segment (e.g. "the-formula-for-agentic-ai-value")."""
    normalized_path = candidate.url.lower().replace("-", " ").replace("_", " ")
    haystack = f"{(candidate.title or '').lower()} {normalized_path}"
    return sum(1 for keyword in topic_keywords if keyword in haystack)


def discover_candidates(
    source: SourceDefinition,
    *,
    http_client: HTTPClient,
    max_candidates: int,
    topic_keywords: Optional[List[str]] = None,
    cache_policy: CachePolicy = CachePolicy.CACHE,
) -> List[CandidateURL]:
    """`max_candidates` caps how many links become candidates for a source
    whose discovery page lists more links than that (a real, observed case:
    a research-firm's insights hub can easily list 150-200+ links). Without
    `topic_keywords`, the cap simply takes however many links were found
    first in raw page/DOM order -- an arbitrary cutoff with no relevance
    signal at all, which silently discarded a genuinely on-topic publication
    that merely happened to appear below the cutoff position on its own
    hub page (confirmed real-world regression). When `topic_keywords` is
    supplied (the project's own existing AI/financial-services vocabulary),
    ALL discovered links are collected first and ranked by topical-keyword
    presence (Python's stable sort preserves original order among ties --
    including the all-zero case, i.e. a source with no keyword signal
    anywhere behaves exactly as before), and only THEN truncated to
    `max_candidates` -- so an on-topic link is never discarded merely for
    appearing later in the page than an off-topic one."""
    candidates: List[CandidateURL] = []
    seen: Set[str] = set()
    fetch_failures: List[RetrievalFailure] = []

    for source_url in source.urls:
        page = http_client.get(source_url, cache_policy=cache_policy)
        if not page.success or not page.html:
            failure = page.failure or RetrievalFailure(reason="empty_response")
            logger.warning("Could not fetch source page %s for %s (%s)", source_url, source.name, failure.reason)
            fetch_failures.append(failure)
            continue

        entries = discover_candidates_from_page(page, source)

        for entry in entries:
            key = _canonical(entry.url)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(entry)

        # PR Newswire recall fix (see `_PR_NEWSWIRE_MAX_PAGES`'s docstring):
        # follow a bounded number of `rel="next"` pages for this domain
        # only -- every other source's single-page behavior is unchanged.
        if source.domain == "prnewswire.com":
            next_url = _pr_newswire_next_page_url(page.html, page.final_url)
            pages_fetched = 1
            while next_url is not None and pages_fetched < _PR_NEWSWIRE_MAX_PAGES:
                next_page = http_client.get(next_url, cache_policy=cache_policy)
                pages_fetched += 1
                if not next_page.success or not next_page.html:
                    break
                for entry in discover_candidates_from_page(next_page, source):
                    key = _canonical(entry.url)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(entry)
                next_url = _pr_newswire_next_page_url(next_page.html, next_page.final_url)

    # Targeted interactive-search discovery (additive, see retrieval.ft_search
    # module docstring). Gated entirely on `source.search` being configured --
    # today only Financial Times has it, so this never fires for any other
    # source regardless of category/domain. Deliberately run BEFORE the
    # all-URLs-failed check below: FT's search endpoint is a completely
    # separate URL from the configured AI hub page, so a hub-page fetch
    # failure must not prevent search discovery from still finding
    # something.
    search_entries: List[CandidateURL] = []
    if source.search is not None and source.search.enabled:
        search_entries = discover_via_ft_search(source, http_client=http_client, cache_policy=cache_policy)
        for entry in search_entries:
            key = _canonical(entry.url)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(entry)

    # Targeted content-hub API discovery (additive, see retrieval.
    # natwest_contenthub module docstring). Gated entirely on
    # `source.domain == NATWEST_DOMAIN` -- never fires for any other
    # source regardless of category. NatWest's own hub page never yields
    # real article links through the HTML-link path above (confirmed
    # live), so this is currently the sole source of real candidates for
    # it; run before the all-URLs-failed check below for the same reason
    # as FT's search block -- a completely separate endpoint from the
    # configured hub page, so a hub-page fetch failure must not prevent
    # this from still finding something.
    contenthub_entries: List[CandidateURL] = []
    if source.domain == NATWEST_DOMAIN:
        contenthub_entries = discover_via_natwest_contenthub(source, robots=http_client.robots)
        # Prepended, not appended: these come from an explicit, authoritative
        # tag filter (the requested "AI & Data" section), so they must never
        # lose a same-score tie against a generically on-topic-scoring but
        # actually irrelevant HTML-link candidate (confirmed real case: this
        # source's own generic nav links -- "Compliance statement...",
        # "Tax responsibilities..." -- each score 1 keyword hit via a
        # coincidental "compliance"/"tax" match, identical to a real AI
        # article's own score) when `max_candidates`/`max_articles_per_source`
        # later truncates the list -- Python's stable sort preserves this
        # list's existing order among ties, so appearing first here is what
        # wins that tie.
        new_contenthub_entries: List[CandidateURL] = []
        for entry in contenthub_entries:
            key = _canonical(entry.url)
            if key in seen:
                continue
            seen.add(key)
            new_contenthub_entries.append(entry)
        candidates = new_contenthub_entries + candidates

    if source.urls and len(fetch_failures) == len(source.urls) and not search_entries and not contenthub_entries:
        # Every configured URL failed to fetch AND search discovery (if
        # configured) found nothing either -- categorically different from
        # "fetched fine, found nothing" (see SourceUnreachableError). A
        # single-URL source (the overwhelming majority) reduces this to
        # exactly one reason, e.g. "connection_error". `failures` keeps the
        # structured per-URL RetrievalFailure objects so callers can
        # classify honestly (TLS vs timeout vs HTTP 401/403, ...) instead of
        # parsing this joined message.
        raise SourceUnreachableError(
            ", ".join(f.reason for f in fetch_failures), failures=fetch_failures
        )

    # Phase 13F: sitemap-based scavenging, additive to whatever feed/HTML-
    # link discovery above already found. Scoped to ANALYST-category
    # sources specifically -- this is the category the task explicitly
    # calls out ("For analyst sources especially... Investigate and
    # implement where technically feasible") and whose own hub pages have
    # repeatedly shown the thinnest/most JS-heavy coverage in prior-phase
    # live validation; applying it to every category would issue an extra
    # network call per run for sources that already have perfectly good
    # feed/HTML-link coverage, for no real benefit. This is a category-level
    # policy, not a per-source hardcode: it is driven entirely by
    # `source.category`/`source.domain`, never a specific name or URL. Only
    # attempted when at least one configured URL was actually reachable (if
    # every URL failed, the SourceUnreachableError above already fired, and
    # a sitemap probe on the same unreachable domain would not recover
    # anything genuine).
    #
    # CNBC coverage-gap fix (confirmed real, 2026-10-15): CNBC's configured
    # hub page (`https://www.cnbc.com/ai-artificial-intelligence/`) is a
    # single, unpaginated "latest stories" list with no search/RSS fallback
    # (unlike FT, which has `source.search`) -- live-verified that a real,
    # in-window CNBC article ("Meta debuts Muse Wonder AI model...",
    # published 2026-10-12T17:00 UTC) was never linked from that hub page
    # across any of ~15 crawls spanning 2026-10-12 through 2026-10-14, even
    # though it stayed within the 48h CURRENT window the whole time -- the
    # article was simply never featured on that one page at any crawl
    # moment. CNBC's own `robots.txt` advertises `sitemap_news.xml`, a
    # rolling ~48h site-wide news sitemap (live-confirmed to include that
    # exact article's publish window); `_CANDIDATE_SITEMAP_PATHS` now tries
    # it as a fallback path, so enabling this same sitemap-scavenging block
    # for CNBC specifically (gated purely by `source.domain`, not a new
    # mechanism) recovers this class of miss. The existing topical ranking
    # + `max_candidates` truncation below still applies, so non-AI/finance
    # CNBC stories the sitemap also lists (sports, lifestyle, ...) are
    # ranked below genuinely relevant ones, not dumped into triage
    # unfiltered.
    if (
        (source.category == SourceCategory.ANALYST or source.domain == "cnbc.com")
        and len(fetch_failures) < len(source.urls)
    ):
        sitemap_entries = discover_via_sitemap(
            domain=source.domain, source_name=source.name, http_client=http_client,
            source_netlocs={urlparse(u).netloc for u in source.urls}, cache_policy=cache_policy,
        )
        for entry in sitemap_entries:
            key = _canonical(entry.url)
            if key in seen:
                continue
            if not _is_relevant_candidate(urlparse(entry.url).path, entry.title, in_scope=True):
                continue
            seen.add(key)
            candidates.append(entry)

    if topic_keywords:
        candidates.sort(key=lambda c: _topical_rank_key(c, topic_keywords), reverse=True)

    # Observability fix (forensic finding, CURRENT run `31fa192e4d7a7175`):
    # this truncation previously had no visible trace anywhere -- a source
    # whose own page(s) listed more on-topic links than `max_candidates`
    # silently lost the weakest-ranked excess with nothing to show it ever
    # happened, making "the source under-covered its own topic page" and
    # "max_candidates_per_source is genuinely too small for a high-volume
    # day" indistinguishable without a live re-fetch. Never changes which
    # candidates are kept (still topic-ranked, strongest first, exactly as
    # before) -- purely additive visibility into HOW MANY were discarded
    # and by what cap.
    if len(candidates) > max_candidates:
        logger.warning(
            "discover_candidates: %s found %d candidate link(s) across its configured page(s) -- "
            "truncating to max_candidates_per_source=%d (topic-ranked, strongest first); "
            "%d discarded this run",
            source.name, len(candidates), max_candidates, len(candidates) - max_candidates,
        )

    return candidates[:max_candidates]


def _discover_links_from_html(
    html: str, *, base_url: str, source: SourceDefinition,
    source_netlocs: Set[str], source_urls_canonical: Set[str],
) -> List[CandidateURL]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as exc:
        logger.warning("Could not parse HTML for %s: %s", source.name, exc)
        return []

    now = datetime.now(timezone.utc)
    candidates: List[CandidateURL] = []
    seen: Set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith("#") or href.lower().startswith(("javascript:", "mailto:", "tel:")):
            continue

        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        if source.domain not in parsed.netloc:
            continue

        key = _canonical(absolute)
        if key in source_urls_canonical:
            # The page self-links back to its own configured hub/index URL
            # (e.g. an active nav tab or breadcrumb) -- that URL is a
            # discovery entry point, not a discovered article, and must
            # never be promoted to a retrievable/extractable candidate in
            # its own right (it would otherwise become an "Article" whose
            # content is the whole hub page, not a single news item).
            continue

        # PR Newswire-specific override (see `_is_pr_newswire_release_url`'s
        # docstring): the generic keep-keyword filter below cannot
        # distinguish a real release from this domain's own category/topic
        # listing pages, so for this one domain the numeric-release-ID URL
        # shape is the deciding rule instead -- never applied to any other
        # source's links.
        if source.domain == "prnewswire.com" and not _is_pr_newswire_release_url(parsed.path):
            continue

        # Financial Times-specific override (forensic finding, CURRENT run
        # `e818e08adc2e4dd1`): FT's topic/section/navigation pages
        # ("/war-in-ukraine", "/the-big-read", "/middle-east-north-africa",
        # the configured hub pages themselves, ...) reached triage as
        # candidates and were assigned a fabricated "today" publication
        # date, wasting 5 of that run's 7 triage slots. Reuses the exact
        # same `/content/<uuid>` shape rule `retrieval.ft_search` already
        # proved, via extensive live inspection, is the ONLY shape FT's own
        # individual articles ever use -- never a blanket ft.com exclusion
        # (a genuine `/content/<uuid>` article is always kept), and never
        # keyed to a specific topic/section slug name, so it also catches
        # any FT topic page not yet observed.
        if source.domain == "ft.com" and not _is_ft_article_url(parsed.path):
            continue

        title = anchor.get_text(strip=True) or None
        in_scope = parsed.netloc in source_netlocs
        if not _is_relevant_candidate(parsed.path, title, in_scope=in_scope):
            continue

        if key in seen:
            continue
        seen.add(key)

        candidates.append(
            CandidateURL(
                url=absolute,
                source_name=source.name,
                domain=source.domain,
                discovery_method=DiscoveryMethod.HTML_LINKS,
                discovered_at=now,
                title=title,
            )
        )

    return candidates


def _is_relevant_candidate(path: str, anchor_text: Optional[str], *, in_scope: bool = True) -> bool:
    """Cheap, deterministic pre-filter: does this URL plausibly point at
    research/news/announcement content, as opposed to navigation, account,
    marketing/commerce, corporate-directory, or reference/utility content?

    `in_scope` is True when the link stays on the same netloc as one of the
    source's own configured URLs (see `discover_candidates`); it defaults to
    True so callers with no source context (e.g. direct unit tests) get the
    original, purely keyword-based behavior.

    Precedence is deliberately conservative:
    1. A locale/country-selector homepage link (e.g. "/au/en/home.html") is
       rejected outright, regardless of any keyword match -- see
       `_LOCALE_HOMEPAGE_RE`.
    2. A short practice-area/taxonomy hub page ("/capabilities/ai-agents",
       "/industries/banking") is rejected outright, same precedence as a
       locale homepage -- see `_is_short_taxonomy_hub_page`.
    3. A reject-keyword hit is final (never overridden by a keep keyword
       appearing elsewhere in the same path, e.g. ".../newsroom/executive-bios"
       is rejected despite "newsroom" also being a keep keyword).
    4. A keep-keyword hit on the URL path always keeps it, in-scope or not.
    5. A bare, single-segment, non-descriptive path (e.g. "/microsoft-365")
       reads as a brand/product landing page rather than an article, unless
       the link's own anchor text signals news/research content.
    6. An in-scope, keyword-free path is ambiguous and is kept rather than
       aggressively filtered.
    7. An out-of-scope, keyword-free path (a different subdomain/section of
       the same domain than the source's own declared research area) is
       rejected — without a keep signal, there is no reason to trust that a
       link into an unrelated part of the domain is research content.
    """
    lowered_path = path.lower()
    if lowered_path.endswith(_BINARY_URL_EXTENSIONS):
        return False  # PDF/image/media asset -- never a candidate, regardless of path shape
    if lowered_path.endswith(_FEED_URL_EXTENSIONS):
        return False  # RSS/Atom/XML feed endpoint -- never an article in its own right

    segments = [seg for seg in path.lower().replace("_", "-").split("/") if seg]
    if not segments:
        return False  # bare domain / homepage link — not an article

    if _LOCALE_HOMEPAGE_RE.match(path.lower()):
        return False  # country/language selector homepage -- never a content candidate

    if _is_short_taxonomy_hub_page(segments):
        return False  # practice-area/taxonomy category page -- never a content candidate

    joined = "-".join(segments)
    if _contains_phrase(joined, _REJECT_KEYWORDS):
        return False
    if segments and all(seg in _HUB_INDEX_SEGMENTS for seg in segments):
        return False  # bare index/listing root (e.g. "/news", "/research/index") -- not a specific article
    if _contains_phrase(joined, _KEEP_KEYWORDS):
        return True

    if _looks_like_bare_product_page(segments):
        if anchor_text and _contains_phrase(_normalize_text(anchor_text), _STRONG_KEEP_KEYWORDS):
            return True
        return False

    return in_scope  # ambiguous path shape: keep only within the source's own declared scope


def _looks_like_bare_product_page(segments: List[str]) -> bool:
    """A single, short path segment (e.g. "/microsoft-365", "/xbox") reads as
    a top-level brand/product page. Genuine articles are either nested under
    a news/blog/press-shaped path (caught by `_KEEP_KEYWORDS` above) or have
    a long, headline-like slug (more than two hyphen-separated words) — so
    this only fires for exactly one short segment, to avoid over-rejecting
    ambiguous content."""
    if len(segments) != 1:
        return False
    tokens = [t for t in segments[0].split("-") if t]
    return len(tokens) <= 2


def _contains_phrase(joined: str, keywords: Set[str]) -> bool:
    """True if any keyword (single word or hyphenated phrase) appears in
    `joined` as a whole hyphen-delimited word sequence — never as a partial
    match inside a longer word (e.g. keyword "news" must not match inside
    "newsroom")."""
    padded = f"-{joined}-"
    return any(f"-{keyword}-" in padded for keyword in keywords)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _canonical(url: str) -> str:
    without_fragment, _ = urldefrag(url)
    return without_fragment.rstrip("/")
