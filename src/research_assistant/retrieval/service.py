"""The single entry point later phases should use to obtain content.

Agents/pipeline code calls `RetrievalService.retrieve_source(source)` (or
the lower-level `discover_candidates` / `retrieve` / `extract`) and never
touches RSS parsing, HTTP retries, robots.txt, caching, or Playwright
directly — those are implementation details that can change without
breaking agents. This mirrors the mandated architecture:

    Peer Agent -> Retrieval Service -> Source Registry -> HTTP / RSS / Browser
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse

from research_assistant.config.schema import AppConfig
from research_assistant.retrieval.browser import fetch_with_browser, is_browser_available
from research_assistant.retrieval.cache import PageCache
from research_assistant.retrieval.classification import classify
from research_assistant.retrieval.client import DEFAULT_USER_AGENT, HTTPClient
from research_assistant.retrieval.discovery import _BINARY_URL_EXTENSIONS
from research_assistant.retrieval.discovery import _topical_rank_key
from research_assistant.retrieval.discovery import discover_candidates as _discover_candidates
from research_assistant.retrieval.discovery import discover_candidates_from_page
from research_assistant.retrieval.exceptions import BrowserUnavailableError, SourceUnreachableError
from research_assistant.retrieval.extractor import extract_article
from research_assistant.retrieval.models import (
    CachePolicy,
    CandidateURL,
    ExtractedArticle,
    ExtractionStatus,
    RetrievalFailure,
    RetrievedPage,
    SourceCategory,
    SourceDefinition,
)
from research_assistant.retrieval.rate_limiter import DomainRateLimiter
from research_assistant.retrieval.robots import RobotsChecker
from research_assistant.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

# Registry sections that carry a fixed URL list to scrape. `startups` is
# intentionally excluded: per config/sources.yaml it is discovery-only
# (an optional discovery provider, out of scope for Phase 1) and has no
# per-company URLs to crawl yet.
_SCRAPABLE_CATEGORIES = {
    SourceCategory.PEER: "peers",
    SourceCategory.AI_PROVIDER: "ai_providers",
    SourceCategory.ANALYST: "analysts",
    SourceCategory.SECONDARY: "secondary_sources",
}

# HTTP statuses that mean "this resource definitively does not exist" --
# browser rendering cannot recover content that was never there, so these
# are excluded from browser fallback eligibility (see `_try_browser_fallback`).
_NON_RECOVERABLE_HTTP_STATUSES = {"404", "410"}

# A URL whose path is already known, by shape, to be a feed/binary/non-HTML
# resource is never worth a browser retry: rendering a PDF or an RSS/Atom
# feed in a JS-capable browser cannot turn it into usable article HTML, so
# this would just cost a browser launch for a result we can already rule
# out. Reuses the project's one existing binary-extension list rather than
# inventing a second one (see `discovery._BINARY_URL_EXTENSIONS`).
_NON_BROWSER_URL_EXTENSIONS = _BINARY_URL_EXTENSIONS + (".xml", ".rss")

# Yahoo Finance coverage fix (confirmed real, live-verified): every page on
# this domain (its topic hub pages, its general /news/ listing, ...) is a
# client-side-rendered React shell whose plain HTML contains ONLY site nav
# (dozens of "/topic/<x>/", "/markets/<x>/" category links) and ZERO real
# article links -- confirmed by direct inspection, not a guess. The existing
# "zero candidates -> browser retry" fallback below never fires here because
# the nav noise itself still counts as "candidates" (246 found on one probe,
# all nav). Browser-rendering the same URL DOES surface real, on-topic
# article links (e.g. "/markets/stocks/article/tsmc-posts-record-profit-
# amid-extremely-robust-demand-for-ai-chips-<id>.html") -- confirmed live --
# because they only exist in the post-JS DOM. Scoped purely to this one
# domain, exactly like the existing per-domain overrides elsewhere in this
# module (FT search, NatWest content-hub, CNBC sitemap scavenging): no other
# source's behavior changes.
_YAHOO_FINANCE_DOMAIN = "finance.yahoo.com"


def _is_browser_fallback_eligible(url: str, failure: Optional["RetrievalFailure"]) -> bool:
    if urlparse(url).path.lower().endswith(_NON_BROWSER_URL_EXTENSIONS):
        return False
    if failure is None:
        return True
    if failure.reason == "robots_disallowed":
        return False  # never bypass robots.txt via the browser fallback either
    if failure.reason == "http_error" and failure.detail in _NON_RECOVERABLE_HTTP_STATUSES:
        # A definitive "page does not exist" response (404/410) can never be
        # recovered by rendering -- only transient/blocked/JS-dependent
        # failures (timeout, connection_error, 5xx, other 4xx such as
        # anti-bot 401/403s) are plausibly worth a browser retry.
        return False
    return True


def _flatten_topic_keywords(config: AppConfig) -> List[str]:
    """The existing `config/topics.yaml` AI + financial-services vocabulary
    (not a new/invented keyword list -- the same one `agents.base`'s
    pre-filter and `pipeline.nodes`' candidate ranking already use), reused
    here purely to RANK discovery candidates before `max_candidates`
    truncates them (see `discovery.discover_candidates`) -- never to reject
    one outright at this stage."""
    topics = config.topics
    buckets = [
        topics.ai_topics.core, topics.ai_topics.models, topics.ai_topics.infrastructure,
        topics.ai_topics.enterprise, topics.ai_topics.governance,
        topics.financial_services.core, topics.financial_services.markets,
        topics.financial_services.wealth, topics.financial_services.risk, topics.financial_services.operations,
    ]
    return [keyword.lower() for bucket in buckets for keyword in bucket]


def build_source_definitions(config: AppConfig) -> List[SourceDefinition]:
    """Flatten config.sources into the retrieval engine's SourceDefinition list."""
    sources: List[SourceDefinition] = []
    for category, field_name in _SCRAPABLE_CATEGORIES.items():
        for entry in getattr(config.sources, field_name):
            sources.append(
                SourceDefinition(
                    name=entry.name,
                    domain=entry.domain,
                    urls=entry.urls,
                    source_type=entry.source_type,
                    priority=entry.priority,
                    category=category,
                    search=entry.search,
                )
            )
    return sources


class RetrievalService:
    def __init__(
        self,
        config: AppConfig,
        *,
        http_client: Optional[HTTPClient] = None,
        cache: Optional[PageCache] = None,
        robots_checker: Optional[RobotsChecker] = None,
        rate_limiter: Optional[DomainRateLimiter] = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._config = config
        self._user_agent = user_agent
        retrieval_settings = config.settings.retrieval

        self._robots = robots_checker or RobotsChecker(
            enabled=retrieval_settings.respect_robots_txt, user_agent=user_agent
        )
        self._rate_limiter = rate_limiter
        if self._rate_limiter is None and retrieval_settings.respect_rate_limits:
            self._rate_limiter = DomainRateLimiter(min_delay_seconds=retrieval_settings.retry_delay_seconds)
        self._cache = cache or PageCache(cache_dir=DATA_DIR / "cache", enabled=retrieval_settings.cache_pages)
        self._http_client = http_client or HTTPClient(
            timeout_seconds=retrieval_settings.request_timeout_seconds,
            max_retries=retrieval_settings.max_retries,
            retry_delay_seconds=retrieval_settings.retry_delay_seconds,
            user_agent=user_agent,
            robots_checker=self._robots,
            rate_limiter=self._rate_limiter,
            cache=self._cache,
        )
        self._sources = build_source_definitions(config)
        self._topic_keywords = _flatten_topic_keywords(config)

    def list_sources(self, category: Optional[SourceCategory] = None) -> List[SourceDefinition]:
        if category is None:
            return list(self._sources)
        return [s for s in self._sources if s.category == category]

    def discover_candidates(
        self, source: SourceDefinition, *, cache_policy: CachePolicy = CachePolicy.CACHE
    ) -> List[CandidateURL]:
        max_candidates = self._config.settings.retrieval.max_candidates_per_source
        try:
            candidates = _discover_candidates(
                source, http_client=self._http_client, max_candidates=max_candidates,
                topic_keywords=self._topic_keywords, cache_policy=cache_policy,
            )
        except SourceUnreachableError as exc:
            # Bug fix: previously this exception propagated straight out of
            # discover_candidates() and the browser-fallback branch below was
            # never reached at all -- so a source whose plain HTTP fetch
            # failed outright (TLS error, timeout, 401/403, ...) never got a
            # browser retry, even though browser_fallback exists exactly for
            # this. A fetch that returned zero candidates (the branch below)
            # always got the retry; a fetch that *failed* never did. Fixed
            # by attempting the same browser-rendered discovery here, then
            # re-raising (now classified) only if that also comes up empty.
            if not self._config.settings.retrieval.browser_fallback:
                raise
            candidates, browser_failures = self._try_browser_rendered_discovery(source, max_candidates)
            if candidates:
                return candidates
            # Browser fallback didn't recover anything either -- re-raise so
            # the existing diagnostic/metrics behavior in
            # pipeline.nodes.retrieve() is preserved. `failures` carries
            # BOTH the original HTTP failures and whatever the browser
            # attempt independently produced, so classification (done once,
            # at the point the message is actually rendered) can see that a
            # browser retry was genuinely attempted and what it found --
            # e.g. a matching 401 from both paths -- rather than only ever
            # seeing the original HTTP-side failure.
            raise SourceUnreachableError(str(exc), failures=exc.failures + browser_failures) from exc

        if not candidates and self._config.settings.retrieval.browser_fallback:
            # The plain HTTP fetch succeeded (otherwise SourceUnreachableError
            # would already have been raised) but found zero candidate links
            # at all -- a narrow, source-level signal that the page is a
            # client-side-rendered shell (e.g. Lloyds Banking Group's
            # newsroom page: 200 OK, ~1 raw link in the HTML). This never
            # fires merely because a page returned fewer candidates than
            # hoped, or for any page that already yielded something.
            candidates, _ = self._try_browser_rendered_discovery(source, max_candidates)
        elif source.domain == _YAHOO_FINANCE_DOMAIN and self._config.settings.retrieval.browser_fallback:
            # See `_YAHOO_FINANCE_DOMAIN`'s docstring: the plain-HTTP pass
            # always finds nav-only "candidates" here (never zero), so the
            # branch above never fires -- this source needs browser
            # rendering unconditionally, not just on a zero-candidate signal.
            # The plain-HTTP nav links are discarded in favor of the
            # browser-rendered result; falling back to them would just
            # re-surface the same nav noise this fix exists to avoid.
            browser_candidates, _ = self._try_browser_rendered_discovery(source, max_candidates)
            if browser_candidates:
                candidates = browser_candidates
        return candidates

    def _try_browser_rendered_discovery(
        self, source: SourceDefinition, max_candidates: int
    ) -> tuple[List[CandidateURL], List[RetrievalFailure]]:
        if not is_browser_available():
            logger.info("Discovery found zero candidates for %s; browser rendering unavailable", source.name)
            return [], []

        candidates: List[CandidateURL] = []
        failures: List[RetrievalFailure] = []
        seen = set()
        for source_url in source.urls:
            if not self._robots.is_allowed(source_url):
                continue
            if urlparse(source_url).path.lower().endswith(_NON_BROWSER_URL_EXTENSIONS):
                continue  # RSS/XML/PDF/binary -- rendering it cannot produce usable article HTML
            try:
                page = fetch_with_browser(
                    source_url,
                    timeout_seconds=self._config.settings.retrieval.request_timeout_seconds,
                    user_agent=self._user_agent,
                )
            except BrowserUnavailableError as exc:
                logger.info("Browser-rendered discovery unavailable for %s: %s", source.name, exc)
                continue
            except Exception as exc:
                logger.warning("Browser-rendered discovery failed for %s: %s", source.name, exc)
                failures.append(RetrievalFailure(reason="browser_retrieval_failed", detail=str(exc)))
                continue
            if not page.success or not page.html:
                if page.failure:
                    failures.append(page.failure)
                continue

            logger.info("Browser-rendered discovery recovered content for %s", source.name)
            for entry in discover_candidates_from_page(page, source):
                if entry.url in seen:
                    continue
                seen.add(entry.url)
                candidates.append(entry)

        # Yahoo Finance coverage fix: a rendered page's nav (header/footer
        # links, repeated on every page) can easily number in the hundreds,
        # while the real article teasers it exists to recover appear later
        # in DOM order -- truncating by raw collection order (the previous
        # behavior, harmless for a single-thin-newsroom-page source like
        # Lloyds) silently filled the entire `max_candidates` cap with nav
        # before a single real article was ever considered. Now ranked by
        # the same topic-keyword vocabulary `discover_candidates()` already
        # uses for its own plain-HTTP truncation, so a genuinely on-topic
        # article always outranks a generic nav link regardless of DOM
        # position.
        if self._topic_keywords:
            candidates.sort(key=lambda c: _topical_rank_key(c, self._topic_keywords), reverse=True)
        return candidates[:max_candidates], failures

    def retrieve(
        self, candidate: CandidateURL, *, cache_policy: CachePolicy = CachePolicy.CACHE
    ) -> RetrievedPage:
        page = self._http_client.get(candidate.url, cache_policy=cache_policy)
        if not page.success and self._config.settings.retrieval.browser_fallback:
            page = self._try_browser_fallback(candidate.url, page)
        return page

    def _try_browser_fallback(self, url: str, original: RetrievedPage) -> RetrievedPage:
        if not _is_browser_fallback_eligible(url, original.failure):
            logger.info("Skipping browser fallback for %s: not eligible (%s)", url, original.failure)
            return original
        if not is_browser_available():
            logger.info("Browser fallback requested for %s but Playwright is unavailable", url)
            return original
        if not self._robots.is_allowed(url):
            return original

        try:
            logger.info("Falling back to browser rendering for %s", url)
            browser_page = fetch_with_browser(
                url,
                timeout_seconds=self._config.settings.retrieval.request_timeout_seconds,
                user_agent=self._user_agent,
            )
        except BrowserUnavailableError as exc:
            logger.info("Browser fallback unavailable for %s: %s", url, exc)
            return original
        except Exception as exc:
            logger.warning("Browser fallback failed for %s: %s", url, exc)
            return original

        if (
            not browser_page.success
            and browser_page.failure
            and browser_page.failure.reason == "http_error"
            and original.failure
            and original.failure.reason == "http_error"
            and browser_page.failure.detail == original.failure.detail
            and original.failure.detail in {"401", "403"}
        ):
            # The HTTP client AND a real rendered browser independently got
            # the same 401/403 for this exact URL -- that's confirmed
            # anti-bot/access-control, not a transient failure a retry could
            # still recover from. Reclassified so downstream diagnostics
            # never call this "connection_error" or a generic HTTP failure.
            browser_page = browser_page.model_copy(
                update={
                    "failure": RetrievalFailure(
                        reason="anti_bot_access_blocked", detail=browser_page.failure.detail
                    )
                }
            )
        return browser_page

    def extract(self, page: RetrievedPage) -> ExtractedArticle:
        if not page.success or not page.html:
            reason = page.failure.reason if page.failure else "no_content"
            # Discovery (the candidate URL/title/date) and content retrieval
            # are separate outcomes -- this ExtractedArticle is still
            # returned (never dropped) so the already-discovered candidate
            # survives into Article/persistence with no clean_text, which
            # `evidence.conversion.article_to_evidence` already refuses to
            # turn into Evidence. `content_unavailable_reason` is the
            # honest classification label (ROBOTS_BLOCKED, HTTP_403,
            # ANTI_BOT_ACCESS_BLOCKED, CONNECTION_ERROR, ...) for
            # observability (Scan Explorer) -- purely descriptive, never
            # read by any gating logic.
            classification = classify(page.failure) if page.failure else None
            return ExtractedArticle(
                url=page.url, status=ExtractionStatus.FAILED,
                metadata={"reason": reason, "content_unavailable_reason": classification},
            )
        return extract_article(page.html, page.url, content_type=page.content_type)

    def retrieve_source(
        self, source: SourceDefinition, *, cache_policy: CachePolicy = CachePolicy.CACHE
    ) -> List[ExtractedArticle]:
        max_articles = self._config.settings.retrieval.max_articles_per_source
        candidates = self.discover_candidates(source, cache_policy=cache_policy)

        articles: List[ExtractedArticle] = []
        for candidate in candidates:
            if len(articles) >= max_articles:
                break
            page = self.retrieve(candidate, cache_policy=cache_policy)
            article = self.extract(page)
            articles.append(_backfill_from_candidate(article, candidate, source))
        return articles

    def retrieve_all(
        self, category: Optional[SourceCategory] = None, *, cache_policy: CachePolicy = CachePolicy.CACHE
    ) -> Dict[str, List[ExtractedArticle]]:
        results: Dict[str, List[ExtractedArticle]] = {}
        for source in self.list_sources(category):
            try:
                results[source.name] = self.retrieve_source(source, cache_policy=cache_policy)
            except Exception as exc:
                logger.error("Retrieval failed entirely for source %s: %s", source.name, exc)
                results[source.name] = []
        return results


def _backfill_from_candidate(
    article: ExtractedArticle, candidate: CandidateURL, source: SourceDefinition
) -> ExtractedArticle:
    """Fill gaps in extraction (common on JS-heavy pages) with feed/link metadata."""
    updates = {}
    if article.published_at is None and candidate.published_at is not None:
        updates["published_at"] = candidate.published_at
    if not article.title and candidate.title:
        updates["title"] = candidate.title
    if not article.publisher:
        updates["publisher"] = source.name
    if article.status == ExtractionStatus.FAILED and not article.text and candidate.feed_summary:
        # The article page itself was blocked/unreachable (paywall,
        # anti-bot, ...), but the source's own RSS/Atom feed already
        # publicly syndicated a summary for this exact URL. Using that
        # summary as degraded evidence is not a paywall/anti-bot bypass --
        # it's content the publisher already chose to publish openly; it
        # still has to clear the normal substantive-body/evidence gates
        # downstream like anything else, so a too-short summary is
        # correctly excluded rather than force-fed through as if it were a
        # full article.
        updates["text"] = candidate.feed_summary
        updates["status"] = ExtractionStatus.PARTIAL
        updates["metadata"] = {**article.metadata, "text_source": "feed_summary"}
    return article.model_copy(update=updates) if updates else article
