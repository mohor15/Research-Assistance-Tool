"""A reusable, retrying HTTP client for the retrieval engine.

This is the *only* module that should issue plain HTTP GETs for retrieval
purposes. It composes robots.txt checking, per-domain rate limiting, and
page caching around a plain `requests` session, and always returns a
`RetrievedPage` rather than raising — network problems are data, not
exceptions, so one bad source never aborts a retrieval run.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional
from urllib.parse import urlparse

import requests

from research_assistant.retrieval.cache import PageCache
from research_assistant.retrieval.models import CachePolicy, RetrievalFailure, RetrievalMethod, RetrievedPage
from research_assistant.retrieval.rate_limiter import DomainRateLimiter
from research_assistant.retrieval.robots import RobotsChecker

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "NomuraResearchAssistant/0.1 (+local research bot; non-commercial internal research use)"
)
_SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie"}


def _sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _SENSITIVE_HEADERS}


class HTTPClient:
    def __init__(
        self,
        *,
        timeout_seconds: float,
        max_retries: int,
        retry_delay_seconds: float,
        user_agent: str = DEFAULT_USER_AGENT,
        robots_checker: Optional[RobotsChecker] = None,
        rate_limiter: Optional[DomainRateLimiter] = None,
        cache: Optional[PageCache] = None,
        session: Optional[requests.Session] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._retry_delay = retry_delay_seconds
        self._user_agent = user_agent
        self._robots = robots_checker
        self._rate_limiter = rate_limiter
        self._cache = cache
        self._session = session or requests.Session()
        self._sleep = sleep_fn

    @property
    def robots(self) -> Optional[RobotsChecker]:
        """Exposed so a narrow, non-HTTPClient retrieval path (e.g.
        `retrieval.ft_search`'s Crawl4AI/Playwright tier, which never goes
        through `HTTPClient.get()` and so would otherwise never consult
        robots.txt at all) can still check robots.txt before using a
        browser -- never bypassing it just because a different fetch
        mechanism is involved."""
        return self._robots

    def get(self, url: str, *, cache_policy: CachePolicy = CachePolicy.CACHE) -> RetrievedPage:
        # FRESH never reads a cached page (see CachePolicy's docstring --
        # PageCache has no TTL, so a stale entry would otherwise be reused
        # forever), but a successful live fetch is still written to cache
        # below exactly as with CACHE, refreshing it for later CACHE-policy
        # callers too.
        cached = self._cache.get(url) if (self._cache and cache_policy == CachePolicy.CACHE) else None
        if cached is not None:
            logger.debug("Cache hit for %s", url)
            return cached

        if self._robots is not None and not self._robots.is_allowed(url):
            return RetrievedPage(
                url=url,
                final_url=url,
                retrieved_at=_now(),
                method=RetrievalMethod.HTTP,
                success=False,
                failure=RetrievalFailure(reason="robots_disallowed"),
            )

        if self._rate_limiter is not None:
            self._rate_limiter.wait(urlparse(url).netloc)

        page = self._get_with_retries(url)
        if page.success and self._cache is not None:
            self._cache.set(url, page)
        return page

    def _get_with_retries(self, url: str) -> RetrievedPage:
        total_attempts = self._max_retries + 1
        last_failure = RetrievalFailure(reason="unknown_error")

        for attempt in range(1, total_attempts + 1):
            try:
                response = self._session.get(
                    url,
                    timeout=self._timeout,
                    headers={"User-Agent": self._user_agent},
                    allow_redirects=True,
                )
            except requests.Timeout as exc:
                last_failure = RetrievalFailure(reason="timeout", detail=str(exc))
                logger.warning("Timeout fetching %s (attempt %d/%d)", url, attempt, total_attempts)
            except requests.exceptions.SSLError as exc:
                # Distinguished from a generic connection_error: this is a
                # local trust-store/certificate problem (e.g. TLS-intercepting
                # antivirus/proxy software), not the remote site being down --
                # see research_assistant/__init__.py's truststore injection.
                last_failure = RetrievalFailure(reason="tls_certificate_error", detail=str(exc))
                logger.warning(
                    "TLS certificate error fetching %s (attempt %d/%d): %s", url, attempt, total_attempts, exc
                )
            except requests.RequestException as exc:
                last_failure = RetrievalFailure(reason="connection_error", detail=str(exc))
                logger.warning(
                    "Connection error fetching %s (attempt %d/%d): %s", url, attempt, total_attempts, exc
                )
            else:
                if response.status_code >= 500 and attempt < total_attempts:
                    last_failure = RetrievalFailure(reason="server_error", detail=str(response.status_code))
                    logger.info(
                        "Server error %s for %s (attempt %d/%d), retrying",
                        response.status_code, url, attempt, total_attempts,
                    )
                else:
                    return self._page_from_response(url, response)

            if attempt < total_attempts:
                self._sleep(self._retry_delay)

        return RetrievedPage(
            url=url,
            final_url=url,
            retrieved_at=_now(),
            method=RetrievalMethod.HTTP,
            success=False,
            failure=last_failure,
        )

    def _page_from_response(self, url: str, response: requests.Response) -> RetrievedPage:
        content_type = response.headers.get("Content-Type")
        return RetrievedPage(
            url=url,
            final_url=response.url,
            status_code=response.status_code,
            retrieved_at=_now(),
            method=RetrievalMethod.HTTP,
            content_type=content_type,
            html=response.text,
            headers=_sanitize_headers(dict(response.headers)),
            success=response.ok,
            failure=None if response.ok else RetrievalFailure(reason="http_error", detail=str(response.status_code)),
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)
