"""Optional Playwright-based fallback for pages that genuinely need JS rendering.

Only invoked by `RetrievalService` when plain HTTP retrieval/extraction was
insufficient *and* `retrieval.browser_fallback` is enabled — never for every
URL. Playwright (and its browser binaries) are optional: if either is
missing, `fetch_with_browser` raises `BrowserUnavailableError` so the caller
can fall back to the plain HTTP result instead of crashing the pipeline.

This module never attempts to bypass logins, paywalls, CAPTCHAs, or other
access controls — it renders the page a normal (anonymous) browser would see.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from research_assistant.retrieval.exceptions import BrowserUnavailableError
from research_assistant.retrieval.models import RetrievalFailure, RetrievalMethod, RetrievedPage

logger = logging.getLogger(__name__)

try:
    from playwright.sync_api import sync_playwright  # type: ignore

    PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when playwright is absent
    sync_playwright = None  # keeps the name patchable/importable even when playwright isn't installed
    PLAYWRIGHT_AVAILABLE = False


def is_browser_available() -> bool:
    return PLAYWRIGHT_AVAILABLE


def fetch_with_browser(url: str, *, timeout_seconds: float, user_agent: str) -> RetrievedPage:
    if not PLAYWRIGHT_AVAILABLE:
        raise BrowserUnavailableError("Playwright is not installed")

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as exc:
            # Playwright installed but `playwright install` (browser binaries) was never run.
            raise BrowserUnavailableError(f"Could not launch a browser: {exc}") from exc

        try:
            page = browser.new_page(user_agent=user_agent)
            try:
                response = page.goto(url, timeout=timeout_seconds * 1000, wait_until="domcontentloaded")
            except Exception as exc:
                # Browser launched fine; this particular navigation failed --
                # categorically different from BrowserUnavailableError (the
                # mechanism itself is fine), so this is returned as data
                # (a failed RetrievedPage) rather than raised, exactly like
                # every other retrieval-layer failure.
                upper = str(exc).upper()
                reason = "browser_http2_error" if "HTTP2" in upper or "PROTOCOL_ERROR" in upper else "browser_retrieval_failed"
                return RetrievedPage(
                    url=url,
                    final_url=url,
                    retrieved_at=datetime.now(timezone.utc),
                    method=RetrievalMethod.BROWSER,
                    success=False,
                    failure=RetrievalFailure(reason=reason, detail=str(exc)),
                )
            html = page.content()
            final_url = page.url
            status_code = response.status if response is not None else 200
        finally:
            browser.close()

    success = 200 <= status_code < 400
    return RetrievedPage(
        url=url,
        final_url=final_url,
        status_code=status_code,
        retrieved_at=datetime.now(timezone.utc),
        method=RetrievalMethod.BROWSER,
        content_type="text/html",
        html=html,
        success=success,
        failure=None if success else RetrievalFailure(reason="http_error", detail=str(status_code)),
    )
