"""Playwright fallback: unavailable-degrades-gracefully, invocation, failure handling.

Playwright is not installed in this sandbox (no network access to fetch
it), so `PLAYWRIGHT_AVAILABLE` is monkeypatched together with a fake
`sync_playwright` to exercise the real orchestration logic in
`fetch_with_browser` without the actual dependency. The
"genuinely unavailable" tests run as-is and need no monkeypatching.
"""

from __future__ import annotations

import pytest

import research_assistant.retrieval.browser as browser_module
from research_assistant.retrieval.exceptions import BrowserUnavailableError
from research_assistant.retrieval.models import RetrievalMethod


def test_browser_reports_unavailable_when_playwright_not_installed(monkeypatch):
    monkeypatch.setattr(browser_module, "PLAYWRIGHT_AVAILABLE", False)
    assert browser_module.is_browser_available() is False


def test_fetch_raises_browser_unavailable_when_playwright_not_installed(monkeypatch):
    monkeypatch.setattr(browser_module, "PLAYWRIGHT_AVAILABLE", False)
    with pytest.raises(BrowserUnavailableError):
        browser_module.fetch_with_browser("https://example.com", timeout_seconds=5, user_agent="TestBot")


class _FakeResponse:
    def __init__(self, status: int):
        self.status = status


class _FakePage:
    def __init__(self, html: str, final_url: str, status: int = 200):
        self._html = html
        self.url = final_url
        self._status = status
        self.goto_calls = []

    def goto(self, url, timeout=None, wait_until=None):
        self.goto_calls.append((url, timeout, wait_until))
        return _FakeResponse(self._status)

    def content(self):
        return self._html


class _FakeBrowser:
    def __init__(self, page: _FakePage):
        self._page = page
        self.closed = False

    def new_page(self, user_agent=None):
        return self._page

    def close(self):
        self.closed = True


class _FakeChromium:
    def __init__(self, browser: _FakeBrowser, *, fail_launch=False):
        self._browser = browser
        self._fail_launch = fail_launch

    def launch(self):
        if self._fail_launch:
            raise RuntimeError("Executable doesn't exist at ...")
        return self._browser


class _FakePlaywrightContext:
    def __init__(self, chromium: _FakeChromium):
        self.chromium = chromium

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_fetch_with_browser_renders_page_and_returns_retrieved_page(monkeypatch):
    fake_page = _FakePage(html="<html>rendered</html>", final_url="https://example.com/final")
    fake_browser = _FakeBrowser(fake_page)
    fake_chromium = _FakeChromium(fake_browser)

    monkeypatch.setattr(browser_module, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(
        browser_module, "sync_playwright", lambda: _FakePlaywrightContext(fake_chromium)
    )

    page = browser_module.fetch_with_browser("https://example.com", timeout_seconds=5, user_agent="TestBot")

    assert page.success is True
    assert page.method == RetrievalMethod.BROWSER
    assert page.html == "<html>rendered</html>"
    assert page.final_url == "https://example.com/final"
    assert fake_browser.closed is True
    assert fake_page.goto_calls[0][0] == "https://example.com"


def test_fetch_with_browser_raises_unavailable_when_browser_binary_missing(monkeypatch):
    fake_chromium = _FakeChromium(browser=None, fail_launch=True)

    monkeypatch.setattr(browser_module, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(
        browser_module, "sync_playwright", lambda: _FakePlaywrightContext(fake_chromium)
    )

    with pytest.raises(BrowserUnavailableError):
        browser_module.fetch_with_browser("https://example.com", timeout_seconds=5, user_agent="TestBot")


def test_fetch_with_browser_returns_failed_page_on_navigation_error(monkeypatch):
    class _FailingPage(_FakePage):
        def goto(self, url, timeout=None, wait_until=None):
            raise RuntimeError("navigation timeout")

    fake_browser = _FakeBrowser(_FailingPage(html="", final_url="https://example.com"))
    fake_chromium = _FakeChromium(fake_browser)

    monkeypatch.setattr(browser_module, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(
        browser_module, "sync_playwright", lambda: _FakePlaywrightContext(fake_chromium)
    )

    # A navigation failure is data, not an exception -- the browser mechanism
    # itself worked fine, this particular page just didn't render -- so
    # callers (RetrievalService._try_browser_fallback) can inspect/classify
    # it exactly like an HTTP-path failure, rather than catching an exception.
    page = browser_module.fetch_with_browser("https://example.com", timeout_seconds=5, user_agent="TestBot")

    assert page.success is False
    assert page.failure.reason == "browser_retrieval_failed"
    assert fake_browser.closed is True  # cleaned up even on failure


def test_fetch_with_browser_classifies_http2_protocol_errors(monkeypatch):
    class _FailingPage(_FakePage):
        def goto(self, url, timeout=None, wait_until=None):
            raise RuntimeError("net::ERR_HTTP2_PROTOCOL_ERROR")

    fake_browser = _FakeBrowser(_FailingPage(html="", final_url="https://example.com"))
    fake_chromium = _FakeChromium(fake_browser)

    monkeypatch.setattr(browser_module, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(
        browser_module, "sync_playwright", lambda: _FakePlaywrightContext(fake_chromium)
    )

    page = browser_module.fetch_with_browser("https://example.com", timeout_seconds=5, user_agent="TestBot")

    assert page.success is False
    assert page.failure.reason == "browser_http2_error"


def test_fetch_with_browser_reports_non_ok_status_as_failure(monkeypatch):
    fake_page = _FakePage(html="<html>blocked</html>", final_url="https://example.com", status=401)
    fake_browser = _FakeBrowser(fake_page)
    fake_chromium = _FakeChromium(fake_browser)

    monkeypatch.setattr(browser_module, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(
        browser_module, "sync_playwright", lambda: _FakePlaywrightContext(fake_chromium)
    )

    page = browser_module.fetch_with_browser("https://example.com", timeout_seconds=5, user_agent="TestBot")

    assert page.success is False
    assert page.status_code == 401
    assert page.failure.reason == "http_error"
    assert page.failure.detail == "401"
