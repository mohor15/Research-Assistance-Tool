"""CachePolicy (CURRENT-freshness fix): `PageCache` has no TTL, so a page
fetched once would otherwise be reused forever -- incompatible with
CURRENT's `[now-24h, now)` contract. `CachePolicy.FRESH` never reads a
cached page but still writes the live result afterward; `CachePolicy.CACHE`
(the default, used by Morning/Evening/manual) is byte-for-byte the
pre-existing behavior."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.retrieval.cache import PageCache
from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.models import CachePolicy, RetrievalMethod, RetrievedPage

from tests.retrieval.fakes import FakeResponse, FakeSession


def _stale_cached_page(url: str, *, days_old: int) -> RetrievedPage:
    return RetrievedPage(
        url=url, final_url=url, status_code=200,
        retrieved_at=datetime(2026, 9, 29, tzinfo=timezone.utc),  # several days before "now" in these tests
        method=RetrievalMethod.HTTP, html="<html>stale cached snapshot</html>", success=True,
    )


def test_a_current_fresh_policy_bypasses_a_stale_cache_entry(tmp_path):
    url = "https://example.com/newsroom"
    cache = PageCache(cache_dir=tmp_path, enabled=True)
    cache.set(url, _stale_cached_page(url, days_old=5))

    session = FakeSession([FakeResponse(status_code=200, url=url, text="<html>fresh live page</html>")])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session, cache=cache)

    page = client.get(url, cache_policy=CachePolicy.FRESH)

    assert page.html == "<html>fresh live page</html>"
    assert page.from_cache is False
    assert len(session.calls) == 1  # a live fetch genuinely happened, not a cache hit


def test_b_normal_cache_policy_still_reuses_the_cache(tmp_path):
    url = "https://example.com/newsroom"
    cache = PageCache(cache_dir=tmp_path, enabled=True)
    cache.set(url, _stale_cached_page(url, days_old=5))

    session = FakeSession([])  # no scripted response at all -- a live fetch here would raise
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session, cache=cache)

    page = client.get(url)  # default cache_policy=CachePolicy.CACHE

    assert page.html == "<html>stale cached snapshot</html>"
    assert page.from_cache is True
    assert session.calls == []  # no live fetch was attempted -- the cache was reused


def test_c_fresh_policy_result_is_written_back_to_cache_for_later_cache_policy_runs(tmp_path):
    """FRESH never *reads* the cache, but a Morning/Evening run immediately
    afterward must see the now-refreshed snapshot, not the old one --
    FRESH must not "globally disable caching", only bypass stale reads."""
    url = "https://example.com/newsroom"
    cache = PageCache(cache_dir=tmp_path, enabled=True)
    cache.set(url, _stale_cached_page(url, days_old=5))

    session = FakeSession([FakeResponse(status_code=200, url=url, text="<html>fresh live page</html>")])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session, cache=cache)

    client.get(url, cache_policy=CachePolicy.FRESH)

    cached_now = cache.get(url)
    assert cached_now is not None
    assert cached_now.html == "<html>fresh live page</html>"


def test_fresh_policy_with_no_prior_cache_entry_performs_a_normal_live_fetch(tmp_path):
    url = "https://example.com/newsroom"
    cache = PageCache(cache_dir=tmp_path, enabled=True)

    session = FakeSession([FakeResponse(status_code=200, url=url, text="<html>live</html>")])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session, cache=cache)

    page = client.get(url, cache_policy=CachePolicy.FRESH)

    assert page.success is True
    assert page.html == "<html>live</html>"


def test_fresh_policy_without_any_cache_configured_still_works(tmp_path):
    """A client with no cache at all (cache=None) must behave identically
    under either policy -- FRESH never requires a cache to be present."""
    url = "https://example.com/newsroom"
    session = FakeSession([FakeResponse(status_code=200, url=url, text="<html>live</html>")])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session, cache=None)

    page = client.get(url, cache_policy=CachePolicy.FRESH)

    assert page.success is True
    assert len(session.calls) == 1
