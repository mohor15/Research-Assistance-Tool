"""Page cache: miss, hit, disabled, sensitive headers never written to disk."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.retrieval.cache import PageCache
from research_assistant.retrieval.models import RetrievalMethod, RetrievedPage


def _page(url: str) -> RetrievedPage:
    return RetrievedPage(
        url=url,
        final_url=url,
        status_code=200,
        retrieved_at=datetime.now(timezone.utc),
        method=RetrievalMethod.HTTP,
        content_type="text/html",
        html="<html>content</html>",
        headers={"Content-Type": "text/html", "Set-Cookie": "session=abc123"},
        success=True,
    )


def test_cache_miss_returns_none(tmp_path):
    cache = PageCache(cache_dir=tmp_path, enabled=True)
    assert cache.get("https://example.com/a") is None


def test_cache_hit_returns_stored_page_marked_from_cache(tmp_path):
    cache = PageCache(cache_dir=tmp_path, enabled=True)
    page = _page("https://example.com/a")

    cache.set("https://example.com/a", page)
    cached = cache.get("https://example.com/a")

    assert cached is not None
    assert cached.html == page.html
    assert cached.from_cache is True


def test_disabled_cache_never_persists_or_returns_anything(tmp_path):
    cache = PageCache(cache_dir=tmp_path, enabled=False)
    page = _page("https://example.com/a")

    cache.set("https://example.com/a", page)

    assert cache.get("https://example.com/a") is None
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_different_urls_get_different_cache_entries(tmp_path):
    cache = PageCache(cache_dir=tmp_path, enabled=True)
    cache.set("https://example.com/a", _page("https://example.com/a"))
    cache.set("https://example.com/b", _page("https://example.com/b"))

    assert cache.get("https://example.com/a").url == "https://example.com/a"
    assert cache.get("https://example.com/b").url == "https://example.com/b"


def test_sensitive_headers_are_stripped_before_writing_to_disk(tmp_path):
    cache = PageCache(cache_dir=tmp_path, enabled=True)
    page = _page("https://example.com/a")

    cache.set("https://example.com/a", page)

    written_files = list(tmp_path.glob("*.json"))
    assert len(written_files) == 1
    raw_contents = written_files[0].read_text(encoding="utf-8")
    assert "session=abc123" not in raw_contents
    assert "Set-Cookie" not in raw_contents
