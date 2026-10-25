"""Article-level deduplication: same canonical URL, cross-URL same content hash, distinct articles preserved.

Explicitly verifies that two DIFFERENT articles about the same real-world
event (different URLs, different content) are never merged here — that
remains Phase 5's job.
"""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.evidence.dedup import deduplicate_articles
from research_assistant.models.article import Article
from research_assistant.models.enums import SourcePriority, SourceType


def _article(**overrides) -> Article:
    kwargs = dict(
        id=overrides.pop("id", "id-" + overrides.get("url", "x")),
        url="https://example.com/a",
        canonical_url="https://example.com/a",
        domain="example.com",
        source_name="Example Bank",
        source_type=SourceType.PRIMARY,
        source_priority=SourcePriority.HIGH,
        retrieved_at=datetime(2025, 9, 1, tzinfo=timezone.utc),
        content_hash=None,
        clean_text=None,
        title=None,
        author=None,
        publication_date=None,
    )
    kwargs.update(overrides)
    return Article(**kwargs)


def test_same_canonical_url_is_deduplicated_to_one_article():
    a = _article(id="a1", url="https://example.com/a?utm_source=x", canonical_url="https://example.com/a")
    b = _article(id="a2", url="https://example.com/a/", canonical_url="https://example.com/a")

    result = deduplicate_articles([a, b])

    assert len(result) == 1


def test_different_urls_with_same_content_hash_are_deduplicated():
    a = _article(id="a1", url="https://mirror1.example.com/a", canonical_url="https://mirror1.example.com/a", content_hash="hash1")
    b = _article(id="a2", url="https://mirror2.example.com/a", canonical_url="https://mirror2.example.com/a", content_hash="hash1")

    result = deduplicate_articles([a, b])

    assert len(result) == 1


def test_different_urls_and_different_content_are_both_kept():
    a = _article(id="a1", url="https://example.com/a", canonical_url="https://example.com/a", content_hash="hash1")
    b = _article(id="a2", url="https://example.com/b", canonical_url="https://example.com/b", content_hash="hash2")

    result = deduplicate_articles([a, b])

    assert len(result) == 2
    assert {r.id for r in result} == {"a1", "a2"}


def test_distinct_articles_about_the_same_event_are_never_merged():
    """Reuters and Bloomberg covering the same announcement stay separate Articles."""
    reuters = _article(
        id="reuters", url="https://reuters.com/story-a", canonical_url="https://reuters.com/story-a",
        content_hash="reuters-hash", title="Bank announces AI partnership",
    )
    bloomberg = _article(
        id="bloomberg", url="https://bloomberg.com/story-a", canonical_url="https://bloomberg.com/story-a",
        content_hash="bloomberg-hash", title="Bank unveils new AI partnership",
    )

    result = deduplicate_articles([reuters, bloomberg])

    assert len(result) == 2


def test_more_complete_article_is_kept_among_duplicates():
    thin = _article(id="thin", url="https://example.com/a", canonical_url="https://example.com/a", content_hash="h1")
    complete = _article(
        id="complete", url="https://example.com/a?ref=y", canonical_url="https://example.com/a",
        content_hash="h1", title="Full Title", author="Jane Doe",
        publication_date=datetime(2025, 9, 1, tzinfo=timezone.utc), clean_text="A full body of text.",
    )

    result = deduplicate_articles([thin, complete])

    assert len(result) == 1
    assert result[0].id == "complete"


def test_empty_input_returns_empty_output():
    assert deduplicate_articles([]) == []


def test_single_article_passes_through_unchanged():
    a = _article(id="only", url="https://example.com/a", canonical_url="https://example.com/a")
    result = deduplicate_articles([a])
    assert result == [a]


def test_transitive_grouping_via_shared_hash_and_url():
    # a & b share canonical_url; b & c share content_hash -> all one group
    a = _article(id="a", url="https://example.com/a", canonical_url="https://example.com/a", content_hash="hash-b")
    b = _article(id="b", url="https://example.com/a/", canonical_url="https://example.com/a", content_hash="hash-b")
    c = _article(id="c", url="https://mirror.example.com/a", canonical_url="https://mirror.example.com/a", content_hash="hash-b")

    result = deduplicate_articles([a, b, c])

    assert len(result) == 1
