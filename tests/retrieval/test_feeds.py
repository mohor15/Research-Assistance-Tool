"""RSS/Atom parsing: valid feeds, malformed feeds, missing fields.

`feedparser` is optional; when it's not installed these exercise the stdlib
fallback parser directly (which is the path this sandbox always runs
without network access). The `feedparser`-specific tests auto-skip when it
isn't installed and start running the moment it's `pip install`ed.
"""

from __future__ import annotations

import pytest

from research_assistant.retrieval.feeds import is_feedparser_available, looks_like_feed, parse_feed
from research_assistant.retrieval.models import DiscoveryMethod

VALID_RSS = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Example Newsroom</title>
    <item>
      <title>Bank launches new AI copilot</title>
      <link>https://example.com/news/ai-copilot</link>
      <pubDate>Mon, 01 Sep 2025 10:00:00 GMT</pubDate>
      <author>press@example.com</author>
      <description>A short summary of the launch.</description>
    </item>
    <item>
      <title>Untitled item missing some fields</title>
      <link>https://example.com/news/second-item</link>
    </item>
  </channel>
</rss>
"""

VALID_ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Example AI Blog</title>
  <entry>
    <title>New model release</title>
    <link href="https://example.com/blog/new-model" rel="alternate"/>
    <updated>2025-09-01T10:00:00Z</updated>
    <author><name>Research Team</name></author>
    <summary>Summary of the new model.</summary>
  </entry>
</feed>
"""

MALFORMED_XML = "<rss version=\"2.0\"><channel><item><title>Broken"

RSS_MISSING_LINK = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <item>
      <title>No link here</title>
    </item>
  </channel>
</rss>
"""


def test_looks_like_feed_detects_rss_and_atom():
    assert looks_like_feed(VALID_RSS, "application/rss+xml") is True
    assert looks_like_feed(VALID_ATOM, None) is True
    assert looks_like_feed("<html><body>hi</body></html>", "text/html") is False


def test_valid_rss_parses_into_candidate_urls():
    entries = parse_feed(VALID_RSS, source_name="Example Bank", domain="example.com")

    assert len(entries) == 2
    first = entries[0]
    assert first.url == "https://example.com/news/ai-copilot"
    assert first.title == "Bank launches new AI copilot"
    assert first.discovery_method == DiscoveryMethod.RSS
    assert first.published_at is not None
    assert first.published_at.year == 2025
    assert first.feed_summary == "A short summary of the launch."


def test_rss_item_missing_optional_fields_still_parses():
    entries = parse_feed(VALID_RSS, source_name="Example Bank", domain="example.com")
    second = entries[1]
    assert second.url == "https://example.com/news/second-item"
    assert second.published_at is None
    assert second.feed_summary is None


def test_valid_atom_parses_into_candidate_urls():
    entries = parse_feed(VALID_ATOM, source_name="Example AI Co", domain="example.com")

    assert len(entries) == 1
    entry = entries[0]
    assert entry.url == "https://example.com/blog/new-model"
    assert entry.discovery_method == DiscoveryMethod.ATOM
    assert entry.feed_author == "Research Team"
    assert entry.published_at is not None


def test_malformed_feed_returns_empty_list_without_raising():
    entries = parse_feed(MALFORMED_XML, source_name="Broken Source", domain="example.com")
    assert entries == []


def test_feed_item_missing_link_is_skipped_not_crashed():
    entries = parse_feed(RSS_MISSING_LINK, source_name="Example Bank", domain="example.com")
    assert entries == []


def test_empty_content_returns_empty_list():
    assert parse_feed("", source_name="x", domain="example.com") == []
    assert parse_feed("   ", source_name="x", domain="example.com") == []


@pytest.mark.skipif(not is_feedparser_available(), reason="feedparser not installed in this environment")
def test_feedparser_backend_produces_equivalent_results():
    entries = parse_feed(VALID_RSS, source_name="Example Bank", domain="example.com")
    assert len(entries) == 2
    assert entries[0].url == "https://example.com/news/ai-copilot"
