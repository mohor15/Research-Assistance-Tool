"""URL canonicalization: fragments, trailing slash, tracking params, canonical metadata, dedup keys."""

from __future__ import annotations

from research_assistant.evidence.urls import canonicalize_url


def test_fragment_is_removed():
    assert canonicalize_url("https://example.com/a#section-2") == "https://example.com/a"


def test_trailing_slash_is_removed_except_root():
    assert canonicalize_url("https://example.com/a/") == "https://example.com/a"
    assert canonicalize_url("https://example.com/") == "https://example.com/"


def test_scheme_and_host_are_lowercased_but_path_is_preserved():
    assert canonicalize_url("HTTPS://Example.COM/Path/Article") == "https://example.com/Path/Article"


def test_tracking_parameters_are_stripped():
    url = "https://example.com/a?utm_source=twitter&utm_medium=social&gclid=xyz&id=42"
    assert canonicalize_url(url) == "https://example.com/a?id=42"


def test_duplicate_query_parameters_are_deduplicated():
    url = "https://example.com/a?id=42&id=42&ref=abc"
    # ref/id are safely deduplicated when literally repeated; ref is also a tracking param
    assert canonicalize_url(url) == "https://example.com/a?id=42"


def test_non_tracking_query_parameters_are_preserved_and_sorted():
    url = "https://example.com/a?z=1&a=2"
    assert canonicalize_url(url) == "https://example.com/a?a=2&z=1"


def test_page_supplied_canonical_url_takes_precedence():
    result = canonicalize_url(
        "https://example.com/a?utm_source=twitter",
        page_canonical_url="https://example.com/a-real-canonical",
    )
    assert result == "https://example.com/a-real-canonical"


def test_default_port_is_stripped():
    assert canonicalize_url("https://example.com:443/a") == "https://example.com/a"
    assert canonicalize_url("http://example.com:80/a") == "http://example.com/a"


def test_non_default_port_is_preserved():
    assert canonicalize_url("https://example.com:8443/a") == "https://example.com:8443/a"


def test_two_different_urls_pointing_at_the_same_resource_canonicalize_identically():
    a = "https://example.com/news/story/?utm_campaign=newsletter#top"
    b = "HTTPS://EXAMPLE.COM/news/story"
    assert canonicalize_url(a) == canonicalize_url(b)


def test_query_only_differences_that_are_not_tracking_params_remain_distinct():
    a = canonicalize_url("https://example.com/a?page=1")
    b = canonicalize_url("https://example.com/a?page=2")
    assert a != b
