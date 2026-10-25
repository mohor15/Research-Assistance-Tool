"""Honest failure classification (Part A1): a 401/403 actually received from
the server must never be reported as a generic "connection_error", and a
local TLS/cert problem must be distinguishable from the remote site being
genuinely unreachable."""

from __future__ import annotations

from research_assistant.retrieval.classification import classify, classify_many
from research_assistant.retrieval.models import RetrievalFailure


def test_tls_certificate_error_is_classified_distinctly_from_connection_error():
    assert classify(RetrievalFailure(reason="tls_certificate_error", detail="x")) == "TLS_CERTIFICATE_ERROR"
    assert classify(RetrievalFailure(reason="connection_error", detail="x")) == "CONNECTION_ERROR"


def test_timeout_is_classified_as_connection_timeout():
    assert classify(RetrievalFailure(reason="timeout")) == "CONNECTION_TIMEOUT"


def test_http_401_and_403_are_classified_distinctly_not_as_connection_error():
    assert classify(RetrievalFailure(reason="http_error", detail="401")) == "HTTP_401"
    assert classify(RetrievalFailure(reason="http_error", detail="403")) == "HTTP_403"


def test_other_4xx_and_5xx_are_bucketed_separately():
    assert classify(RetrievalFailure(reason="http_error", detail="400")) == "HTTP_4XX"
    assert classify(RetrievalFailure(reason="http_error", detail="503")) == "HTTP_5XX"


def test_robots_disallowed_is_classified_as_robots_blocked():
    assert classify(RetrievalFailure(reason="robots_disallowed")) == "ROBOTS_BLOCKED"


def test_browser_http2_and_generic_browser_failures_are_distinct():
    assert classify(RetrievalFailure(reason="browser_http2_error", detail="x")) == "BROWSER_HTTP2_ERROR"
    assert classify(RetrievalFailure(reason="browser_retrieval_failed", detail="x")) == "BROWSER_RETRIEVAL_FAILED"


def test_a_401_confirmed_by_a_matching_browser_failure_is_anti_bot_access_blocked():
    http_failure = RetrievalFailure(reason="http_error", detail="401")
    browser_failure = RetrievalFailure(reason="http_error", detail="401")
    assert classify(http_failure, browser_failure=browser_failure) == "ANTI_BOT_ACCESS_BLOCKED"


def test_a_401_without_browser_corroboration_is_not_yet_anti_bot():
    http_failure = RetrievalFailure(reason="http_error", detail="401")
    assert classify(http_failure) == "HTTP_401"


def test_classify_many_upgrades_a_repeated_matching_401_to_anti_bot_access_blocked():
    """Generalizes the single-URL corroboration rule: when the HTTP path AND
    a browser-rendering retry (see RetrievalService.discover_candidates'
    Part A2 fallback) both independently landed on the same 401 for a
    source, that's a confirmed access restriction, not two unrelated bare
    HTTP_401s."""
    failures = [
        RetrievalFailure(reason="http_error", detail="401"),
        RetrievalFailure(reason="http_error", detail="401"),
    ]
    assert classify_many(failures) == ["ANTI_BOT_ACCESS_BLOCKED"]


def test_classify_many_dedupes_and_sorts():
    failures = [
        RetrievalFailure(reason="timeout"),
        RetrievalFailure(reason="timeout"),
        RetrievalFailure(reason="tls_certificate_error"),
    ]
    assert classify_many(failures) == ["CONNECTION_TIMEOUT", "TLS_CERTIFICATE_ERROR"]
