"""Recipient-address and link-safety validation."""

from __future__ import annotations

from research_assistant.delivery.validation import is_safe_link, is_valid_email


def test_valid_emails_accepted():
    assert is_valid_email("ghosalmohor@gmail.com")
    assert is_valid_email("a.b+tag@example.co.uk")


def test_invalid_emails_rejected():
    assert not is_valid_email("")
    assert not is_valid_email(None)
    assert not is_valid_email("not-an-email")
    assert not is_valid_email("missing@domain")
    assert not is_valid_email("@missing-local.com")


def test_safe_links_require_http_or_https_scheme():
    assert is_safe_link("https://example.com/a")
    assert is_safe_link("http://example.com/a")


def test_unsafe_links_are_rejected():
    assert not is_safe_link("javascript:alert(1)")
    assert not is_safe_link("data:text/html,<script>alert(1)</script>")
    assert not is_safe_link("ftp://example.com")
    assert not is_safe_link("")
    assert not is_safe_link(None)
