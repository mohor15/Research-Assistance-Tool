"""Content hashing: identity for same/whitespace-normalized content, difference for different content."""

from __future__ import annotations

from research_assistant.evidence.hashing import hash_content
from research_assistant.evidence.text import normalize_whitespace


def test_same_content_produces_same_hash():
    text = "Identical article body."
    assert hash_content(text) == hash_content(text)


def test_different_content_produces_different_hash():
    assert hash_content("Article A") != hash_content("Article B")


def test_none_and_empty_text_hash_to_none():
    assert hash_content(None) is None
    assert hash_content("") is None
    assert hash_content("   \n\t  ") is None


def test_whitespace_only_differences_hash_the_same_after_normalization():
    raw_a = "Some   article    text.\n\n\n\nMore text."
    raw_b = "Some article text.\n\nMore text."
    assert hash_content(normalize_whitespace(raw_a)) == hash_content(normalize_whitespace(raw_b))


def test_hash_is_a_valid_sha256_hex_digest():
    digest = hash_content("some content")
    assert digest is not None
    assert len(digest) == 64
    int(digest, 16)  # raises if not valid hex
