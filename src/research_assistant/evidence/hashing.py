"""Deterministic content hashing for provenance and duplicate detection.

This is an identity check, not a similarity score: two articles with the
same hash have byte-identical *normalized* text; anything less exact
(paraphrases, different articles about the same event) is Phase 5's job.
"""

from __future__ import annotations

import hashlib
from typing import Optional


def hash_content(text: Optional[str]) -> Optional[str]:
    """SHA-256 hex digest of `text`, or None if there's nothing meaningful to hash.

    Returning None (rather than hashing an empty string) matters for dedup:
    two articles that both failed extraction must never be treated as
    "the same content" just because they're both empty.
    """
    if not text or not text.strip():
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
