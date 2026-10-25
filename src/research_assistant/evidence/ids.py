"""Deterministic identifier generation.

IDs are derived from content, not assigned sequentially — normalizing the
same URL twice (e.g. on a re-crawl) always yields the same `Article.id`,
which is what makes URL-based duplicate detection meaningful across runs.
"""

from __future__ import annotations

import hashlib
from typing import Optional

_ID_LENGTH = 16  # 64 bits of a sha256 digest — plenty for local dedup, still short


def generate_article_id(canonical_url: str) -> str:
    return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:_ID_LENGTH]


def generate_evidence_id(article_id: str, content_hash: Optional[str]) -> str:
    basis = f"{article_id}:{content_hash or ''}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:_ID_LENGTH]
