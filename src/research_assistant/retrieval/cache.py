"""Local on-disk page cache, keyed by a hash of the URL.

Deliberately simple for Phase 1: presence in the cache means "reuse it", no
TTL/invalidation logic. Cache files live under `data/cache/`, which is
gitignored, and headers are sanitized before being written so nothing
sensitive ends up on disk.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional

from research_assistant.retrieval.models import RetrievedPage

logger = logging.getLogger(__name__)

_SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie"}


def _sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _SENSITIVE_HEADERS}


class PageCache:
    def __init__(self, *, cache_dir: Path, enabled: bool) -> None:
        self._dir = cache_dir
        self._enabled = enabled
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, url: str) -> Path:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self._dir / f"{key}.json"

    def get(self, url: str) -> Optional[RetrievedPage]:
        if not self._enabled:
            return None
        path = self._path(url)
        if not path.exists():
            return None
        try:
            page = RetrievedPage.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Cache read failed for %s: %s", url, exc)
            return None
        return page.model_copy(update={"from_cache": True})

    def set(self, url: str, page: RetrievedPage) -> None:
        if not self._enabled:
            return
        sanitized = page.model_copy(update={"headers": _sanitize_headers(page.headers)})
        try:
            self._path(url).write_text(sanitized.model_dump_json(), encoding="utf-8")
        except OSError as exc:
            logger.warning("Cache write failed for %s: %s", url, exc)
