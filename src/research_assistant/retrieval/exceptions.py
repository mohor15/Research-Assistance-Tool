"""Retrieval-layer exceptions.

Most failures are represented as data (`RetrievedPage.failure`,
`ExtractedArticle.status`) rather than exceptions, so a bad source never
crashes a retrieval run. These exceptions are for the few cases where a
caller needs to distinguish "this fallback mechanism itself isn't usable"
from an ordinary fetch failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from research_assistant.retrieval.models import RetrievalFailure


class RetrievalError(Exception):
    """Base class for retrieval-layer errors."""


class BrowserUnavailableError(RetrievalError):
    """Playwright is not installed, or its browser binaries are not installed."""


class SourceUnreachableError(RetrievalError):
    """Every one of a source's configured URLs failed to fetch at all (SSL,
    connection, timeout, robots.txt, ...). Raised by
    `discovery.discover_candidates()` so this is categorically distinguishable
    from "fetched fine, but found zero candidate links" — the pipeline's
    existing per-source exception handling (`pipeline.nodes.retrieve`)
    turns this into a visible `stage="retrieve"` diagnostic, rather than the
    source's failure being silently indistinguishable from a genuine
    zero-content outcome.

    `failures` carries the structured per-URL `RetrievalFailure` objects
    behind the joined message (reason + HTTP detail, e.g. a 401 vs a TLS
    error), so callers that need to classify *why* a source was unreachable
    (see `retrieval.classification.classify`) don't have to parse the
    message string. Optional/defaults to `None` purely so existing callers
    that only ever constructed this from a plain string keep working
    unchanged."""

    def __init__(self, message: str, *, failures: Optional[List["RetrievalFailure"]] = None) -> None:
        super().__init__(message)
        self.failures: List["RetrievalFailure"] = failures or []
