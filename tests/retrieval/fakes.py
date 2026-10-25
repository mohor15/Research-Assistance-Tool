"""Network-free fakes shared across retrieval tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FakeResponse:
    """Duck-types the bits of `requests.Response` the retrieval client uses."""

    status_code: int = 200
    url: str = "http://example.com/"
    text: str = ""
    headers: Dict[str, str] = field(default_factory=lambda: {"Content-Type": "text/html"})

    @property
    def ok(self) -> bool:
        return self.status_code < 400


class FakeSession:
    """A scripted stand-in for `requests.Session`.

    `responses` is a list of either `FakeResponse` instances or exceptions;
    each call to `.get()` pops the next one (raising it if it's an exception).
    """

    def __init__(self, responses: List):
        self._responses = list(responses)
        self.calls: List[dict] = []

    def get(self, url, *, timeout=None, headers=None, allow_redirects=True):
        self.calls.append({"url": url, "timeout": timeout, "headers": headers})
        if not self._responses:
            # Phase 13F: an unscripted call (e.g. the sitemap-scavenging
            # probe `discovery.py` now issues additively for ANALYST-
            # category sources) is answered with a plain 404 rather than
            # raising -- a real server genuinely returns 404 for a path a
            # test fixture never anticipated, and the production code under
            # test already handles that gracefully (see `sitemap.py`'s
            # "never raises on a missing/malformed sitemap"). Tests that
            # specifically want to assert "no further calls happened" still
            # can, via `session.calls`.
            return FakeResponse(status_code=404, url=url, text="")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
