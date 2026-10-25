"""Human-readable failure classification for diagnostics/reporting only.

Nothing in the retrieval engine's control flow (retry decisions, browser
fallback eligibility, evidence gating) depends on these labels — they exist
purely so logs/diagnostics/pipeline reports say e.g. "TLS_CERTIFICATE_ERROR"
or "ANTI_BOT_ACCESS_BLOCKED" instead of a generic "connection_error" that
conflates unrelated causes. The underlying `RetrievalFailure.reason` strings
(checked by existing tests and control-flow code) are unchanged by this
module.
"""

from __future__ import annotations

from typing import Optional

from research_assistant.retrieval.models import RetrievalFailure

_ANTI_BOT_STATUSES = {"401", "403"}


def classify(
    failure: Optional[RetrievalFailure], *, browser_failure: Optional[RetrievalFailure] = None
) -> Optional[str]:
    """`failure` is the HTTP-path failure; `browser_failure` (if provided) is
    what the browser fallback independently produced for the *same*
    resource. Only when both agree on a 401/403 do we call it a confirmed
    `ANTI_BOT_ACCESS_BLOCKED` rather than a bare `HTTP_401`/`HTTP_403` --
    otherwise a single 401 might just be a transient/misconfigured response
    that a browser retry could still recover from.
    """
    if failure is None:
        return None

    reason = failure.reason
    detail = failure.detail

    if reason == "robots_disallowed":
        return "ROBOTS_BLOCKED"
    if reason == "tls_certificate_error":
        return "TLS_CERTIFICATE_ERROR"
    if reason == "timeout":
        return "CONNECTION_TIMEOUT"
    if reason == "anti_bot_access_blocked":
        return "ANTI_BOT_ACCESS_BLOCKED"
    if reason == "browser_http2_error":
        return "BROWSER_HTTP2_ERROR"
    if reason == "browser_retrieval_failed":
        return "BROWSER_RETRIEVAL_FAILED"
    if reason == "connection_error":
        return "CONNECTION_ERROR"
    if reason == "empty_response":
        return "EXTRACTION_FAILED"
    if reason == "http_error" and detail:
        if detail in _ANTI_BOT_STATUSES:
            if (
                browser_failure is not None
                and browser_failure.reason == "http_error"
                and browser_failure.detail == detail
            ):
                return "ANTI_BOT_ACCESS_BLOCKED"
            return f"HTTP_{detail}"
        try:
            code = int(detail)
        except ValueError:
            return "HTTP_4XX"
        return "HTTP_4XX" if code < 500 else "HTTP_5XX"

    return reason.upper()


def classify_many(failures: list) -> list:
    """Dedup'd, sorted classification labels for a list of `RetrievalFailure`
    (e.g. `SourceUnreachableError.failures`, which -- after the Part A2
    discovery fallback -- may hold BOTH the original HTTP-path failures and
    whatever a browser-rendering retry independently produced for the same
    source). If two or more of them independently landed on the same
    401/403, that is upgraded to a confirmed `ANTI_BOT_ACCESS_BLOCKED`
    rather than reported as a bare `HTTP_401`/`HTTP_403` -- the same
    corroboration rule `classify(failure, browser_failure=...)` applies for
    a single URL, generalized to "this access-blocked status was seen more
    than once via independent retrieval paths"."""
    non_null = [f for f in failures if f is not None]
    labels = {classify(f) for f in non_null}
    labels.discard(None)

    anti_bot_statuses = [
        f.detail for f in non_null if f.reason == "http_error" and f.detail in _ANTI_BOT_STATUSES
    ]
    for detail, count in {d: anti_bot_statuses.count(d) for d in set(anti_bot_statuses)}.items():
        if count >= 2:
            labels.discard(f"HTTP_{detail}")
            labels.add("ANTI_BOT_ACCESS_BLOCKED")

    return sorted(labels)
