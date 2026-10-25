"""Deterministic URL canonicalization for duplicate detection.

Conservative by design: only transformations that cannot change which
resource the URL points to. Scheme/host are lowercased (safe — DNS names
and schemes are case-insensitive), but path/query values are left as-is
since many servers treat paths case-sensitively. Fragments are dropped
(they identify a spot within the same resource, not a different one).
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Common analytics/referrer parameters that do not affect which resource is
# served. Removing them lets two links to the same article (one shared
# organically, one from a newsletter) canonicalize to the same URL.
_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAM_NAMES = {
    "gclid", "fbclid", "msclkid", "mc_cid", "mc_eid", "igshid",
    "ref", "ref_src", "refsrc", "spm", "yclid", "mkt_tok", "cmpid",
}


def canonicalize_url(url: str, *, page_canonical_url: Optional[str] = None) -> str:
    """Normalize `url` (preferring `page_canonical_url` when the page supplied one)
    into a stable form suitable as a duplicate-detection key."""
    target = page_canonical_url or url
    parsed = urlsplit(target)

    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[: -len(":80")]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[: -len(":443")]

    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    query = _canonical_query(parsed.query)

    return urlunsplit((scheme, netloc, path, query, ""))  # fragment always dropped


def _canonical_query(raw_query: str) -> str:
    pairs = parse_qsl(raw_query, keep_blank_values=True)
    seen = set()
    kept = []
    for key, value in pairs:
        if key.lower().startswith(_TRACKING_PARAM_PREFIXES) or key.lower() in _TRACKING_PARAM_NAMES:
            continue
        pair = (key, value)
        if pair in seen:
            continue
        seen.add(pair)
        kept.append(pair)
    kept.sort(key=lambda kv: (kv[0], kv[1]))
    return urlencode(kept)
