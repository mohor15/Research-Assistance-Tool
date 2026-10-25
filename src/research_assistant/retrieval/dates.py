"""Date normalization shared by feed parsing and article extraction.

Every retrieval-layer date field is normalized to a timezone-aware
`datetime` (assumed UTC when the source gave no timezone) so later phases
never have to guess what a naive datetime meant.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Optional

from dateutil import parser as dateutil_parser

# A bare numeric string (no separators dateutil would otherwise recognize)
# is sometimes a Unix-epoch-seconds timestamp in a non-standard, site-
# specific metadata field (e.g. a real, observed JPMorganChase page's
# `<meta name="alg-search-date" content="1790697480"/>` -- an Algolia-
# style search-indexing field, not a named date tag, yet it is this
# project's extraction's only real evidence of that page's true
# publication date). Narrowly guarded to a plausible calendar-year range so
# an unrelated small/large bare integer (a product SKU, a count, "42") is
# never misread as a date -- if it's outside this range, we deliberately
# return None rather than let dateutil guess at a bare number.
_EPOCH_SECONDS_MIN = 946684800  # 2000-01-01T00:00:00Z
_EPOCH_SECONDS_MAX = 4102444800  # 2100-01-01T00:00:00Z


def parse_date(value: Optional[str]) -> Optional[datetime]:
    """Parse a date string from RSS/Atom/HTML metadata into a UTC-aware datetime.

    Returns None (never raises) if `value` is empty or unparseable — a
    missing or malformed date must not abort retrieval.
    """
    if not value:
        return None
    stripped = value.strip()
    if stripped.isdigit():
        as_int = int(stripped)
        if _EPOCH_SECONDS_MIN <= as_int <= _EPOCH_SECONDS_MAX:
            return datetime.fromtimestamp(as_int, tz=timezone.utc)
        return None  # a bare number outside any plausible epoch range is not a date
    try:
        parsed = dateutil_parser.parse(value)
    except (ValueError, OverflowError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


_TIME_COMPONENT_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")


def date_string_has_time_component(value: Optional[str]) -> bool:
    """True if the RAW date string itself carries an explicit time-of-day
    (any `HH:MM` or `HH:MM:SS` shape, as found in ISO8601
    ("2026-10-09T13:20:00-04:00"), RFC822 ("Thu, 09 Oct 2026 13:20:00
    -0400"), or human-readable ("October 9, 2026 1:20 PM") date strings) --
    never derived from the PARSED result (a parsed value landing exactly at
    midnight UTC is indistinguishable from a bare date after parsing, which
    is exactly the ambiguity this function exists to resolve by looking at
    the source string instead). Generic and format-agnostic: matched purely
    by the presence of a colon-separated time shape, never a specific
    source/publisher. A bare date string ("2026-10-09", "October 9, 2026")
    has no such shape and returns False."""
    if not value:
        return False
    return bool(_TIME_COMPONENT_RE.search(value))


def struct_time_to_datetime(struct: Optional[time.struct_time]) -> Optional[datetime]:
    """Convert feedparser's `*_parsed` struct_time (already UTC) to a datetime."""
    if not struct:
        return None
    try:
        return datetime(*struct[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
