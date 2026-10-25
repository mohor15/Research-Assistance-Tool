"""Recipient-address and link-safety validation shared by the UI, the
delivery service, and tests. Kept dependency-free (no Streamlit, no
pydantic) so it can be imported from any layer."""

from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import Optional

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(address: Optional[str]) -> bool:
    if not address:
        return False
    return bool(_EMAIL_PATTERN.match(address.strip()))


def is_safe_link(url: Optional[str]) -> bool:
    """True only for an absolute http(s) URL -- never `javascript:`, `data:`,
    a bare relative path, or anything else unsafe to render as a clickable
    `<a href>` in an HTML email."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)
