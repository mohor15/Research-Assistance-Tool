"""Whitespace/content cleanup shared by article normalization.

Purely mechanical (regex-based) cleanup — no semantic rewriting, no
truncation, no summarization.
"""

from __future__ import annotations

import re
from typing import Optional

_TRAILING_INLINE_WHITESPACE = re.compile(r"[ \t]*\n[ \t]*")
_REPEATED_SPACES = re.compile(r"[ \t]{2,}")
_REPEATED_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_whitespace(value: Optional[str]) -> Optional[str]:
    """Collapse repeated spaces/blank lines and strip. Returns None for empty input."""
    if not value:
        return None
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_INLINE_WHITESPACE.sub("\n", text)
    text = _REPEATED_SPACES.sub(" ", text)
    text = _REPEATED_BLANK_LINES.sub("\n\n", text)
    text = text.strip()
    return text or None
