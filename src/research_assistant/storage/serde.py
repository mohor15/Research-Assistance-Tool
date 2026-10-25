"""Tiny shared serialization helpers used by every repository.

Kept in one place so datetime/JSON handling is consistent across all six
tables rather than reimplemented per repository.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional


def dt_to_str(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def dt_from_str(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None


def json_dump(value: Any) -> str:
    return json.dumps(value)


def json_load(value: Optional[str], default: Any) -> Any:
    return json.loads(value) if value else default
