"""The user-editable default-recipient / auto-delivery-enabled preference
(Section 5 of the email-delivery feature) -- a single-row singleton table,
exactly like `scheduler_state`/`current_run_lock`. Absent any row yet,
`get()` returns `None` and callers fall back to
`config.settings.email.default_recipient`/`.enabled` from settings.yaml.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional

_SINGLETON_ID = "singleton"


@dataclass
class EmailSettingsRecord:
    default_recipient: str
    auto_delivery_enabled: bool


class EmailSettingsRepository:
    def get(self, conn: sqlite3.Connection) -> Optional[EmailSettingsRecord]:
        row = conn.execute(
            "SELECT default_recipient, auto_delivery_enabled FROM email_settings WHERE id = ?", (_SINGLETON_ID,)
        ).fetchone()
        if row is None:
            return None
        return EmailSettingsRecord(
            default_recipient=row["default_recipient"], auto_delivery_enabled=bool(row["auto_delivery_enabled"]),
        )

    def set_default_recipient(self, conn: sqlite3.Connection, recipient: str) -> None:
        conn.execute(
            """
            INSERT INTO email_settings (id, default_recipient, auto_delivery_enabled) VALUES (?, ?, 1)
            ON CONFLICT(id) DO UPDATE SET default_recipient = excluded.default_recipient
            """,
            (_SINGLETON_ID, recipient),
        )
        conn.commit()

    def set_auto_delivery_enabled(self, conn: sqlite3.Connection, enabled: bool, *, default_recipient: str) -> None:
        """`default_recipient` is required so a first-ever call (no row
        yet) never inserts an empty recipient -- the caller always has one
        on hand (either the previously-read record's, or config's)."""
        conn.execute(
            """
            INSERT INTO email_settings (id, default_recipient, auto_delivery_enabled) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET auto_delivery_enabled = excluded.auto_delivery_enabled
            """,
            (_SINGLETON_ID, default_recipient, 1 if enabled else 0),
        )
        conn.commit()
