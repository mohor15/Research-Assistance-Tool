"""Newsletter selection persistence (Phase 16 — user-facing CURRENT
dashboard). A bare `event_id -> selected_at` pointer table: it never stores
a copy of the Event, so a user's selection can never mutate or duplicate
the underlying intelligence. `PRIMARY KEY (event_id)` makes a duplicate
selection structurally impossible."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import List

from research_assistant.storage.serde import dt_to_str


class NewsletterSelectionRepository:
    def add(self, conn: sqlite3.Connection, event_id: str, *, selected_at: datetime) -> None:
        conn.execute(
            """
            INSERT INTO newsletter_selections (event_id, selected_at) VALUES (?, ?)
            ON CONFLICT(event_id) DO NOTHING
            """,
            (event_id, dt_to_str(selected_at)),
        )
        conn.commit()

    def remove(self, conn: sqlite3.Connection, event_id: str) -> None:
        conn.execute("DELETE FROM newsletter_selections WHERE event_id = ?", (event_id,))
        conn.commit()

    def is_selected(self, conn: sqlite3.Connection, event_id: str) -> bool:
        row = conn.execute("SELECT 1 FROM newsletter_selections WHERE event_id = ?", (event_id,)).fetchone()
        return row is not None

    def list_ids(self, conn: sqlite3.Connection) -> List[str]:
        rows = conn.execute("SELECT event_id FROM newsletter_selections ORDER BY selected_at ASC").fetchall()
        return [row["event_id"] for row in rows]
