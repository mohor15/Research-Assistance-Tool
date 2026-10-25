"""Digest persistence, including the digest<->event junction table (which
preserves ranked order via `position`). Themes are a genuinely nested
sub-structure (label + description + event_ids each) and are stored as
JSON on the digest row itself rather than further normalized."""

from __future__ import annotations

import sqlite3
from typing import List, Optional

from research_assistant.models.digest import Digest, DigestTheme, DigestType
from research_assistant.storage.serde import dt_from_str, dt_to_str, json_dump, json_load


class DigestRepository:
    def upsert(self, conn: sqlite3.Connection, digest: Digest) -> None:
        conn.execute(
            """
            INSERT INTO digests (id, digest_type, generated_at, period_start, period_end, themes_json)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                digest_type=excluded.digest_type, generated_at=excluded.generated_at,
                period_start=excluded.period_start, period_end=excluded.period_end,
                themes_json=excluded.themes_json
            """,
            (
                digest.id, digest.digest_type.value, dt_to_str(digest.generated_at),
                dt_to_str(digest.period_start), dt_to_str(digest.period_end),
                json_dump([theme.model_dump(mode="json") for theme in digest.themes]),
            ),
        )
        conn.execute("DELETE FROM digest_events WHERE digest_id = ?", (digest.id,))
        conn.executemany(
            "INSERT INTO digest_events (digest_id, event_id, position) VALUES (?,?,?)",
            [(digest.id, event_id, position) for position, event_id in enumerate(digest.events)],
        )

    def get(self, conn: sqlite3.Connection, digest_id: str) -> Optional[Digest]:
        row = conn.execute("SELECT * FROM digests WHERE id = ?", (digest_id,)).fetchone()
        return self._hydrate(conn, row) if row else None

    def get_recent(self, conn: sqlite3.Connection, *, digest_type: Optional[DigestType] = None, limit: int = 20) -> List[Digest]:
        if digest_type is not None:
            rows = conn.execute(
                "SELECT * FROM digests WHERE digest_type = ? ORDER BY generated_at DESC LIMIT ?",
                (digest_type.value, limit),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM digests ORDER BY generated_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._hydrate(conn, row) for row in rows]

    def get_event_ids(self, conn: sqlite3.Connection, digest_id: str) -> List[str]:
        rows = conn.execute(
            "SELECT event_id FROM digest_events WHERE digest_id = ? ORDER BY position", (digest_id,)
        ).fetchall()
        return [row["event_id"] for row in rows]

    def _hydrate(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Digest:
        return Digest(
            id=row["id"], digest_type=DigestType(row["digest_type"]), generated_at=dt_from_str(row["generated_at"]),
            period_start=dt_from_str(row["period_start"]), period_end=dt_from_str(row["period_end"]),
            events=self.get_event_ids(conn, row["id"]),
            themes=[DigestTheme.model_validate(theme) for theme in json_load(row["themes_json"], [])],
        )
