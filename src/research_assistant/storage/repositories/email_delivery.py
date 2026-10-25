"""Email delivery tracking (automatic CURRENT/Morning/Evening digest
delivery + manual newsletter sends). One row per (digest, recipient, kind)
delivery attempt-group. `kind='automatic'` rows are deduplicated by the
database's own partial unique index (`idx_email_deliveries_auto_dedup`) --
the idempotency key is `(digest_id, recipient)` for automatic delivery --
so `create_automatic` below relies on `INSERT ... ON CONFLICT DO NOTHING`
to make "create if absent, otherwise leave the existing row untouched"
atomic rather than a separate check-then-insert that could race. Manual
sends (`kind='manual'`) intentionally get a fresh row per `create_manual`
call -- a user may deliberately resend the same newsletter.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from research_assistant.storage.serde import dt_from_str, dt_to_str

STATUS_PENDING = "pending"
STATUS_SENDING = "sending"
STATUS_SENT = "sent"
STATUS_FAILED = "failed"

KIND_AUTOMATIC = "automatic"
KIND_MANUAL = "manual"


@dataclass
class EmailDeliveryRecord:
    id: str
    digest_id: str
    digest_type: str
    recipient: str
    kind: str
    status: str
    attempt_count: int
    last_attempt_at: Optional[datetime]
    created_at: datetime
    sent_at: Optional[datetime]
    failure_reason: Optional[str]


def _hydrate(row: sqlite3.Row) -> EmailDeliveryRecord:
    return EmailDeliveryRecord(
        id=row["id"], digest_id=row["digest_id"], digest_type=row["digest_type"], recipient=row["recipient"],
        kind=row["kind"], status=row["status"], attempt_count=row["attempt_count"],
        last_attempt_at=dt_from_str(row["last_attempt_at"]), created_at=dt_from_str(row["created_at"]),
        sent_at=dt_from_str(row["sent_at"]), failure_reason=row["failure_reason"],
    )


class EmailDeliveryRepository:
    def create_automatic(
        self, conn: sqlite3.Connection, *, digest_id: str, digest_type: str, recipient: str, now: datetime,
    ) -> EmailDeliveryRecord:
        """Idempotent: if an automatic row for this (digest_id, recipient)
        already exists, returns the EXISTING row unchanged -- never a
        second row, never silently overwriting a prior attempt's history."""
        conn.execute(
            """
            INSERT INTO email_deliveries (
                id, digest_id, digest_type, recipient, kind, status, attempt_count, created_at
            ) VALUES (?, ?, ?, ?, 'automatic', 'pending', 0, ?)
            ON CONFLICT(digest_id, recipient) WHERE kind = 'automatic' DO NOTHING
            """,
            (str(uuid.uuid4()), digest_id, digest_type, recipient, dt_to_str(now)),
        )
        conn.commit()
        record = self.get_automatic(conn, digest_id=digest_id, recipient=recipient)
        assert record is not None  # the INSERT above (or a prior call) guarantees a row exists
        return record

    def create_manual(
        self, conn: sqlite3.Connection, *, recipient: str, now: datetime,
        digest_id: Optional[str] = None, digest_type: str = "newsletter",
    ) -> EmailDeliveryRecord:
        """Always inserts a fresh row -- a deliberate manual resend to the
        same recipient is allowed. Preventing an ACCIDENTAL duplicate (a
        double-click / Streamlit rerun) is the UI layer's responsibility
        (see `ui/app.py`), not this repository's. `digest_id` defaults to
        `None`/`digest_type="newsletter"` since a manual send is built from
        an arbitrary user-selected set of events, not necessarily one
        single persisted Digest row."""
        delivery_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO email_deliveries (
                id, digest_id, digest_type, recipient, kind, status, attempt_count, created_at
            ) VALUES (?, ?, ?, ?, 'manual', 'pending', 0, ?)
            """,
            (delivery_id, digest_id, digest_type, recipient, dt_to_str(now)),
        )
        conn.commit()
        record = self.get(conn, delivery_id)
        assert record is not None
        return record

    def mark_sent(self, conn: sqlite3.Connection, *, delivery_id: str, now: datetime) -> None:
        conn.execute(
            "UPDATE email_deliveries SET status = ?, attempt_count = attempt_count + 1, "
            "last_attempt_at = ?, sent_at = ?, failure_reason = NULL WHERE id = ?",
            (STATUS_SENT, dt_to_str(now), dt_to_str(now), delivery_id),
        )
        conn.commit()

    def mark_failed(self, conn: sqlite3.Connection, *, delivery_id: str, now: datetime, reason: str) -> None:
        """`reason` must already be sanitized by the caller (the mailer
        layer) -- never a raw SMTP exception that could embed a password
        or full server response; see `delivery.mailer`."""
        conn.execute(
            "UPDATE email_deliveries SET status = ?, attempt_count = attempt_count + 1, "
            "last_attempt_at = ?, failure_reason = ? WHERE id = ?",
            (STATUS_FAILED, dt_to_str(now), reason, delivery_id),
        )
        conn.commit()

    def get(self, conn: sqlite3.Connection, delivery_id: str) -> Optional[EmailDeliveryRecord]:
        row = conn.execute("SELECT * FROM email_deliveries WHERE id = ?", (delivery_id,)).fetchone()
        return _hydrate(row) if row else None

    def get_automatic(self, conn: sqlite3.Connection, *, digest_id: str, recipient: str) -> Optional[EmailDeliveryRecord]:
        row = conn.execute(
            "SELECT * FROM email_deliveries WHERE digest_id = ? AND recipient = ? AND kind = 'automatic'",
            (digest_id, recipient),
        ).fetchone()
        return _hydrate(row) if row else None

    def list_for_digest(self, conn: sqlite3.Connection, digest_id: str) -> List[EmailDeliveryRecord]:
        rows = conn.execute(
            "SELECT * FROM email_deliveries WHERE digest_id = ? ORDER BY created_at ASC", (digest_id,)
        ).fetchall()
        return [_hydrate(row) for row in rows]
