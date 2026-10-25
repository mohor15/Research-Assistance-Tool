"""EmailDeliveryRepository: automatic-kind dedup via the partial unique
index (the idempotency key is (digest_id, recipient) for kind='automatic'),
manual-kind rows always fresh, and attempt/status tracking."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.storage.database import Database
from research_assistant.storage.repositories.email_delivery import (
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_SENT,
    EmailDeliveryRepository,
)

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def _insert_minimal_digest(database: Database, digest_id: str) -> None:
    """`email_deliveries.digest_id` is an FK into `digests` -- insert a
    minimal satisfying row directly (this file is testing the delivery
    repository, not digest persistence, which is covered elsewhere)."""
    conn = database.connect()
    conn.execute(
        "INSERT INTO digests (id, digest_type, generated_at, period_start, period_end) VALUES (?, 'morning', ?, ?, ?)",
        (digest_id, D.isoformat(), D.isoformat(), D.isoformat()),
    )
    conn.commit()
    conn.close()


def test_create_automatic_starts_pending_with_zero_attempts(database: Database):
    _insert_minimal_digest(database, "dg1")
    conn = database.connect()
    record = EmailDeliveryRepository().create_automatic(conn, digest_id="dg1", digest_type="morning", recipient="a@b.com", now=D)
    conn.close()

    assert record.status == STATUS_PENDING
    assert record.attempt_count == 0
    assert record.kind == "automatic"


def test_create_automatic_is_idempotent_for_the_same_digest_and_recipient(database: Database):
    _insert_minimal_digest(database, "dg1")
    conn = database.connect()
    repo = EmailDeliveryRepository()
    first = repo.create_automatic(conn, digest_id="dg1", digest_type="morning", recipient="a@b.com", now=D)
    second = repo.create_automatic(conn, digest_id="dg1", digest_type="morning", recipient="a@b.com", now=D)
    conn.close()

    assert first.id == second.id
    count_conn = database.connect()
    count = count_conn.execute("SELECT COUNT(*) FROM email_deliveries").fetchone()[0]
    count_conn.close()
    assert count == 1


def test_create_automatic_tracks_different_recipients_independently(database: Database):
    _insert_minimal_digest(database, "dg1")
    conn = database.connect()
    repo = EmailDeliveryRepository()
    first = repo.create_automatic(conn, digest_id="dg1", digest_type="morning", recipient="a@b.com", now=D)
    second = repo.create_automatic(conn, digest_id="dg1", digest_type="morning", recipient="c@d.com", now=D)
    conn.close()

    assert first.id != second.id


def test_create_manual_always_inserts_a_fresh_row(database: Database):
    conn = database.connect()
    repo = EmailDeliveryRepository()
    first = repo.create_manual(conn, recipient="a@b.com", now=D)
    second = repo.create_manual(conn, recipient="a@b.com", now=D)
    conn.close()

    assert first.id != second.id
    assert first.digest_id is None
    assert first.digest_type == "newsletter"


def test_mark_sent_updates_status_and_attempt_count(database: Database):
    _insert_minimal_digest(database, "dg1")
    conn = database.connect()
    repo = EmailDeliveryRepository()
    record = repo.create_automatic(conn, digest_id="dg1", digest_type="morning", recipient="a@b.com", now=D)
    repo.mark_sent(conn, delivery_id=record.id, now=D)
    updated = repo.get(conn, record.id)
    conn.close()

    assert updated.status == STATUS_SENT
    assert updated.attempt_count == 1
    assert updated.sent_at == D
    assert updated.failure_reason is None


def test_mark_failed_updates_status_attempt_count_and_reason(database: Database):
    _insert_minimal_digest(database, "dg1")
    conn = database.connect()
    repo = EmailDeliveryRepository()
    record = repo.create_automatic(conn, digest_id="dg1", digest_type="morning", recipient="a@b.com", now=D)
    repo.mark_failed(conn, delivery_id=record.id, now=D, reason="SMTP authentication failed")
    updated = repo.get(conn, record.id)
    conn.close()

    assert updated.status == STATUS_FAILED
    assert updated.attempt_count == 1
    assert updated.failure_reason == "SMTP authentication failed"
    assert updated.sent_at is None


def test_list_for_digest_returns_every_delivery_row(database: Database):
    _insert_minimal_digest(database, "dg1")
    conn = database.connect()
    repo = EmailDeliveryRepository()
    repo.create_automatic(conn, digest_id="dg1", digest_type="morning", recipient="a@b.com", now=D)
    repo.create_manual(conn, recipient="c@d.com", now=D, digest_id="dg1", digest_type="newsletter")
    rows = repo.list_for_digest(conn, "dg1")
    conn.close()

    assert len(rows) == 2
