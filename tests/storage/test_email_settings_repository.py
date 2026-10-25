"""EmailSettingsRepository: singleton-row preference, like `scheduler_state`."""

from __future__ import annotations

from research_assistant.storage.database import Database
from research_assistant.storage.repositories.email_settings import EmailSettingsRepository


def test_get_returns_none_before_any_write(database: Database):
    conn = database.connect()
    record = EmailSettingsRepository().get(conn)
    conn.close()
    assert record is None


def test_set_default_recipient_then_get_round_trips(database: Database):
    conn = database.connect()
    repo = EmailSettingsRepository()
    repo.set_default_recipient(conn, "a@b.com")
    record = repo.get(conn)
    conn.close()

    assert record.default_recipient == "a@b.com"
    assert record.auto_delivery_enabled is True  # default on first write


def test_set_auto_delivery_enabled_preserves_recipient(database: Database):
    conn = database.connect()
    repo = EmailSettingsRepository()
    repo.set_default_recipient(conn, "a@b.com")
    repo.set_auto_delivery_enabled(conn, False, default_recipient="a@b.com")
    record = repo.get(conn)
    conn.close()

    assert record.auto_delivery_enabled is False
    assert record.default_recipient == "a@b.com"


def test_set_default_recipient_does_not_clobber_the_enabled_flag(database: Database):
    conn = database.connect()
    repo = EmailSettingsRepository()
    repo.set_default_recipient(conn, "a@b.com")
    repo.set_auto_delivery_enabled(conn, False, default_recipient="a@b.com")
    repo.set_default_recipient(conn, "c@d.com")
    record = repo.get(conn)
    conn.close()

    assert record.default_recipient == "c@d.com"
    assert record.auto_delivery_enabled is False


def test_repeated_writes_remain_a_single_row(database: Database):
    conn = database.connect()
    repo = EmailSettingsRepository()
    repo.set_default_recipient(conn, "a@b.com")
    repo.set_default_recipient(conn, "b@c.com")
    repo.set_auto_delivery_enabled(conn, False, default_recipient="b@c.com")

    count = conn.execute("SELECT COUNT(*) FROM email_settings").fetchone()[0]
    conn.close()
    assert count == 1
