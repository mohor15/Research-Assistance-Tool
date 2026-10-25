"""The dashboard-facing email-delivery wrappers `ui/app.py` uses (Sections
5-6): recipient preference get/set, address validation, and a manual send
-- none of which may ever trigger a pipeline run or real SMTP connection."""

from __future__ import annotations

from research_assistant.config import get_config

import research_assistant.ui.dashboard_data as dd

from tests.storage.conftest import make_article, make_event, make_report, make_research_item


def _persist_one_event(persistence_service):
    article = make_article("art1")
    item = make_research_item("ri1", article_id=article.id)
    event = make_event("ev1", research_item_ids=[item.id], article_ids=[article.id])
    report = make_report("ev1")
    persistence_service.persist_pipeline_result(articles=[article], research_items=[item], events=[event], reports=[report])


def test_get_email_settings_view_defaults_to_config(persistence_service):
    config = get_config()
    view = dd.get_email_settings_view(persistence_service, config)

    assert view.default_recipient == "ghosalmohor@gmail.com"
    assert view.auto_delivery_enabled is True
    assert view.smtp_configured is False  # nothing is configured in settings.yaml by default


def test_set_email_default_recipient_updates_the_view(persistence_service):
    config = get_config()
    dd.set_email_default_recipient(persistence_service, "someone@example.com")

    view = dd.get_email_settings_view(persistence_service, config)
    assert view.default_recipient == "someone@example.com"


def test_set_email_auto_delivery_enabled_updates_the_view(persistence_service):
    config = get_config()
    dd.set_email_auto_delivery_enabled(persistence_service, False, config)

    view = dd.get_email_settings_view(persistence_service, config)
    assert view.auto_delivery_enabled is False


def test_is_valid_email_address_accepts_and_rejects_correctly():
    assert dd.is_valid_email_address("a@b.com")
    assert not dd.is_valid_email_address("not-an-email")


def test_send_newsletter_email_records_a_safe_failure_when_smtp_not_configured(persistence_service):
    config = get_config()
    _persist_one_event(persistence_service)

    record = dd.send_newsletter_email(persistence_service, config, event_ids=["ev1"], recipient="x@y.com")

    assert record.status == "failed"
    assert "not configured" in record.failure_reason


def test_sending_or_configuring_email_never_triggers_a_pipeline_run(persistence_service):
    config = get_config()
    _persist_one_event(persistence_service)

    dd.get_email_settings_view(persistence_service, config)
    dd.set_email_default_recipient(persistence_service, "a@b.com")
    dd.set_email_auto_delivery_enabled(persistence_service, True, config)
    dd.send_newsletter_email(persistence_service, config, event_ids=["ev1"], recipient="a@b.com")

    conn = persistence_service.database.connect()
    try:
        count = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
    finally:
        conn.close()
    assert count == 0
