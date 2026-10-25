"""Regression: newsletter generation/selection/viewing must never depend on
the email-delivery subsystem being configured, available, or even
functioning. Before this fix, `_render_newsletter_section` called
`_render_automatic_delivery_settings` (which calls
`dd.get_email_settings_view`) unconditionally, with no error boundary --
any exception there (missing attribute, a raising repository, etc.)
propagated up and broke newsletter generation entirely. The fix wraps both
email-specific rendering calls (`_render_automatic_delivery_settings`,
`_render_newsletter_send_controls`) in their own isolation boundary so a
failure is caught and reported in the email section itself, never allowed
to prevent the user from selecting, generating, or viewing a newsletter.
"""

from __future__ import annotations

import research_assistant.ui.app as app_module
import research_assistant.ui.dashboard_data as dd
from research_assistant.config import get_config

from tests.storage.conftest import make_article, make_event, make_report, make_research_item


def _persist_one_selectable_event(persistence_service):
    article = make_article("art1")
    item = make_research_item("ri1", article_id=article.id)
    event = make_event("ev1", research_item_ids=[item.id], article_ids=[article.id])
    report = make_report("ev1")
    persistence_service.persist_pipeline_result(articles=[article], research_items=[item], events=[event], reports=[report])
    dd.toggle_newsletter_selection(persistence_service, "ev1")


def test_newsletter_section_renders_when_email_delivery_is_disabled(persistence_service):
    """Default settings.yaml has `email.enabled: false` -- the common case."""
    config = get_config()
    assert config.settings.email.enabled is False
    _persist_one_selectable_event(persistence_service)

    app_module._render_newsletter_section(persistence_service, config)  # must not raise


def test_newsletter_section_renders_when_smtp_credentials_are_missing(persistence_service, monkeypatch):
    config = get_config().model_copy(deep=True)
    config.settings.email.enabled = True
    config.settings.email.smtp_host = "smtp.example.com"
    config.settings.email.sender_address = "noreply@example.com"
    config.settings.email.smtp_username = "bot@example.com"
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    _persist_one_selectable_event(persistence_service)

    app_module._render_newsletter_section(persistence_service, config)  # must not raise


def test_newsletter_section_renders_when_email_settings_lookup_raises(persistence_service, monkeypatch):
    config = get_config()
    _persist_one_selectable_event(persistence_service)

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated email-settings repository failure")

    monkeypatch.setattr(dd, "get_email_settings_view", boom)

    app_module._render_newsletter_section(persistence_service, config)  # must not raise


def test_newsletter_section_renders_when_get_email_settings_view_is_missing_entirely(persistence_service, monkeypatch):
    """Reproduces the exact reported regression: `dd.get_email_settings_view`
    unavailable (e.g. an attribute error from a stale/partial import) must
    not break newsletter generation."""
    config = get_config()
    _persist_one_selectable_event(persistence_service)

    monkeypatch.delattr(dd, "get_email_settings_view", raising=False)

    app_module._render_newsletter_section(persistence_service, config)  # must not raise with AttributeError


def test_email_specific_failure_is_reported_not_swallowed_silently(persistence_service, monkeypatch):
    config = get_config()
    _persist_one_selectable_event(persistence_service)

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated email-settings repository failure")

    monkeypatch.setattr(dd, "get_email_settings_view", boom)

    errors = []
    monkeypatch.setattr(app_module.st, "error", lambda message: errors.append(message))

    app_module._render_newsletter_section(persistence_service, config)

    assert any("email" in message.lower() for message in errors)


def test_unrelated_newsletter_generation_errors_are_not_swallowed(persistence_service):
    """The isolation boundary covers ONLY the email-rendering calls -- a
    real failure in `dd.get_event_detail`/newsletter rendering itself must
    still propagate normally, not be silently hidden."""
    config = get_config()
    _persist_one_selectable_event(persistence_service)

    import pytest

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated unrelated newsletter failure")

    import unittest.mock as mock

    with mock.patch.object(dd, "get_newsletter_selection", boom):
        with pytest.raises(RuntimeError, match="simulated unrelated newsletter failure"):
            app_module._render_newsletter_section(persistence_service, config)


def test_email_settings_still_render_and_can_be_saved_when_configured_correctly(persistence_service):
    config = get_config()
    _persist_one_selectable_event(persistence_service)

    app_module._render_automatic_delivery_settings_isolated(persistence_service, config)  # must not raise

    dd.set_email_default_recipient(persistence_service, "new-recipient@example.com")
    view = dd.get_email_settings_view(persistence_service, config)
    assert view.default_recipient == "new-recipient@example.com"


def test_manual_send_controls_still_work_when_correctly_configured(persistence_service):
    from research_assistant.delivery.service import send_manual

    class _FakeMailer:
        def __init__(self):
            self.sent = []

        def send(self, *, recipient, content):
            self.sent.append((recipient, content))

    config = get_config()
    _persist_one_selectable_event(persistence_service)
    mailer = _FakeMailer()

    record = send_manual(persistence_service, config, event_ids=["ev1"], recipient="someone@example.com", mailer=mailer)

    assert record.status == "sent"
    assert mailer.sent[0][0] == "someone@example.com"
