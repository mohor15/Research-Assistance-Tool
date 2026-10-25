"""`build_mailer_from_config`: safe degrade-to-disabled behavior for every
missing/incomplete configuration case -- never raises, never connects to a
real SMTP server. `SmtpMailer.send` itself is exercised with a monkeypatched
`smtplib.SMTP` (never a real connection) to confirm failures are sanitized
and surfaced as `MailerSendError`, never left as a raw SMTP exception that
could carry a password or full server response."""

from __future__ import annotations

import smtplib

import pytest

from research_assistant.delivery.mailer import MailerSendError, SmtpMailer, build_mailer_from_config
from research_assistant.delivery.renderer import EmailContent


def _content() -> EmailContent:
    return EmailContent(subject="Subject", html_body="<p>hi</p>", text_body="hi")


def _configured(config):
    cfg = config.model_copy(deep=True)
    cfg.settings.email.enabled = True
    cfg.settings.email.smtp_host = "smtp.example.com"
    cfg.settings.email.sender_address = "noreply@example.com"
    cfg.settings.email.smtp_username = ""
    return cfg


def test_returns_none_when_delivery_disabled(config):
    cfg = config.model_copy(deep=True)
    cfg.settings.email.enabled = False
    assert build_mailer_from_config(cfg) is None


def test_returns_none_when_enabled_but_smtp_host_missing(config):
    cfg = config.model_copy(deep=True)
    cfg.settings.email.enabled = True
    cfg.settings.email.smtp_host = ""
    cfg.settings.email.sender_address = "noreply@example.com"
    assert build_mailer_from_config(cfg) is None


def test_returns_none_when_enabled_but_sender_address_missing(config):
    cfg = config.model_copy(deep=True)
    cfg.settings.email.enabled = True
    cfg.settings.email.smtp_host = "smtp.example.com"
    cfg.settings.email.sender_address = ""
    assert build_mailer_from_config(cfg) is None


def test_returns_none_when_username_set_but_password_env_var_missing(config, monkeypatch):
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    cfg = _configured(config)
    cfg.settings.email.smtp_username = "bot@example.com"
    assert build_mailer_from_config(cfg) is None


def test_returns_a_mailer_when_fully_configured_with_no_auth(config):
    cfg = _configured(config)
    mailer = build_mailer_from_config(cfg)
    assert isinstance(mailer, SmtpMailer)


def test_returns_a_mailer_when_username_and_password_both_present(config, monkeypatch):
    monkeypatch.setenv("SMTP_PASSWORD", "secret-value")
    cfg = _configured(config)
    cfg.settings.email.smtp_username = "bot@example.com"
    mailer = build_mailer_from_config(cfg)
    assert isinstance(mailer, SmtpMailer)


def test_credentials_are_never_logged(config, monkeypatch, caplog):
    monkeypatch.setenv("SMTP_PASSWORD", "super-secret-password")
    cfg = _configured(config)
    cfg.settings.email.smtp_username = "bot@example.com"
    build_mailer_from_config(cfg)
    assert "super-secret-password" not in caplog.text


class _FakeSmtpSuccess:
    def __init__(self, *_args, **_kwargs):
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def starttls(self, **_kwargs):
        pass

    def login(self, *_args):
        pass

    def sendmail(self, sender, recipients, message):
        self.sent.append((sender, recipients, message))


class _FakeSmtpAuthFailure(_FakeSmtpSuccess):
    def login(self, *_args):
        raise smtplib.SMTPAuthenticationError(535, b"bad credentials")


class _FakeSmtpConnectionFailure(_FakeSmtpSuccess):
    def __enter__(self):
        raise OSError("connection refused")


def test_smtp_mailer_send_succeeds_with_a_fake_transport(monkeypatch):
    monkeypatch.setattr(smtplib, "SMTP", _FakeSmtpSuccess)
    mailer = SmtpMailer(host="h", port=587, use_tls=True, username="", password=None, sender="noreply@example.com")
    mailer.send(recipient="to@example.com", content=_content())  # must not raise


def test_smtp_mailer_wraps_authentication_failure(monkeypatch):
    monkeypatch.setattr(smtplib, "SMTP", _FakeSmtpAuthFailure)
    mailer = SmtpMailer(host="h", port=587, use_tls=True, username="bot", password="wrong", sender="noreply@example.com")
    with pytest.raises(MailerSendError):
        mailer.send(recipient="to@example.com", content=_content())


def test_smtp_mailer_wraps_connection_failure_and_never_leaks_password(monkeypatch):
    monkeypatch.setattr(smtplib, "SMTP", _FakeSmtpConnectionFailure)
    mailer = SmtpMailer(host="h", port=587, use_tls=True, username="bot", password="super-secret", sender="noreply@example.com")
    with pytest.raises(MailerSendError) as excinfo:
        mailer.send(recipient="to@example.com", content=_content())
    assert "super-secret" not in str(excinfo.value)
