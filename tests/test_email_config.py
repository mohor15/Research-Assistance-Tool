"""Email-delivery configuration (Section 4): safe defaults, validation,
and the credential-is-env-var-only contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_assistant.config.loader import load_settings, load_smtp_password
from research_assistant.config.schema import EmailSettings


def test_default_email_settings_disabled_with_ghosalmohor_recipient():
    settings = EmailSettings()
    assert settings.enabled is False
    assert settings.default_recipient == "ghosalmohor@gmail.com"


def test_invalid_default_recipient_rejected():
    with pytest.raises(ValidationError):
        EmailSettings(default_recipient="not-an-email")


def test_invalid_sender_address_rejected():
    with pytest.raises(ValidationError):
        EmailSettings(sender_address="not-an-email")


def test_empty_sender_address_is_allowed_not_required():
    settings = EmailSettings(sender_address="")
    assert settings.sender_address == ""


def test_real_settings_yaml_loads_an_email_section_with_safe_defaults():
    settings = load_settings()
    assert settings.email.default_recipient == "ghosalmohor@gmail.com"
    assert settings.email.enabled is False


def test_load_smtp_password_reads_from_the_environment(monkeypatch):
    monkeypatch.setenv("SMTP_PASSWORD", "secret123")
    assert load_smtp_password(env_path=None) == "secret123"


def test_load_smtp_password_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    assert load_smtp_password(env_path=None) is None


def test_load_smtp_password_treats_empty_string_as_unset(monkeypatch):
    monkeypatch.setenv("SMTP_PASSWORD", "")
    assert load_smtp_password(env_path=None) is None
