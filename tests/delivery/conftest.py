"""Shared fixtures for delivery-package tests. Every test uses a fake
`Mailer` -- never a real SMTP connection (Section 11: "Use a fake SMTP
transport or mocked mailer. Never connect to real SMTP servers in tests")."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from research_assistant.config import get_config
from research_assistant.delivery.mailer import MailerSendError
from research_assistant.delivery.renderer import EmailContent

from tests.storage.conftest import database, persistence_service  # noqa: F401


@dataclass
class RecordedSend:
    recipient: str
    content: EmailContent


class FakeMailer:
    """Records every send. `fail=True` makes every `send()` raise
    `MailerSendError`, simulating an SMTP failure without any network I/O."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.sent: List[RecordedSend] = []

    def send(self, *, recipient: str, content: EmailContent) -> None:
        if self.fail:
            raise MailerSendError("simulated SMTP failure")
        self.sent.append(RecordedSend(recipient=recipient, content=content))


@pytest.fixture
def config():
    return get_config()


@pytest.fixture
def fake_mailer():
    return FakeMailer()


@pytest.fixture
def failing_mailer():
    return FakeMailer(fail=True)
