"""SMTP transport for email delivery. `SmtpMailer` is the only thing in
this project that ever opens a real network connection to send mail --
every test uses a fake `Mailer` instead (Section 11: "Use a fake SMTP
transport or mocked mailer. Never connect to real SMTP servers in tests").

Credentials: the SMTP password is read once, at mailer-construction time,
from `config.loader.load_smtp_password()` (the `SMTP_PASSWORD` environment
variable) -- never from settings.yaml, never logged, and never attached to
any exception message or delivery-tracking record (see
`storage.repositories.email_delivery`, whose `failure_reason` column only
ever receives the sanitized messages raised here).
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Protocol

from research_assistant.config.loader import load_smtp_password
from research_assistant.config.schema import AppConfig
from research_assistant.delivery.renderer import EmailContent

logger = logging.getLogger(__name__)


class MailerError(Exception):
    """Base class for every delivery-layer send failure. Every message
    raised anywhere in this module is already sanitized -- never includes
    the password or a raw SMTP server response."""


class MailerSendError(MailerError):
    pass


class Mailer(Protocol):
    def send(self, *, recipient: str, content: EmailContent) -> None: ...


class SmtpMailer:
    def __init__(
        self, *, host: str, port: int, use_tls: bool, username: str, password: Optional[str], sender: str,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._host = host
        self._port = port
        self._use_tls = use_tls
        self._username = username
        self._password = password
        self._sender = sender
        self._timeout_seconds = timeout_seconds

    def send(self, *, recipient: str, content: EmailContent) -> None:
        message = MIMEMultipart("alternative")
        message["Subject"] = content.subject
        message["From"] = self._sender
        message["To"] = recipient
        # Plain text first, HTML second: RFC 2046 §5.1.4 requires
        # multipart/alternative parts in order of increasing preference, so
        # a client that can't render HTML correctly falls back to text.
        message.attach(MIMEText(content.text_body, "plain", "utf-8"))
        message.attach(MIMEText(content.html_body, "html", "utf-8"))

        try:
            with smtplib.SMTP(self._host, self._port, timeout=self._timeout_seconds) as server:
                if self._use_tls:
                    server.starttls(context=ssl.create_default_context())
                if self._username:
                    server.login(self._username, self._password or "")
                server.sendmail(self._sender, [recipient], message.as_string())
        except smtplib.SMTPAuthenticationError as exc:
            logger.error("email delivery: SMTP authentication failed (host=%s port=%s)", self._host, self._port)
            raise MailerSendError("SMTP authentication failed") from exc
        except (smtplib.SMTPException, OSError) as exc:
            logger.error("email delivery: SMTP send failed: %s", type(exc).__name__)
            raise MailerSendError(f"SMTP send failed: {type(exc).__name__}") from exc


def build_mailer_from_config(config: AppConfig) -> Optional[Mailer]:
    """Returns `None` -- never raises -- whenever delivery is disabled or
    SMTP is not safely configured, so a missing/incomplete configuration
    degrades to "delivery stays off" rather than an application crash
    (Section 4: "the existing application must still work normally")."""
    email_settings = config.settings.email
    if not email_settings.enabled:
        return None
    if not email_settings.smtp_host or not email_settings.sender_address:
        logger.warning("email delivery: enabled but smtp_host/sender_address is not configured; delivery disabled")
        return None
    password = load_smtp_password()
    if email_settings.smtp_username and not password:
        logger.warning("email delivery: smtp_username is set but SMTP_PASSWORD is not configured; delivery disabled")
        return None
    return SmtpMailer(
        host=email_settings.smtp_host, port=email_settings.smtp_port, use_tls=email_settings.smtp_use_tls,
        username=email_settings.smtp_username, password=password, sender=email_settings.sender_address,
    )
