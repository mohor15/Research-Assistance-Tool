"""Delivery orchestration: automatic post-persistence delivery, manual UI
sends, and bounded backfill discovery. This is the only module the rest of
the application (pipeline.runner, scheduling.daemon, ui.app) should import
from `delivery` -- `mailer`/`renderer`/`validation` are internal building
blocks.

None of these functions ever trigger retrieval, an LLM call, or a new
pipeline run; they only read already-persisted digests/events/reports and
send mail. Every entry point swallows and records (never re-raises)
mailer failures, so a broken/unconfigured SMTP setup can never affect
pipeline, run, or digest-persistence success semantics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from research_assistant.config.schema import AppConfig
from research_assistant.models.digest import DigestType
from research_assistant.storage.persistence_service import PersistenceService
from research_assistant.storage.repositories.email_delivery import (
    STATUS_SENT,
    EmailDeliveryRecord,
)

from research_assistant.delivery.mailer import Mailer, MailerError, build_mailer_from_config
from research_assistant.delivery.renderer import render_event_list_email
from research_assistant.delivery.validation import is_valid_email

logger = logging.getLogger(__name__)

# Bounded backfill policy (Section 3, "Already-persisted digests and
# recovery"): startup discovery only ever looks at digests generated
# within this rolling window, capped to this many digests, and retries a
# failed/stuck automatic delivery only up to this many attempts -- it NEVER
# scans or emails the entire historical digest inventory.
BACKFILL_WINDOW_HOURS = 48
BACKFILL_MAX_DIGESTS = 20
BACKFILL_MAX_AUTOMATIC_ATTEMPTS = 5

_DIGEST_TYPE_LABELS = {"morning": "Morning", "evening": "Evening", "current": "CURRENT"}


def get_effective_recipient_and_enabled(service: PersistenceService, config: AppConfig) -> Tuple[str, bool]:
    """The DB-persisted `email_settings` row (Section 5's user-editable
    preference) takes precedence when it exists; otherwise falls back to
    `config.settings.email.default_recipient` with automatic delivery
    enabled by default -- SMTP being configured at all
    (`config.settings.email.enabled`) is checked separately, by
    `build_mailer_from_config`, not here."""
    conn = service.database.connect()
    try:
        record = service.email_settings.get(conn)
    finally:
        conn.close()
    if record is not None:
        return record.default_recipient, record.auto_delivery_enabled
    return config.settings.email.default_recipient, True


def set_default_recipient(service: PersistenceService, recipient: str) -> None:
    if not is_valid_email(recipient):
        raise ValueError(f"Not a valid email address: {recipient!r}")
    conn = service.database.connect()
    try:
        service.email_settings.set_default_recipient(conn, recipient)
    finally:
        conn.close()


def set_auto_delivery_enabled(service: PersistenceService, enabled: bool, config: AppConfig) -> None:
    recipient, _ = get_effective_recipient_and_enabled(service, config)
    conn = service.database.connect()
    try:
        service.email_settings.set_auto_delivery_enabled(conn, enabled, default_recipient=recipient)
    finally:
        conn.close()


def _digest_subject(digest_type: str) -> str:
    label = _DIGEST_TYPE_LABELS.get(digest_type, digest_type.title())
    return f"Nomura AI & Fintech Intelligence — {label} Digest"


def _attempt_send(
    service: PersistenceService, config: AppConfig, *, record: EmailDeliveryRecord, event_ids: List[str],
    subject: str, now: datetime, mailer: Optional[Mailer], tier_b_reports: Optional[dict] = None,
) -> EmailDeliveryRecord:
    resolved_mailer = mailer if mailer is not None else build_mailer_from_config(config)

    if resolved_mailer is None:
        conn = service.database.connect()
        try:
            service.email_deliveries.mark_failed(
                conn, delivery_id=record.id, now=now, reason="email delivery not configured",
            )
            return service.email_deliveries.get(conn, record.id)
        finally:
            conn.close()

    content = render_event_list_email(service, event_ids, subject=subject, config=config, tier_b_reports=tier_b_reports)
    conn = service.database.connect()
    try:
        try:
            resolved_mailer.send(recipient=record.recipient, content=content)
        except MailerError as exc:
            service.email_deliveries.mark_failed(conn, delivery_id=record.id, now=now, reason=str(exc))
        else:
            service.email_deliveries.mark_sent(conn, delivery_id=record.id, now=now)
        return service.email_deliveries.get(conn, record.id)
    finally:
        conn.close()


def deliver_digest_if_eligible(
    service: PersistenceService, config: AppConfig, *, digest_id: str, digest_type: DigestType,
    now: Optional[datetime] = None, mailer: Optional[Mailer] = None,
) -> Optional[EmailDeliveryRecord]:
    """Called exactly once, synchronously, right after `PipelineRunner`
    confirms a digest was persisted WITH qualifying events (never for
    `NO_QUALIFYING_EVENTS`/`PERSISTENCE_FAILED`/`NOT_ATTEMPTED` -- see its
    one call site, `pipeline.runner.PipelineRunner._run_locked`). Any
    failure here is caught and recorded, never re-raised."""
    now = now or datetime.now(timezone.utc)
    recipient, auto_enabled = get_effective_recipient_and_enabled(service, config)
    if not auto_enabled:
        logger.info("email delivery: automatic delivery is disabled; skipping digest_id=%s", digest_id)
        return None

    conn = service.database.connect()
    try:
        digest = service.digests.get(conn, digest_id)
    finally:
        conn.close()
    if digest is None or not digest.events:
        return None

    conn = service.database.connect()
    try:
        record = service.email_deliveries.create_automatic(
            conn, digest_id=digest_id, digest_type=digest_type.value, recipient=recipient, now=now,
        )
    finally:
        conn.close()

    if record.status == STATUS_SENT:
        return record  # already delivered -- the (digest_id, recipient) idempotency key did its job

    return _attempt_send(
        service, config, record=record, event_ids=digest.events, subject=_digest_subject(digest_type.value),
        now=now, mailer=mailer,
    )


def send_manual(
    service: PersistenceService, config: AppConfig, *, event_ids: List[str], recipient: str,
    subject: Optional[str] = None, now: Optional[datetime] = None, mailer: Optional[Mailer] = None,
    tier_b_reports: Optional[dict] = None,
) -> EmailDeliveryRecord:
    """A deliberate, explicit send triggered by the UI's Send button
    (Section 6) -- always creates a NEW `email_deliveries` row (manual
    sends are never deduplicated against a prior send; see
    `EmailDeliveryRepository.create_manual`). Preventing an ACCIDENTAL
    duplicate caused by a double click / Streamlit rerun is the UI's
    responsibility (see `ui/app.py`), not this function's. Uses exactly the
    `event_ids` passed in -- never regenerates or re-selects a newsletter.
    `tier_b_reports`, if given, is an already-resolved dict (see
    `render_event_list_email`'s own docstring) -- this function never
    resolves one itself, so passing it in never triggers a model call
    here, only reuses whatever was already generated for this click."""
    if not is_valid_email(recipient):
        raise ValueError(f"Not a valid email address: {recipient!r}")
    now = now or datetime.now(timezone.utc)
    subject = subject or "Nomura AI & Fintech Intelligence — Newsletter"

    conn = service.database.connect()
    try:
        record = service.email_deliveries.create_manual(conn, recipient=recipient, now=now)
    finally:
        conn.close()

    return _attempt_send(
        service, config, record=record, event_ids=event_ids, subject=subject, now=now, mailer=mailer,
        tier_b_reports=tier_b_reports,
    )


def discover_and_deliver_pending(
    service: PersistenceService, config: AppConfig, *, now: Optional[datetime] = None, mailer: Optional[Mailer] = None,
    window_hours: int = BACKFILL_WINDOW_HOURS, max_digests: int = BACKFILL_MAX_DIGESTS,
    max_attempts: int = BACKFILL_MAX_AUTOMATIC_ATTEMPTS,
) -> List[EmailDeliveryRecord]:
    """Bounded recovery for digests that became persisted/eligible before
    this delivery mechanism observed them -- a feature-deploy gap, or a
    prior SMTP failure (Section 3, "Already-persisted digests and
    recovery"). Deliberately bounded on two axes so a restart/deploy can
    NEVER trigger a bulk historical mailing: only digests generated within
    the last `window_hours`, capped to the `max_digests` most recently
    generated ones. A delivery already `sent` is left untouched; any other
    (pending/failed, including one orphaned by a crash between creation and
    the send attempt) is retried, up to `max_attempts` total attempts.
    Intended to be called once, at process startup, from the ONE
    long-running process (`scheduling.daemon`) so Streamlit and the daemon
    never run competing discovery workers concurrently (Section 8)."""
    now = now or datetime.now(timezone.utc)
    recipient, auto_enabled = get_effective_recipient_and_enabled(service, config)
    if not auto_enabled:
        return []

    cutoff = now - timedelta(hours=window_hours)
    conn = service.database.connect()
    try:
        recent_digests = service.digests.get_recent(conn, limit=max_digests)
    finally:
        conn.close()

    results: List[EmailDeliveryRecord] = []
    for digest in recent_digests:
        if digest.generated_at < cutoff or not digest.events:
            continue

        conn = service.database.connect()
        try:
            existing = service.email_deliveries.get_automatic(conn, digest_id=digest.id, recipient=recipient)
        finally:
            conn.close()

        if existing is not None:
            if existing.status == STATUS_SENT:
                continue
            if existing.attempt_count >= max_attempts:
                continue

        result = deliver_digest_if_eligible(
            service, config, digest_id=digest.id, digest_type=digest.digest_type, now=now, mailer=mailer,
        )
        if result is not None:
            results.append(result)
    return results
