"""Delivery orchestration: automatic post-persistence delivery, manual UI
sends, bounded backfill discovery, and the recipient/auto-delivery
preference. Every test uses a fake `Mailer` -- no real network, no real
SMTP server, no live pipeline run."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from research_assistant.delivery import service as delivery_service
from research_assistant.models.digest import Digest, DigestType
from research_assistant.storage.repositories.email_delivery import STATUS_FAILED, STATUS_SENT

from tests.storage.conftest import make_article, make_digest, make_event, make_report, make_research_item

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def _persist_digest_with_one_event(persistence_service, *, digest_id="dg1", digest_type=DigestType.MORNING, events=True):
    article = make_article("art1")
    item = make_research_item("ri1", article_id=article.id)
    event = make_event("ev1", research_item_ids=[item.id], article_ids=[article.id])
    report = make_report("ev1")
    if events:
        digest = make_digest(digest_id, event_ids=["ev1"], digest_type=digest_type)
    else:
        # `make_digest`'s `event_ids = event_ids or ["ev1"]` treats an
        # explicit `[]` as falsy too -- build the zero-events Digest
        # directly here to actually get an empty `events` list.
        digest = Digest(
            id=digest_id, digest_type=digest_type, generated_at=D, period_start=D, period_end=D, events=[],
        )
    persistence_service.persist_pipeline_result(
        articles=[article], research_items=[item], events=[event], reports=[report], digest=digest,
    )
    return digest


def _persist_digest_at(persistence_service, *, digest_id: str, generated_at: datetime):
    article = make_article(f"art-{digest_id}")
    item = make_research_item(f"ri-{digest_id}", article_id=article.id)
    event = make_event(f"ev-{digest_id}", research_item_ids=[item.id], article_ids=[article.id])
    report = make_report(f"ev-{digest_id}")
    digest = Digest(
        id=digest_id, digest_type=DigestType.MORNING, generated_at=generated_at,
        period_start=generated_at, period_end=generated_at, events=[f"ev-{digest_id}"],
    )
    persistence_service.persist_pipeline_result(
        articles=[article], research_items=[item], events=[event], reports=[report], digest=digest,
    )
    return digest


class TestDeliverDigestIfEligible:
    def test_sends_once_for_a_newly_persisted_digest(self, persistence_service, config, fake_mailer):
        digest = _persist_digest_with_one_event(persistence_service)

        record = delivery_service.deliver_digest_if_eligible(
            persistence_service, config, digest_id=digest.id, digest_type=digest.digest_type, now=D, mailer=fake_mailer,
        )

        assert record.status == STATUS_SENT
        assert len(fake_mailer.sent) == 1
        assert fake_mailer.sent[0].recipient == "ghosalmohor@gmail.com"

    def test_does_not_send_twice_for_the_same_digest_and_recipient(self, persistence_service, config, fake_mailer):
        digest = _persist_digest_with_one_event(persistence_service)

        delivery_service.deliver_digest_if_eligible(persistence_service, config, digest_id=digest.id, digest_type=digest.digest_type, now=D, mailer=fake_mailer)
        delivery_service.deliver_digest_if_eligible(persistence_service, config, digest_id=digest.id, digest_type=digest.digest_type, now=D, mailer=fake_mailer)

        assert len(fake_mailer.sent) == 1

    def test_two_recipients_are_tracked_independently(self, persistence_service, config, fake_mailer):
        digest = _persist_digest_with_one_event(persistence_service)
        delivery_service.set_default_recipient(persistence_service, "first@example.com")
        delivery_service.deliver_digest_if_eligible(persistence_service, config, digest_id=digest.id, digest_type=digest.digest_type, now=D, mailer=fake_mailer)
        delivery_service.set_default_recipient(persistence_service, "second@example.com")
        delivery_service.deliver_digest_if_eligible(persistence_service, config, digest_id=digest.id, digest_type=digest.digest_type, now=D, mailer=fake_mailer)

        recipients = {s.recipient for s in fake_mailer.sent}
        assert recipients == {"first@example.com", "second@example.com"}

    def test_digest_with_no_qualifying_events_is_never_delivered(self, persistence_service, config, fake_mailer):
        digest = _persist_digest_with_one_event(persistence_service, events=False)

        record = delivery_service.deliver_digest_if_eligible(persistence_service, config, digest_id=digest.id, digest_type=digest.digest_type, now=D, mailer=fake_mailer)

        assert record is None
        assert fake_mailer.sent == []

    def test_unknown_digest_id_is_never_delivered(self, persistence_service, config, fake_mailer):
        record = delivery_service.deliver_digest_if_eligible(
            persistence_service, config, digest_id="does-not-exist", digest_type=DigestType.MORNING, now=D, mailer=fake_mailer,
        )
        assert record is None
        assert fake_mailer.sent == []

    def test_automatic_delivery_disabled_prevents_sending(self, persistence_service, config, fake_mailer):
        digest = _persist_digest_with_one_event(persistence_service)
        delivery_service.set_auto_delivery_enabled(persistence_service, False, config)

        record = delivery_service.deliver_digest_if_eligible(persistence_service, config, digest_id=digest.id, digest_type=digest.digest_type, now=D, mailer=fake_mailer)

        assert record is None
        assert fake_mailer.sent == []

    def test_mailer_failure_is_recorded_and_never_raised(self, persistence_service, config, failing_mailer):
        digest = _persist_digest_with_one_event(persistence_service)

        record = delivery_service.deliver_digest_if_eligible(persistence_service, config, digest_id=digest.id, digest_type=digest.digest_type, now=D, mailer=failing_mailer)

        assert record.status == STATUS_FAILED
        assert record.failure_reason

    def test_no_mailer_configured_records_a_clear_failure_without_raising(self, persistence_service, config):
        digest = _persist_digest_with_one_event(persistence_service)

        record = delivery_service.deliver_digest_if_eligible(persistence_service, config, digest_id=digest.id, digest_type=digest.digest_type, now=D)

        assert record.status == STATUS_FAILED
        assert "not configured" in record.failure_reason


class TestDiscoverAndDeliverPending:
    def test_discovers_a_persisted_digest_with_no_prior_delivery_attempt(self, persistence_service, config, fake_mailer):
        _persist_digest_with_one_event(persistence_service)

        results = delivery_service.discover_and_deliver_pending(persistence_service, config, now=D, mailer=fake_mailer)

        assert len(results) == 1
        assert results[0].status == STATUS_SENT

    def test_does_not_redeliver_an_already_sent_digest(self, persistence_service, config, fake_mailer):
        digest = _persist_digest_with_one_event(persistence_service)
        delivery_service.deliver_digest_if_eligible(persistence_service, config, digest_id=digest.id, digest_type=digest.digest_type, now=D, mailer=fake_mailer)

        results = delivery_service.discover_and_deliver_pending(persistence_service, config, now=D, mailer=fake_mailer)

        assert results == []
        assert len(fake_mailer.sent) == 1

    def test_retries_a_previously_failed_delivery(self, persistence_service, config, failing_mailer, fake_mailer):
        digest = _persist_digest_with_one_event(persistence_service)
        delivery_service.deliver_digest_if_eligible(persistence_service, config, digest_id=digest.id, digest_type=digest.digest_type, now=D, mailer=failing_mailer)

        results = delivery_service.discover_and_deliver_pending(persistence_service, config, now=D, mailer=fake_mailer)

        assert len(results) == 1
        assert results[0].status == STATUS_SENT

    def test_never_retries_beyond_the_max_attempts_bound(self, persistence_service, config, failing_mailer):
        digest = _persist_digest_with_one_event(persistence_service)
        delivery_service.deliver_digest_if_eligible(persistence_service, config, digest_id=digest.id, digest_type=digest.digest_type, now=D, mailer=failing_mailer)

        max_attempts = 3
        for _ in range(max_attempts + 3):
            delivery_service.discover_and_deliver_pending(persistence_service, config, now=D, mailer=failing_mailer, max_attempts=max_attempts)

        conn = persistence_service.database.connect()
        try:
            record = persistence_service.email_deliveries.get_automatic(conn, digest_id=digest.id, recipient="ghosalmohor@gmail.com")
        finally:
            conn.close()
        assert record.attempt_count == max_attempts

    def test_respects_the_bounded_time_window(self, persistence_service, config, fake_mailer):
        _persist_digest_at(persistence_service, digest_id="old-dg", generated_at=D - timedelta(hours=100))

        results = delivery_service.discover_and_deliver_pending(persistence_service, config, now=D, mailer=fake_mailer, window_hours=48)

        assert results == []
        assert fake_mailer.sent == []

    def test_discovers_a_digest_within_the_window(self, persistence_service, config, fake_mailer):
        _persist_digest_at(persistence_service, digest_id="recent-dg", generated_at=D - timedelta(hours=1))

        results = delivery_service.discover_and_deliver_pending(persistence_service, config, now=D, mailer=fake_mailer, window_hours=48)

        assert len(results) == 1

    def test_disabled_auto_delivery_discovers_nothing(self, persistence_service, config, fake_mailer):
        _persist_digest_with_one_event(persistence_service)
        delivery_service.set_auto_delivery_enabled(persistence_service, False, config)

        results = delivery_service.discover_and_deliver_pending(persistence_service, config, now=D, mailer=fake_mailer)

        assert results == []
        assert fake_mailer.sent == []


class TestSendManual:
    def test_sends_to_entered_recipient_using_current_selection(self, persistence_service, config, fake_mailer):
        _persist_digest_with_one_event(persistence_service)

        record = delivery_service.send_manual(persistence_service, config, event_ids=["ev1"], recipient="someone@example.com", mailer=fake_mailer)

        assert record.status == STATUS_SENT
        assert fake_mailer.sent[0].recipient == "someone@example.com"

    def test_rejects_invalid_recipient(self, persistence_service, config, fake_mailer):
        _persist_digest_with_one_event(persistence_service)
        with pytest.raises(ValueError):
            delivery_service.send_manual(persistence_service, config, event_ids=["ev1"], recipient="not-an-email", mailer=fake_mailer)
        assert fake_mailer.sent == []

    def test_deliberate_resend_creates_a_new_record_each_time(self, persistence_service, config, fake_mailer):
        _persist_digest_with_one_event(persistence_service)

        first = delivery_service.send_manual(persistence_service, config, event_ids=["ev1"], recipient="someone@example.com", mailer=fake_mailer)
        second = delivery_service.send_manual(persistence_service, config, event_ids=["ev1"], recipient="someone@example.com", mailer=fake_mailer)

        assert first.id != second.id
        assert len(fake_mailer.sent) == 2

    def test_failure_is_recorded_and_never_raised(self, persistence_service, config, failing_mailer):
        _persist_digest_with_one_event(persistence_service)

        record = delivery_service.send_manual(persistence_service, config, event_ids=["ev1"], recipient="someone@example.com", mailer=failing_mailer)

        assert record.status == STATUS_FAILED

    def test_tier_b_reports_passthrough_reaches_the_rendered_email_without_regenerating(
        self, persistence_service, config, fake_mailer,
    ):
        """An already-resolved Tier B report dict (what the UI builds once
        per "Generate Newsletter" click) must reach the rendered email
        content as-is -- `send_manual` must never resolve/generate one
        itself."""
        from research_assistant.models.enums import BacklogStatus, SourcePriority, SourceType
        from research_assistant.models.report import ReportSource, TierBReport
        from research_assistant.reporting.tier_b_newsletter_resolution import TierBReportResolution
        from research_assistant.storage.repositories.backlog import BacklogRow
        from research_assistant.ui.dashboard_data import TIER_B_SELECTION_PREFIX

        conn = persistence_service.database.connect()
        persistence_service.articles.upsert(conn, make_article("art-tb1", company="Acme AI", url="https://acme.example/art-tb1"))
        persistence_service.backlog.upsert(conn, BacklogRow(
            article_id="art-tb1", agent="startup", status=BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value,
            reason="test", attempt_count=0, defer_count=1, first_seen_run_id="run1", first_seen_at=D,
            last_attempt_run_id="run1", last_attempt_at=D, triage_relevant=True, triage_confidence=0.9,
            research_item_id=None, updated_at=D,
        ))
        conn.commit()
        conn.close()

        report = TierBReport(
            article_id="art-tb1", title="Acme headline", opening_context="A resolved opening context.",
            key_insights=["Resolved insight one.", "Resolved insight two."],
            strategic_implications="Resolved strategic implications.",
            nomura_opportunity="This may be relevant to Nomura's consideration of this area.",
            source=ReportSource(
                publisher="Acme AI", title="Acme headline", publication_date=D, url="https://acme.example/art-tb1",
                source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
            ),
        )
        resolutions = {"art-tb1": TierBReportResolution(status="generated", report=report)}

        delivery_service.send_manual(
            persistence_service, config, event_ids=[f"{TIER_B_SELECTION_PREFIX}art-tb1"],
            recipient="someone@example.com", mailer=fake_mailer, tier_b_reports=resolutions,
        )

        assert "A resolved opening context." in fake_mailer.sent[0].content.html_body
        assert "Resolved insight one." in fake_mailer.sent[0].content.text_body


class TestRecipientPreference:
    def test_default_recipient_is_ghosalmohor_when_no_preference_saved_yet(self, persistence_service, config):
        recipient, enabled = delivery_service.get_effective_recipient_and_enabled(persistence_service, config)
        assert recipient == "ghosalmohor@gmail.com"
        assert enabled is True

    def test_set_default_recipient_persists_across_calls(self, persistence_service, config):
        delivery_service.set_default_recipient(persistence_service, "new@example.com")

        recipient, _ = delivery_service.get_effective_recipient_and_enabled(persistence_service, config)

        assert recipient == "new@example.com"

    def test_set_default_recipient_rejects_an_invalid_address(self, persistence_service, config):
        with pytest.raises(ValueError):
            delivery_service.set_default_recipient(persistence_service, "not-an-email")

    def test_manual_send_to_a_different_address_does_not_change_the_saved_default(self, persistence_service, config, fake_mailer):
        _persist_digest_with_one_event(persistence_service)
        delivery_service.send_manual(persistence_service, config, event_ids=["ev1"], recipient="someone-else@example.com", mailer=fake_mailer)

        recipient, _ = delivery_service.get_effective_recipient_and_enabled(persistence_service, config)

        assert recipient == "ghosalmohor@gmail.com"
