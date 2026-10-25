"""`scheduling.daemon._run_email_backfill_once`: the one-time, startup-only
bounded backfill trigger (Section 3). Exercised directly against a stub
scheduler -- no real APScheduler, no real SMTP, no subprocess (the real
daemon subprocess itself is covered by `test_daemon_process.py`)."""

from __future__ import annotations

from research_assistant.scheduling.daemon import _run_email_backfill_once


class _StubScheduler:
    def __init__(self, *, persistence_service=None, config=None):
        self.persistence_service = persistence_service
        self.config = config


def test_noop_when_no_persistence_service_configured(monkeypatch):
    called = []
    monkeypatch.setattr(
        "research_assistant.delivery.service.discover_and_deliver_pending", lambda *a, **k: called.append(1),
    )

    _run_email_backfill_once(_StubScheduler(persistence_service=None))

    assert called == []


def test_calls_discover_and_deliver_pending_exactly_once(monkeypatch, config):
    called = []
    monkeypatch.setattr(
        "research_assistant.delivery.service.discover_and_deliver_pending",
        lambda service, cfg: called.append((service, cfg)),
    )
    stub_service = object()

    _run_email_backfill_once(_StubScheduler(persistence_service=stub_service, config=config))

    assert len(called) == 1
    assert called[0] == (stub_service, config)


def test_a_raised_exception_is_swallowed_scheduler_remains_unaffected(monkeypatch, config):
    def boom(*_args, **_kwargs):
        raise RuntimeError("discovery exploded")

    monkeypatch.setattr("research_assistant.delivery.service.discover_and_deliver_pending", boom)

    _run_email_backfill_once(_StubScheduler(persistence_service=object(), config=config))  # must not raise
