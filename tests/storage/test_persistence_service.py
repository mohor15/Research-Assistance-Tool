"""PersistenceService: full-chain coordination, idempotency, and the
historical query methods (recent events, company/event-type/relevance/
confidence filtering, recent reports, recent digests)."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.evidence import article_to_evidence
from research_assistant.models.enums import BacklogStatus, EventType
from research_assistant.pipeline.state import BacklogUpdate
from research_assistant.storage.persistence_service import PersistenceService

from tests.storage.conftest import make_article, make_digest, make_event, make_report, make_research_item


def _full_chain(article_id="art1", event_id="ev1", **event_kwargs):
    article = make_article(article_id)
    evidence = article_to_evidence(article)
    item = make_research_item("ri1", article_id=article_id, evidence_ids=[evidence.evidence_id])
    event = make_event(event_id, research_item_ids=["ri1"], article_ids=[article_id], **event_kwargs)
    report = make_report(event_id)
    digest = make_digest("dg1", event_ids=[event_id])
    return article, evidence, item, event, report, digest


def test_persist_pipeline_result_persists_the_full_chain(persistence_service: PersistenceService):
    article, evidence, item, event, report, digest = _full_chain()

    persistence_service.persist_pipeline_result(
        articles=[article], evidence=[evidence], research_items=[item], events=[event],
        reports=[report], digest=digest,
    )

    conn = persistence_service.database.connect()
    assert persistence_service.articles.get(conn, "art1") is not None
    assert persistence_service.evidence.get(conn, evidence.evidence_id) is not None
    assert persistence_service.research_items.get(conn, "ri1") is not None
    assert persistence_service.events.get(conn, "ev1") is not None
    assert persistence_service.reports.get(conn, "ev1") is not None
    assert persistence_service.digests.get(conn, "dg1") is not None
    conn.close()


def test_persisting_the_same_article_twice_does_not_duplicate(persistence_service: PersistenceService):
    article, evidence, item, event, report, digest = _full_chain()

    persistence_service.persist_pipeline_result(articles=[article])
    persistence_service.persist_pipeline_result(articles=[article])

    conn = persistence_service.database.connect()
    count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()
    assert count == 1


def test_persisting_the_same_event_twice_updates_rather_than_duplicates(persistence_service: PersistenceService):
    article, evidence, item, event, report, digest = _full_chain()

    persistence_service.persist_pipeline_result(articles=[article], research_items=[item], events=[event])
    updated_event = event.model_copy(update={"relevance_score": 0.99})
    persistence_service.persist_pipeline_result(articles=[article], research_items=[item], events=[updated_event])

    conn = persistence_service.database.connect()
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    fetched = persistence_service.events.get(conn, "ev1")
    conn.close()

    assert count == 1
    assert fetched.relevance_score == 0.99


def test_full_pipeline_result_persisted_twice_is_fully_idempotent(persistence_service: PersistenceService):
    article, evidence, item, event, report, digest = _full_chain()

    for _ in range(2):
        persistence_service.persist_pipeline_result(
            articles=[article], evidence=[evidence], research_items=[item], events=[event],
            reports=[report], digest=digest,
        )

    conn = persistence_service.database.connect()
    counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("articles", "evidence", "research_items", "events", "reports", "digests")
    }
    conn.close()

    assert all(count == 1 for count in counts.values()), counts


def test_get_recent_events_query(persistence_service: PersistenceService):
    article, evidence, item, event, report, digest = _full_chain()
    persistence_service.persist_pipeline_result(articles=[article], research_items=[item], events=[event])

    conn = persistence_service.database.connect()
    recent = persistence_service.events.get_recent(conn, limit=10)
    conn.close()

    assert [e.id for e in recent] == ["ev1"]


def test_company_filtering_query(persistence_service: PersistenceService):
    article, evidence, item, event, report, digest = _full_chain(company="JPMorgan Chase")
    persistence_service.persist_pipeline_result(articles=[article], research_items=[item], events=[event])

    conn = persistence_service.database.connect()
    matches = persistence_service.events.get_by_company(conn, "JPMorgan Chase")
    non_matches = persistence_service.events.get_by_company(conn, "OpenAI")
    conn.close()

    assert [e.id for e in matches] == ["ev1"]
    assert non_matches == []


def test_event_type_filtering_query(persistence_service: PersistenceService):
    article, evidence, item, event, report, digest = _full_chain(event_type=EventType.FUNDING)
    persistence_service.persist_pipeline_result(articles=[article], research_items=[item], events=[event])

    conn = persistence_service.database.connect()
    matches = persistence_service.events.get_by_event_type(conn, EventType.FUNDING)
    non_matches = persistence_service.events.get_by_event_type(conn, EventType.ACQUISITION)
    conn.close()

    assert [e.id for e in matches] == ["ev1"]
    assert non_matches == []


def test_relevance_filtering_query(persistence_service: PersistenceService):
    article, evidence, item, event, report, digest = _full_chain(relevance_score=0.4)
    persistence_service.persist_pipeline_result(articles=[article], research_items=[item], events=[event])

    conn = persistence_service.database.connect()
    matches = persistence_service.events.get_by_min_relevance(conn, 0.3)
    non_matches = persistence_service.events.get_by_min_relevance(conn, 0.9)
    conn.close()

    assert [e.id for e in matches] == ["ev1"]
    assert non_matches == []


def test_confidence_filtering_query(persistence_service: PersistenceService):
    article, evidence, item, event, report, digest = _full_chain(confidence_score=0.4)
    persistence_service.persist_pipeline_result(articles=[article], research_items=[item], events=[event])

    conn = persistence_service.database.connect()
    matches = persistence_service.events.get_by_min_confidence(conn, 0.3)
    non_matches = persistence_service.events.get_by_min_confidence(conn, 0.9)
    conn.close()

    assert [e.id for e in matches] == ["ev1"]
    assert non_matches == []


def test_get_recent_reports_query(persistence_service: PersistenceService):
    article, evidence, item, event, report, digest = _full_chain()
    persistence_service.persist_pipeline_result(articles=[article], research_items=[item], events=[event], reports=[report])

    conn = persistence_service.database.connect()
    recent = persistence_service.reports.get_recent(conn, limit=10)
    conn.close()

    assert [r.event_id for r in recent] == ["ev1"]


def test_get_recent_digests_query(persistence_service: PersistenceService):
    article, evidence, item, event, report, digest = _full_chain()
    persistence_service.persist_pipeline_result(
        articles=[article], research_items=[item], events=[event], digest=digest,
    )

    conn = persistence_service.database.connect()
    recent = persistence_service.digests.get_recent(conn, limit=10)
    conn.close()

    assert [d.id for d in recent] == ["dg1"]


def test_persistence_rolls_back_on_failure_without_partial_writes(persistence_service: PersistenceService, monkeypatch):
    article, evidence, item, event, report, digest = _full_chain()

    def _broken_upsert(conn, ev):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(persistence_service.events, "upsert", _broken_upsert)

    import pytest

    with pytest.raises(RuntimeError):
        persistence_service.persist_pipeline_result(articles=[article], research_items=[item], events=[event])

    conn = persistence_service.database.connect()
    # The article/research_item upserts that ran before the failure are
    # rolled back too, since everything happens in one transaction.
    article_count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()
    assert article_count == 0


# ---------------------------------------------------------------------------
# Candidate backlog: a candidate deferred by a budget-exhausted run is
# durably retrievable (with its real, already-persisted Article/Evidence)
# by a later run, and resuming it never resets its original fairness
# position.
# ---------------------------------------------------------------------------


def test_backlog_update_is_persisted_atomically_with_everything_else(persistence_service: PersistenceService):
    article = make_article("art1")
    evidence = article_to_evidence(article)
    update = BacklogUpdate(
        article_id="art1", agent="ai_product", status=BacklogStatus.PENDING_TRIAGE_BUDGET.value,
        reason="triage pool bound",
    )

    persistence_service.persist_pipeline_result(
        articles=[article], evidence=[evidence], backlog_updates=[update], run_id="run1",
    )

    conn = persistence_service.database.connect()
    row = persistence_service.backlog.get(conn, "art1")
    conn.close()

    assert row is not None
    assert row.status == BacklogStatus.PENDING_TRIAGE_BUDGET.value
    assert row.first_seen_run_id == "run1"  # stamped from persist_pipeline_result's run_id, a brand-new row


def test_get_pending_backlog_candidates_reconstructs_real_article_and_evidence(persistence_service: PersistenceService):
    article = make_article("art1")
    evidence = article_to_evidence(article)
    update = BacklogUpdate(
        article_id="art1", agent="ai_product", status=BacklogStatus.PENDING_TRIAGE_BUDGET.value, reason="x",
    )
    persistence_service.persist_pipeline_result(
        articles=[article], evidence=[evidence], backlog_updates=[update], run_id="run1",
    )

    records = persistence_service.get_pending_backlog_candidates(limit=10)

    assert len(records) == 1
    assert records[0].article.id == "art1"
    assert records[0].article.title == article.title  # the real, already-persisted article, not a copy
    assert records[0].evidence.content == evidence.content


def test_resuming_a_backlog_candidate_preserves_its_original_first_seen_time(persistence_service: PersistenceService):
    article = make_article("art1")
    evidence = article_to_evidence(article)
    first_update = BacklogUpdate(
        article_id="art1", agent="ai_product", status=BacklogStatus.PENDING_TRIAGE_BUDGET.value, reason="x",
    )
    persistence_service.persist_pipeline_result(
        articles=[article], evidence=[evidence], backlog_updates=[first_update], run_id="run1",
    )
    conn = persistence_service.database.connect()
    original = persistence_service.backlog.get(conn, "art1")
    conn.close()

    # A later run resumes it (attempt_count incremented) WITHOUT passing
    # first_seen_run_id/first_seen_at explicitly -- a brand-new candidate
    # with no prior row would get run2/now stamped, but because a row
    # already exists, its original first_seen must be preserved.
    second_update = BacklogUpdate(
        article_id="art1", agent="ai_product", status=BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value,
        reason="y", attempt_count=1,
    )
    persistence_service.persist_pipeline_result(backlog_updates=[second_update], run_id="run2")

    conn = persistence_service.database.connect()
    updated = persistence_service.backlog.get(conn, "art1")
    conn.close()

    assert updated.first_seen_run_id == original.first_seen_run_id == "run1"
    assert updated.first_seen_at == original.first_seen_at
    assert updated.last_attempt_run_id == "run2"
    assert updated.status == BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value
    assert updated.attempt_count == 1


def test_a_resumed_backlog_candidate_that_completes_is_closed_out(persistence_service: PersistenceService):
    article = make_article("art1")
    evidence = article_to_evidence(article)
    persistence_service.persist_pipeline_result(
        articles=[article], evidence=[evidence],
        backlog_updates=[BacklogUpdate(
            article_id="art1", agent="ai_product", status=BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value, reason="x",
        )],
        run_id="run1",
    )

    persistence_service.persist_pipeline_result(
        backlog_updates=[BacklogUpdate(
            article_id="art1", agent="ai_product", status=BacklogStatus.COMPLETED.value,
            reason="deep analysis succeeded", research_item_id="ri1", attempt_count=1,
        )],
        run_id="run2",
    )

    pending = persistence_service.get_pending_backlog_candidates(limit=10)
    conn = persistence_service.database.connect()
    final_row = persistence_service.backlog.get(conn, "art1")
    conn.close()

    assert pending == []  # no longer pending -- never reprocessed again
    assert final_row.status == BacklogStatus.COMPLETED.value
    assert final_row.research_item_id == "ri1"


def test_requeue_exhausted_backlog_candidate_service_method(persistence_service: PersistenceService):
    article = make_article("art1")
    persistence_service.persist_pipeline_result(
        articles=[article],
        backlog_updates=[BacklogUpdate(
            article_id="art1", agent="ai_product", status=BacklogStatus.REJECTED_RETRY_EXHAUSTED.value,
            reason="exhausted", attempt_count=5,
        )],
        run_id="run1",
    )

    result = persistence_service.requeue_exhausted_backlog_candidate("art1")

    conn = persistence_service.database.connect()
    row = persistence_service.backlog.get(conn, "art1")
    conn.close()

    assert result == "requeued"
    assert row.status == BacklogStatus.PENDING_RETRY.value
    assert row.attempt_count == 0


def test_requeue_exhausted_backlog_candidate_service_method_rejects_wrong_status(persistence_service: PersistenceService):
    article = make_article("art1")
    persistence_service.persist_pipeline_result(
        articles=[article],
        backlog_updates=[BacklogUpdate(
            article_id="art1", agent="ai_product", status=BacklogStatus.PENDING_TRIAGE_BUDGET.value, reason="x",
        )],
        run_id="run1",
    )

    result = persistence_service.requeue_exhausted_backlog_candidate("art1")

    assert result == "invalid_status"
