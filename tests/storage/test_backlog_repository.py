"""CURRENT candidate backlog persistence: a candidate deferred by a
budget-exhausted run must be durably retrievable by a later one, never
duplicated, never silently dropped, and bounded storage (cleanup) must
never touch a still-pending row regardless of age."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.models.enums import BacklogStatus
from research_assistant.storage.repositories.backlog import BacklogRow

from tests.storage.conftest import make_article

D = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def _row(article_id="art1", *, status=BacklogStatus.PENDING_TRIAGE_BUDGET.value, agent="ai_product",
         attempt_count=0, defer_count=0, first_seen_at=D, triage_relevant=None, triage_confidence=None) -> BacklogRow:
    return BacklogRow(
        article_id=article_id, agent=agent, status=status, reason="test", attempt_count=attempt_count,
        defer_count=defer_count,
        first_seen_run_id="run1", first_seen_at=first_seen_at, last_attempt_run_id="run1", last_attempt_at=D,
        triage_relevant=triage_relevant, triage_confidence=triage_confidence, research_item_id=None, updated_at=D,
    )


def test_upsert_then_get_round_trips(persistence_service):
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(conn, _row("art1"))
    conn.commit()

    fetched = persistence_service.backlog.get(conn, "art1")
    conn.close()

    assert fetched is not None
    assert fetched.status == BacklogStatus.PENDING_TRIAGE_BUDGET.value
    assert fetched.agent == "ai_product"


def test_upsert_is_idempotent_never_duplicates(persistence_service):
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(conn, _row("art1", attempt_count=0))
    persistence_service.backlog.upsert(conn, _row("art1", attempt_count=1))  # re-discovered/re-deferred
    conn.commit()

    row = conn.execute("SELECT COUNT(*) AS n FROM candidate_backlog WHERE article_id='art1'").fetchone()
    fetched = persistence_service.backlog.get(conn, "art1")
    conn.close()

    assert row["n"] == 1  # one row, not two
    assert fetched.attempt_count == 1  # latest state wins


def test_list_pending_only_returns_pending_statuses(persistence_service):
    conn = persistence_service.database.connect()
    for i, status in enumerate([
        BacklogStatus.PENDING_TRIAGE_BUDGET.value, BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value,
        BacklogStatus.PENDING_RETRY.value, BacklogStatus.COMPLETED.value,
        BacklogStatus.REJECTED_TRIAGE_IRRELEVANT.value, BacklogStatus.REJECTED_DATE_GATE.value,
        BacklogStatus.REJECTED_RETRY_EXHAUSTED.value,
    ]):
        article_id = f"art{i}"
        persistence_service.articles.upsert(conn, make_article(article_id))
        persistence_service.backlog.upsert(conn, _row(article_id, status=status))
    conn.commit()

    pending = persistence_service.backlog.list_pending(conn, limit=100)
    conn.close()

    assert {r.article_id for r in pending} == {"art0", "art1", "art2"}  # only the three PENDING_* rows


def test_list_pending_is_oldest_first_and_bounded_by_limit(persistence_service):
    conn = persistence_service.database.connect()
    for i in range(5):
        article_id = f"art{i}"
        persistence_service.articles.upsert(conn, make_article(article_id))
        persistence_service.backlog.upsert(conn, _row(
            article_id, first_seen_at=D + timedelta(hours=i),  # art0 oldest, art4 newest
        ))
    conn.commit()

    pending = persistence_service.backlog.list_pending(conn, limit=2)
    conn.close()

    assert [r.article_id for r in pending] == ["art0", "art1"]  # oldest two, in order


def test_list_pending_filters_by_agent(persistence_service):
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.articles.upsert(conn, make_article("art2"))
    persistence_service.backlog.upsert(conn, _row("art1", agent="ai_product"))
    persistence_service.backlog.upsert(conn, _row("art2", agent="startup"))
    conn.commit()

    pending = persistence_service.backlog.list_pending(conn, limit=10, agents=["ai_product"])
    conn.close()

    assert [r.article_id for r in pending] == ["art1"]


def test_count_pending_and_oldest_first_seen_at(persistence_service):
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.articles.upsert(conn, make_article("art2"))
    persistence_service.backlog.upsert(conn, _row("art1", first_seen_at=D))
    persistence_service.backlog.upsert(conn, _row("art2", first_seen_at=D + timedelta(hours=3)))
    conn.commit()

    count = persistence_service.backlog.count_pending(conn)
    oldest = persistence_service.backlog.oldest_pending_first_seen_at(conn)
    conn.close()

    assert count == 2
    assert oldest == D


def test_purge_expired_terminal_never_removes_pending_rows(persistence_service):
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("pending1"))
    persistence_service.articles.upsert(conn, make_article("old_completed"))
    old = _row("old_completed", status=BacklogStatus.COMPLETED.value)
    old = old.__class__(**{**old.__dict__, "updated_at": D - timedelta(days=100)})
    persistence_service.backlog.upsert(conn, _row("pending1"))
    persistence_service.backlog.upsert(conn, old)
    conn.commit()

    removed = persistence_service.backlog.purge_expired_terminal(conn, before=D - timedelta(days=14))
    conn.commit()
    remaining_ids = {r["article_id"] for r in conn.execute("SELECT article_id FROM candidate_backlog").fetchall()}
    conn.close()

    assert removed == 1
    assert remaining_ids == {"pending1"}  # the pending row survives regardless of age


def test_purge_expired_terminal_respects_the_cutoff_date(persistence_service):
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("recent_completed"))
    persistence_service.backlog.upsert(conn, _row("recent_completed", status=BacklogStatus.COMPLETED.value))
    conn.commit()

    removed = persistence_service.backlog.purge_expired_terminal(conn, before=D - timedelta(days=14))
    conn.close()

    assert removed == 0  # updated_at (D) is not older than the cutoff


# ---------------------------------------------------------------------------
# Explicit, narrowly-scoped recovery for rejected_retry_exhausted
# ---------------------------------------------------------------------------


def test_requeue_exhausted_moves_it_back_to_pending_retry(persistence_service):
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(conn, _row(
        "art1", status=BacklogStatus.REJECTED_RETRY_EXHAUSTED.value, attempt_count=5,
    ))
    conn.commit()

    result = persistence_service.backlog.requeue_exhausted(conn, "art1", now=D + timedelta(days=1))
    conn.commit()
    row = persistence_service.backlog.get(conn, "art1")
    conn.close()

    assert result == "requeued"
    assert row.status == BacklogStatus.PENDING_RETRY.value
    assert row.attempt_count == 0  # reset by default -- a genuinely fresh shot


def test_requeue_exhausted_can_preserve_attempt_count_instead(persistence_service):
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(conn, _row(
        "art1", status=BacklogStatus.REJECTED_RETRY_EXHAUSTED.value, attempt_count=5,
    ))
    conn.commit()

    persistence_service.backlog.requeue_exhausted(conn, "art1", now=D, reset_attempts=False)
    conn.commit()
    row = persistence_service.backlog.get(conn, "art1")
    conn.close()

    assert row.attempt_count == 5  # explicitly preserved, not reset


def test_requeue_exhausted_preserves_defer_count_either_way(persistence_service):
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(conn, _row(
        "art1", status=BacklogStatus.REJECTED_RETRY_EXHAUSTED.value, attempt_count=5, defer_count=7,
    ))
    conn.commit()

    persistence_service.backlog.requeue_exhausted(conn, "art1", now=D)
    conn.commit()
    row = persistence_service.backlog.get(conn, "art1")
    conn.close()

    assert row.defer_count == 7  # real history, never erased by a recovery action


def test_requeue_exhausted_returns_not_found_for_an_unknown_article(persistence_service):
    conn = persistence_service.database.connect()
    result = persistence_service.backlog.requeue_exhausted(conn, "nope", now=D)
    conn.close()

    assert result == "not_found"


def test_requeue_exhausted_cannot_requeue_a_completed_candidate(persistence_service):
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(conn, _row("art1", status=BacklogStatus.COMPLETED.value))
    conn.commit()

    result = persistence_service.backlog.requeue_exhausted(conn, "art1", now=D)
    row = persistence_service.backlog.get(conn, "art1")
    conn.close()

    assert result == "invalid_status"
    assert row.status == BacklogStatus.COMPLETED.value  # untouched


def test_requeue_exhausted_cannot_requeue_an_irrelevant_or_date_rejected_candidate(persistence_service):
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("irr"))
    persistence_service.articles.upsert(conn, make_article("dated"))
    persistence_service.backlog.upsert(conn, _row("irr", status=BacklogStatus.REJECTED_TRIAGE_IRRELEVANT.value))
    persistence_service.backlog.upsert(conn, _row("dated", status=BacklogStatus.REJECTED_DATE_GATE.value))
    conn.commit()

    result_irr = persistence_service.backlog.requeue_exhausted(conn, "irr", now=D)
    result_dated = persistence_service.backlog.requeue_exhausted(conn, "dated", now=D)
    conn.close()

    assert result_irr == "invalid_status"
    assert result_dated == "invalid_status"


def test_requeue_exhausted_is_idempotent_on_repeated_calls(persistence_service):
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(conn, _row("art1", status=BacklogStatus.REJECTED_RETRY_EXHAUSTED.value))
    conn.commit()

    first = persistence_service.backlog.requeue_exhausted(conn, "art1", now=D)
    conn.commit()
    second = persistence_service.backlog.requeue_exhausted(conn, "art1", now=D)  # already pending_retry now
    conn.close()

    assert first == "requeued"
    assert second == "invalid_status"  # never a second silent mutation, never an exception
