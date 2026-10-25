"""Offline validation (no live scan, no Ollama, no retrieval): a single
deterministic replay scenario for one simulated CURRENT run, combining
Tier A (one successfully analysed, digest-qualified event) with Tier B
(one budget-skipped and one timed-out triage-approved article), plus the
negative cases (an explicitly rejected triage candidate and a
never-triaged budget-limited candidate) that must never leak into either
tier. Everything here is built from already-existing model/repository
code with hand-constructed fixtures -- never a real pipeline run."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.config import get_config
from research_assistant.evidence import article_to_evidence
from research_assistant.models.digest import Digest, DigestType
from research_assistant.models.enums import BacklogStatus, DigestPersistenceStatus
from research_assistant.pipeline.runner import PipelineRunResult, RunStatus
from research_assistant.storage.repositories.backlog import BacklogRow
from research_assistant.ui import dashboard_data as dd

from tests.storage.conftest import D, make_article, make_event, make_report, make_research_item

RUN_ID = "replay_run"
CONFIG = get_config()


def _backlog_row(article_id, *, status, triage_relevant=True):
    return BacklogRow(
        article_id=article_id, agent="startup", status=status, reason="replay fixture", attempt_count=1,
        defer_count=1, first_seen_run_id=RUN_ID, first_seen_at=D, last_attempt_run_id=RUN_ID, last_attempt_at=D,
        triage_relevant=triage_relevant, triage_confidence=0.9, research_item_id=None, updated_at=D,
    )


def test_offline_two_tier_replay(persistence_service):
    service = persistence_service

    # --- Tier A: one candidate that completed a qualifying analysis -----
    tier_a_article = make_article("tier_a_art", content_hash="hash-tier-a")
    tier_a_evidence = article_to_evidence(tier_a_article)
    tier_a_item = make_research_item(
        "tier_a_ri", article_id=tier_a_article.id, evidence_ids=[tier_a_evidence.evidence_id], run_id=RUN_ID,
    )
    tier_a_event = make_event(
        "tier_a_ev", research_item_ids=[tier_a_item.id], article_ids=[tier_a_article.id], event_date=D, run_id=RUN_ID,
    )
    tier_a_report = make_report("tier_a_ev")
    service.persist_pipeline_result(
        articles=[tier_a_article], evidence=[tier_a_evidence], research_items=[tier_a_item],
        events=[tier_a_event], reports=[tier_a_report],
    )
    conn = service.database.connect()
    service.backlog.upsert(conn, _backlog_row("tier_a_art", status=BacklogStatus.COMPLETED.value))
    conn.commit()
    conn.close()

    digest = Digest(
        id="replay_digest", digest_type=DigestType.CURRENT, generated_at=D,
        period_start=D - timedelta(hours=24), period_end=D, events=["tier_a_ev"], themes=[],
    )
    service.persist_pipeline_result(digest=digest)
    run = PipelineRunResult(
        run_id=RUN_ID, digest_type=DigestType.CURRENT, started_at=D - timedelta(hours=1), finished_at=D,
        status=RunStatus.PARTIAL_SUCCESS, counts={"research_items_relevant": 1},
        period_start=D - timedelta(hours=24), period_end=D,
        digest_persistence_status=DigestPersistenceStatus.PERSISTED, digest_id="replay_digest",
    )
    service.persist_run(run)

    # --- Tier B: one budget-skipped + one timed-out, both triage-approved
    # (distinct titles -- two genuinely different articles, not the same
    # release syndicated twice, so the Tier B near-duplicate merge must
    # never collapse them)
    conn = service.database.connect()
    service.articles.upsert(
        conn, make_article("tier_b_budget_art", content_hash="hash-tb-budget").model_copy(
            update={"title": "A budget-skipped candidate article"}
        ),
    )
    service.articles.upsert(
        conn, make_article("tier_b_timeout_art", content_hash="hash-tb-timeout").model_copy(
            update={"title": "A timed-out candidate article"}
        ),
    )
    service.backlog.upsert(conn, _backlog_row("tier_b_budget_art", status=BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value))
    service.backlog.upsert(conn, _backlog_row("tier_b_timeout_art", status=BacklogStatus.PENDING_RETRY.value))

    # --- Negative cases: must never appear in either tier --------------
    service.articles.upsert(conn, make_article("rejected_art", content_hash="hash-rejected"))
    service.backlog.upsert(conn, _backlog_row(
        "rejected_art", status=BacklogStatus.REJECTED_TRIAGE_IRRELEVANT.value, triage_relevant=False,
    ))
    service.articles.upsert(conn, make_article("never_triaged_art", content_hash="hash-never-triaged"))
    service.backlog.upsert(conn, _backlog_row(
        "never_triaged_art", status=BacklogStatus.PENDING_TRIAGE_BUDGET.value, triage_relevant=None,
    ))
    conn.commit()
    conn.close()

    # --- Replay the two-tier read path exactly as the UI does ----------
    view = dd.get_latest_current_intelligence(service, CONFIG)
    tier_b_articles, tier_b_omitted = dd.get_current_tier_b_pending(service, CONFIG, RUN_ID)

    # Tier A: exactly the one qualifying, digest-persisted event -- never twice.
    assert view.status == dd.CURRENT_STATUS_OK
    assert [e.event_id for e in view.events] == ["tier_a_ev"]

    # Tier B: exactly the two triage-approved, not-yet-analysed candidates.
    tier_b_ids = sorted(a.article_id for a in tier_b_articles)
    assert tier_b_ids == ["tier_b_budget_art", "tier_b_timeout_art"]
    assert tier_b_omitted == 0
    statuses = {a.article_id: a.status for a in tier_b_articles}
    assert statuses["tier_b_budget_art"] == "pending_deep_analysis_budget"
    assert statuses["tier_b_timeout_art"] == "pending_retry"

    # No overlap between the two tiers.
    assert not set(tier_b_ids) & {view.events[0].event_id}

    # Negative cases excluded from both tiers.
    assert "rejected_art" not in tier_b_ids
    assert "never_triaged_art" not in tier_b_ids

    # --- Newsletter: Tier B selection preserves excerpt + source link --
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}tier_b_budget_art"
    dd.toggle_newsletter_selection(service, selection_id)
    markdown = dd.render_newsletter_markdown(service, dd.get_newsletter_selection(service), CONFIG)
    assert "Clean article text." in markdown
    assert "https://example.com/tier_b_budget_art" in markdown
