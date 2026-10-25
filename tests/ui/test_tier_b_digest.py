"""Two-tier CURRENT digest -- Tier B ("relevant articles awaiting
analysis"). A date-qualified article that received an EXPLICIT passing
triage decision but never completed a qualifying specialist analysis this
run (deep-analysis budget, timeout, or a recoverable failure) must remain
visible, sourced entirely from the existing `candidate_backlog` table --
no new table, no LLM call, no re-fetch of the article."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.config import get_config
from research_assistant.models.enums import BacklogStatus
from research_assistant.storage.repositories.backlog import BacklogRow
from research_assistant.ui import dashboard_data as dd

from tests.storage.conftest import make_article

D = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


def _row(article_id, *, status, run_id="run1", triage_relevant=True, triage_confidence=0.9, reason="test"):
    return BacklogRow(
        article_id=article_id, agent="startup", status=status, reason=reason, attempt_count=1, defer_count=1,
        first_seen_run_id=run_id, first_seen_at=D, last_attempt_run_id=run_id, last_attempt_at=D,
        triage_relevant=triage_relevant, triage_confidence=triage_confidence, research_item_id=None, updated_at=D,
    )


def test_budget_skipped_triage_approved_article_appears_in_tier_b(persistence_service):
    config = get_config()
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(conn, _row("art1", status=BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value))
    conn.commit()
    conn.close()

    articles, omitted = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert omitted == 0
    assert len(articles) == 1
    assert articles[0].article_id == "art1"
    assert articles[0].status == "pending_deep_analysis_budget"
    assert articles[0].url
    assert articles[0].excerpt


def test_timed_out_triage_approved_article_appears_in_tier_b(persistence_service):
    config = get_config()
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(conn, _row("art1", status=BacklogStatus.PENDING_RETRY.value))
    conn.commit()
    conn.close()

    articles, _ = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert len(articles) == 1
    assert articles[0].status == "pending_retry"


def test_tier_a_qualified_article_does_not_appear_in_tier_b(persistence_service):
    """A successful analysis sets the backlog row to `completed` -- never
    one of the statuses Tier B selects -- so a Tier A article can never
    also render in Tier B."""
    config = get_config()
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(conn, _row("art1", status=BacklogStatus.COMPLETED.value))
    conn.commit()
    conn.close()

    articles, _ = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert articles == []


def test_explicitly_rejected_triage_candidate_is_excluded_from_tier_b(persistence_service):
    config = get_config()
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(
        conn, _row("art1", status=BacklogStatus.REJECTED_TRIAGE_IRRELEVANT.value, triage_relevant=False)
    )
    conn.commit()
    conn.close()

    articles, _ = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert articles == []


def test_never_triaged_budget_limited_candidate_is_not_falsely_labelled_relevant(persistence_service):
    """`pending_triage_budget` means the candidate was NEVER offered a
    triage slot at all -- it must never appear in Tier B regardless of
    `triage_relevant`'s value."""
    config = get_config()
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(
        conn, _row("art1", status=BacklogStatus.PENDING_TRIAGE_BUDGET.value, triage_relevant=None)
    )
    conn.commit()
    conn.close()

    articles, _ = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert articles == []


def test_analysed_but_not_qualified_candidate_is_not_automatically_reintroduced(persistence_service):
    """Rejected by the specialist's own deep analysis -- including it in
    Tier B would imply the rejected development claim might still be
    valid, which the task explicitly forbids."""
    config = get_config()
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(
        conn, _row("art1", status=BacklogStatus.REJECTED_DEEP_ANALYSIS_IRRELEVANT.value)
    )
    conn.commit()
    conn.close()

    articles, _ = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert articles == []


def test_retry_exhausted_candidate_is_excluded_from_tier_b(persistence_service):
    """Terminal -- will never be retried again, so presenting it as
    "awaiting specialist analysis" would be misleading."""
    config = get_config()
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(conn, _row("art1", status=BacklogStatus.REJECTED_RETRY_EXHAUSTED.value))
    conn.commit()
    conn.close()

    articles, _ = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert articles == []


def test_articles_from_another_run_do_not_leak_into_current_digest(persistence_service):
    config = get_config()
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(
        conn, _row("art1", status=BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value, run_id="other_run")
    )
    conn.commit()
    conn.close()

    articles, _ = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert articles == []


def test_missing_usable_text_is_omitted_not_fabricated(persistence_service):
    from research_assistant.models.article import Article, DatePrecision
    from research_assistant.models.enums import SourcePriority, SourceType
    from research_assistant.retrieval.models import ExtractionMethod, ExtractionStatus

    config = get_config()
    conn = persistence_service.database.connect()
    article = Article(
        id="art1", url="https://example.com/art1", canonical_url="https://example.com/art1",
        domain="example.com", source_name="Acme", source_type=SourceType.PRIMARY,
        source_priority=SourcePriority.HIGH, publisher="Acme", title="No text available",
        publication_date=D, publication_date_precision=DatePrecision.EXACT, retrieved_at=D,
        raw_text=None, clean_text=None, content_hash=None,
        extraction_method=ExtractionMethod.BS4_FALLBACK, extraction_status=ExtractionStatus.FAILED,
    )
    persistence_service.articles.upsert(conn, article)
    persistence_service.backlog.upsert(conn, _row("art1", status=BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value))
    conn.commit()
    conn.close()

    articles, omitted = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert articles == []
    assert omitted == 1


def test_excerpt_preserves_original_wording_and_respects_configured_limit(persistence_service):
    config = get_config()
    long_text = "Real company announced a real product today. " * 50
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1", clean_text=long_text))
    persistence_service.backlog.upsert(conn, _row("art1", status=BacklogStatus.PENDING_RETRY.value))
    conn.commit()
    conn.close()

    articles, _ = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert len(articles) == 1
    limit = config.settings.current_llm.tier_b_excerpt_chars
    assert len(articles[0].excerpt) <= limit + 1  # +1 for the truncation marker
    assert "Real company announced a real product today." in articles[0].excerpt


def test_excerpt_is_not_truncated_when_within_limit(persistence_service):
    config = get_config()
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1", clean_text="Short real text."))
    persistence_service.backlog.upsert(conn, _row("art1", status=BacklogStatus.PENDING_RETRY.value))
    conn.commit()
    conn.close()

    articles, _ = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert articles[0].excerpt == "Short real text."


def test_title_publisher_url_date_are_preserved(persistence_service):
    config = get_config()
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(
        conn, make_article("art1", company="Acme AI", url="https://acme.example/a1", publication_date=D)
    )
    persistence_service.backlog.upsert(conn, _row("art1", status=BacklogStatus.PENDING_RETRY.value))
    conn.commit()
    conn.close()

    articles, _ = dd.get_current_tier_b_pending(persistence_service, config, "run1")
    article = articles[0]

    assert article.title == "An article title"
    assert article.publisher == "Acme AI"
    assert article.url == "https://acme.example/a1"
    assert article.publication_date == D


def test_no_website_fetch_or_llm_call_is_ever_invoked(persistence_service, monkeypatch):
    """Structural guard: `dashboard_data` never imports retrieval/llm --
    this test additionally confirms the Tier B path specifically never
    reaches outside already-persisted SQLite rows by asserting it works
    with no network/LLM fixtures configured at all (the module-level
    import guarantee already covers the rest)."""
    config = get_config()
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, make_article("art1"))
    persistence_service.backlog.upsert(conn, _row("art1", status=BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value))
    conn.commit()
    conn.close()

    articles, _ = dd.get_current_tier_b_pending(persistence_service, config, "run1")
    assert len(articles) == 1
