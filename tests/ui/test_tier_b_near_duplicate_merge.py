"""Tier B near-duplicate merge: confirmed real-production defect (CURRENT
run `b2d8c1466a0ade52`, 48h-window live scan) -- PR Newswire syndicates the
IDENTICAL press release under two distinct numeric release URLs (two
genuinely different URLs/content hashes, so article-level dedup correctly
never merges them, and Tier A's event clustering never sees a Tier B
candidate at all since it never reaches event formation). Both copies
independently passed triage and sat in `candidate_backlog`, so the SAME
story appeared twice in `get_current_tier_b_pending`'s output. Fixtures/
`persistence_service` only -- no live network, no Ollama."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.config import get_config
from research_assistant.models.enums import BacklogStatus
from research_assistant.storage.repositories.backlog import BacklogRow
from research_assistant.ui import dashboard_data as dd

from tests.storage.conftest import make_article

D = datetime(2026, 10, 14, 6, 15, tzinfo=timezone.utc)


def _seed_backlog_row(service, *, article, triage_confidence):
    conn = service.database.connect()
    service.articles.upsert(conn, article)
    service.backlog.upsert(conn, BacklogRow(
        article_id=article.id, agent="startup", status=BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value,
        reason="test", attempt_count=0, defer_count=1, first_seen_run_id="run1", first_seen_at=D,
        last_attempt_run_id="run1", last_attempt_at=D, triage_relevant=True, triage_confidence=triage_confidence,
        research_item_id=None, updated_at=D,
    ))
    conn.commit()
    conn.close()


_IFS_TITLE = "IFS Softeon launches Fulfillment Flow to bring context-aware AI to warehouse operations"


def _ifs_article(article_id, *, url_suffix, title=_IFS_TITLE, publication_date=D):
    article = make_article(
        article_id, url=f"https://www.prnewswire.com/news-releases/ifs-softeon-{url_suffix}.html",
        publication_date=publication_date,
    )
    return article.model_copy(update={"title": title})


def test_two_syndicated_copies_of_the_same_release_collapse_to_one(persistence_service):
    config = get_config()
    _seed_backlog_row(persistence_service, article=_ifs_article("ifs1", url_suffix="302907151"), triage_confidence=0.95)
    _seed_backlog_row(persistence_service, article=_ifs_article("ifs2", url_suffix="302907154"), triage_confidence=0.90)

    articles, omitted = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert len(articles) == 1
    assert omitted == 0


def test_the_higher_confidence_copy_is_the_one_kept(persistence_service):
    config = get_config()
    _seed_backlog_row(persistence_service, article=_ifs_article("ifs1", url_suffix="302907151"), triage_confidence=0.60)
    _seed_backlog_row(persistence_service, article=_ifs_article("ifs2", url_suffix="302907154"), triage_confidence=0.95)

    articles, _omitted = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert len(articles) == 1
    assert articles[0].triage_confidence == 0.95


def test_same_title_different_day_is_not_merged(persistence_service):
    """A verbatim title match alone is not enough -- it must also be the
    same day (mirrors the Tier A override's own date-gap bound)."""
    config = get_config()
    _seed_backlog_row(persistence_service, article=_ifs_article("ifs1", url_suffix="302907151"), triage_confidence=0.9)
    _seed_backlog_row(
        persistence_service,
        article=_ifs_article("ifs2", url_suffix="302907154", publication_date=D - timedelta(days=3)),
        triage_confidence=0.9,
    )

    articles, _omitted = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert len(articles) == 2


def test_different_titles_same_day_are_not_merged(persistence_service):
    """Two genuinely different stories published the same day must never
    be collapsed -- only a near-verbatim title match triggers the merge."""
    config = get_config()
    _seed_backlog_row(persistence_service, article=_ifs_article("ifs1", url_suffix="302907151"), triage_confidence=0.9)
    _seed_backlog_row(
        persistence_service,
        article=_ifs_article("other1", url_suffix="302900000", title="A completely unrelated AI funding announcement"),
        triage_confidence=0.9,
    )

    articles, _omitted = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert len(articles) == 2


def test_three_syndicated_copies_still_collapse_to_one(persistence_service):
    config = get_config()
    for i, conf in enumerate([0.7, 0.95, 0.8]):
        _seed_backlog_row(
            persistence_service, article=_ifs_article(f"ifs{i}", url_suffix=f"30290715{i}"), triage_confidence=conf,
        )

    articles, _omitted = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert len(articles) == 1
    assert articles[0].triage_confidence == 0.95
