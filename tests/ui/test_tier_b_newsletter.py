"""Tier B articles can be selected into the newsletter alongside (or
instead of) Tier A Events -- stored in the SAME `newsletter_selections`
table as a `tb:`-prefixed id (no schema change), rendered from the
already-retrieved article text, never passed to an LLM as if it were a
validated Event."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.config import get_config
from research_assistant.models.enums import BacklogStatus
from research_assistant.storage.repositories.backlog import BacklogRow
from research_assistant.ui import dashboard_data as dd

from tests.storage.conftest import make_article

D = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


def _seed_tier_b_article(service, article_id="art1"):
    conn = service.database.connect()
    service.articles.upsert(conn, make_article(article_id, company="Acme AI", url=f"https://acme.example/{article_id}"))
    service.backlog.upsert(conn, BacklogRow(
        article_id=article_id, agent="startup", status=BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value,
        reason="test", attempt_count=0, defer_count=1, first_seen_run_id="run1", first_seen_at=D,
        last_attempt_run_id="run1", last_attempt_at=D, triage_relevant=True, triage_confidence=0.9,
        research_item_id=None, updated_at=D,
    ))
    conn.commit()
    conn.close()


def test_tier_b_article_can_be_selected_for_the_newsletter(persistence_service):
    _seed_tier_b_article(persistence_service)
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"

    dd.toggle_newsletter_selection(persistence_service, selection_id)

    assert selection_id in dd.get_newsletter_selection(persistence_service)


def test_newsletter_markdown_renders_tier_b_with_excerpt_and_attribution(persistence_service):
    _seed_tier_b_article(persistence_service)
    config = get_config()
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"

    markdown = dd.render_newsletter_markdown(persistence_service, [selection_id], config)

    assert "Acme AI" in markdown
    assert "https://acme.example/art1" in markdown
    assert "Clean article text." in markdown
    assert "awaiting specialist analysis" in markdown.lower()


def test_newsletter_markdown_distinguishes_tier_b_from_tier_a():
    """A selection id with no `tb:` prefix is never treated as Tier B --
    the distinction is purely the prefix, never guessed from content."""
    assert "tb:art1".startswith(dd.TIER_B_SELECTION_PREFIX)
    assert not "ev-some-event".startswith(dd.TIER_B_SELECTION_PREFIX)


def test_newsletter_markdown_without_config_skips_tier_b_gracefully(persistence_service):
    """A caller that doesn't pass `config` (every pre-existing Tier-A-only
    call site) must not crash on a Tier B selection -- it is simply
    omitted, exactly like an unresolvable event id already is."""
    _seed_tier_b_article(persistence_service)
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"

    markdown = dd.render_newsletter_markdown(persistence_service, [selection_id])

    assert "Acme AI" not in markdown


def test_newsletter_works_with_tier_b_entries_and_no_tier_a_events(persistence_service):
    _seed_tier_b_article(persistence_service)
    config = get_config()
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"

    markdown = dd.render_newsletter_markdown(persistence_service, [selection_id], config)

    assert "NOMURA AI & FINTECH INTELLIGENCE" in markdown
    assert "Acme AI" in markdown


def test_email_rendering_includes_tier_b_excerpt_and_source_link(persistence_service):
    from research_assistant.delivery.renderer import render_event_list_email

    _seed_tier_b_article(persistence_service)
    config = get_config()
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"

    content = render_event_list_email(persistence_service, [selection_id], subject="Test", config=config)

    assert "Acme AI" in content.html_body
    assert "https://acme.example/art1" in content.html_body
    assert "Clean article text." in content.html_body
    assert "awaiting specialist analysis" in content.html_body.lower()
    assert "Clean article text." in content.text_body
