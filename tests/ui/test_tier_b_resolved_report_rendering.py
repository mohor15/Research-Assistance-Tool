"""Newsletter Markdown/email rendering of an ALREADY-RESOLVED Tier B
report -- `render_newsletter_markdown`/`render_event_list_email`
themselves never call a model or the network; they only ever read a
pre-resolved `TierBReportResolution` dict the caller passes in. Mixed
Tier A + Tier B selections, and every Tier B outcome (generated/cached/
insufficient/failed/none-resolved), are covered here."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.config import get_config
from research_assistant.models.enums import BacklogStatus, SourcePriority, SourceType
from research_assistant.models.report import ReportSource, TierBReport, TierBReportFailure
from research_assistant.reporting.tier_b_newsletter_resolution import TierBReportResolution
from research_assistant.storage.repositories.backlog import BacklogRow
from research_assistant.ui import dashboard_data as dd

from tests.storage.conftest import make_article, make_event, make_report

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


def _seed_tier_a_event(service, event_id="ev1"):
    from tests.storage.conftest import make_research_item

    service.persist_pipeline_result(
        articles=[make_article(f"art-{event_id}")],
        research_items=[make_research_item(f"ri-{event_id}", article_id=f"art-{event_id}")],
        events=[make_event(event_id, research_item_ids=[f"ri-{event_id}"], article_ids=[f"art-{event_id}"])],
        reports=[make_report(event_id)],
    )


def _report(article_id="art1", **overrides) -> TierBReport:
    kwargs = dict(
        article_id=article_id, title="Acme launches new AI feature",
        opening_context="Acme AI described a new feature in a press release.",
        key_insights=["The feature targets enterprise customers.", "No pricing was disclosed."],
        strategic_implications="This may be relevant to firms evaluating enterprise AI tooling.",
        nomura_opportunity="This may be relevant to Nomura's consideration of enterprise AI tooling.",
        source=ReportSource(
            publisher="Acme AI", title="Acme launches new AI feature", publication_date=D,
            url=f"https://acme.example/{article_id}", source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
        ),
        content_partial=False,
    )
    kwargs.update(overrides)
    return TierBReport(**kwargs)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_generated_tier_b_report_renders_full_sections_in_markdown(persistence_service):
    _seed_tier_b_article(persistence_service)
    config = get_config()
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"
    resolutions = {"art1": TierBReportResolution(status="generated", report=_report())}

    markdown = dd.render_newsletter_markdown(persistence_service, [selection_id], config, tier_b_reports=resolutions)

    assert "on-demand analytical report generated" in markdown.lower()
    assert "The feature targets enterprise customers." in markdown
    assert "No pricing was disclosed." in markdown
    assert "This may be relevant to firms evaluating enterprise AI tooling." in markdown
    assert "This may be relevant to Nomura's consideration" in markdown
    # Excerpt-only fallback text must NOT also appear once a real report rendered
    assert "awaiting specialist analysis" not in markdown.lower()


def test_cached_tier_b_report_renders_the_same_as_a_freshly_generated_one(persistence_service):
    _seed_tier_b_article(persistence_service)
    config = get_config()
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"
    resolutions = {"art1": TierBReportResolution(status="cached", report=_report(opening_context="Cached opening."))}

    markdown = dd.render_newsletter_markdown(persistence_service, [selection_id], config, tier_b_reports=resolutions)

    assert "Cached opening." in markdown


def test_insufficient_content_resolution_renders_excerpt_with_explicit_note(persistence_service):
    _seed_tier_b_article(persistence_service)
    config = get_config()
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"
    resolutions = {"art1": TierBReportResolution(status="insufficient_content")}

    markdown = dd.render_newsletter_markdown(persistence_service, [selection_id], config, tier_b_reports=resolutions)

    assert "could not be generated" in markdown.lower()
    assert "insufficient" in markdown.lower()
    assert "Clean article text." in markdown  # the excerpt fallback is still shown


def test_failed_generation_resolution_renders_excerpt_with_explicit_note(persistence_service):
    _seed_tier_b_article(persistence_service)
    config = get_config()
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"
    resolutions = {
        "art1": TierBReportResolution(
            status="failed", failure=TierBReportFailure(article_id="art1", reason="LLMUnavailableError", detail="down"),
        ),
    }

    markdown = dd.render_newsletter_markdown(persistence_service, [selection_id], config, tier_b_reports=resolutions)

    assert "could not be generated" in markdown.lower()
    assert "Clean article text." in markdown


def test_no_resolution_supplied_falls_back_to_the_original_excerpt_only_behavior(persistence_service):
    """Zero behavior change for a caller that never resolved anything --
    exactly the pre-existing `test_tier_b_newsletter.py` behavior."""
    _seed_tier_b_article(persistence_service)
    config = get_config()
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"

    markdown = dd.render_newsletter_markdown(persistence_service, [selection_id], config)

    assert "awaiting specialist analysis" in markdown.lower()
    assert "Clean article text." in markdown


def test_content_partial_flag_renders_a_limitation_note(persistence_service):
    _seed_tier_b_article(persistence_service)
    config = get_config()
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"
    resolutions = {"art1": TierBReportResolution(status="generated", report=_report(content_partial=True))}

    markdown = dd.render_newsletter_markdown(persistence_service, [selection_id], config, tier_b_reports=resolutions)

    assert "may be incomplete" in markdown.lower()


def test_mixed_tier_a_and_resolved_tier_b_selection_renders_both(persistence_service):
    _seed_tier_a_event(persistence_service, "ev1")
    _seed_tier_b_article(persistence_service, "art1")
    config = get_config()
    selections = ["ev1", f"{dd.TIER_B_SELECTION_PREFIX}art1"]
    resolutions = {"art1": TierBReportResolution(status="generated", report=_report())}

    markdown = dd.render_newsletter_markdown(persistence_service, selections, config, tier_b_reports=resolutions)

    assert "An event title" in markdown  # Tier A
    assert "The feature targets enterprise customers." in markdown  # Tier B generated report
    assert "on-demand analytical report generated" in markdown.lower()


def test_rendering_never_mutates_backlog_or_event_state(persistence_service):
    """Rendering a resolved Tier B report must never promote the article's
    backlog status or create a fake Event -- resolution/persistence
    already happened upstream; rendering is pure."""
    _seed_tier_b_article(persistence_service)
    config = get_config()
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"
    resolutions = {"art1": TierBReportResolution(status="generated", report=_report())}

    dd.render_newsletter_markdown(persistence_service, [selection_id], config, tier_b_reports=resolutions)

    conn = persistence_service.database.connect()
    try:
        row = conn.execute("SELECT status FROM candidate_backlog WHERE article_id = 'art1'").fetchone()
        event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    finally:
        conn.close()
    assert row["status"] == BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value
    assert event_count == 0


# ---------------------------------------------------------------------------
# Email rendering
# ---------------------------------------------------------------------------


def test_email_rendering_includes_generated_tier_b_report_sections(persistence_service):
    from research_assistant.delivery.renderer import render_event_list_email

    _seed_tier_b_article(persistence_service)
    config = get_config()
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"
    resolutions = {"art1": TierBReportResolution(status="generated", report=_report())}

    content = render_event_list_email(
        persistence_service, [selection_id], subject="Test", config=config, tier_b_reports=resolutions,
    )

    assert "The feature targets enterprise customers." in content.html_body
    assert "No pricing was disclosed." in content.html_body
    assert "This may be relevant to firms evaluating enterprise AI tooling." in content.text_body


def test_email_rendering_without_tier_b_reports_param_keeps_old_excerpt_behavior(persistence_service):
    from research_assistant.delivery.renderer import render_event_list_email

    _seed_tier_b_article(persistence_service)
    config = get_config()
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"

    content = render_event_list_email(persistence_service, [selection_id], subject="Test", config=config)

    assert "awaiting specialist analysis" in content.html_body.lower()
    assert "Clean article text." in content.text_body


def test_email_rendering_is_consistent_with_markdown_for_the_same_resolution(persistence_service):
    from research_assistant.delivery.renderer import render_event_list_email

    _seed_tier_b_article(persistence_service)
    config = get_config()
    selection_id = f"{dd.TIER_B_SELECTION_PREFIX}art1"
    resolutions = {"art1": TierBReportResolution(status="generated", report=_report())}

    markdown = dd.render_newsletter_markdown(persistence_service, [selection_id], config, tier_b_reports=resolutions)
    content = render_event_list_email(
        persistence_service, [selection_id], subject="Test", config=config, tier_b_reports=resolutions,
    )

    for fragment in ("The feature targets enterprise customers.", "No pricing was disclosed."):
        assert fragment in markdown
        assert fragment in content.text_body
