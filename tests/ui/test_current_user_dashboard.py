"""User-facing CURRENT dashboard + newsletter curation (Phase 16) --
dashboard_data.py's view-model layer only (no Streamlit required). Fixtures/
fakes only: no live network, no LLM, no pipeline execution, no scheduler."""

from __future__ import annotations

import time
from datetime import timedelta

from research_assistant.config import get_config
from research_assistant.evidence import article_to_evidence
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import DigestPersistenceStatus, EventType, GroundingType
from research_assistant.pipeline.runner import PipelineRunResult, RunStatus
from research_assistant.storage.persistence_service import PersistenceService
from research_assistant.ui import dashboard_data as dd

from tests.storage.conftest import D, make_article, make_digest, make_event, make_report, make_research_item

CONFIG = get_config()


def _seed_event(
    service: PersistenceService, *, event_id="ev1", relevance=0.85, confidence=0.75,
    event_type=EventType.PARTNERSHIP, url=None, publisher="JPMorgan Chase", grounding_type=GroundingType.CONTENT,
):
    article = make_article(f"art-{event_id}", company=publisher, url=url)
    evidence = article_to_evidence(article)
    item = make_research_item(f"ri-{event_id}", article_id=article.id, company=publisher, event_type=event_type, evidence_ids=[evidence.evidence_id])
    event = make_event(
        event_id, research_item_ids=[item.id], article_ids=[article.id], company=publisher,
        relevance_score=relevance, confidence_score=confidence, event_type=event_type,
    )
    event = event.model_copy(update={"grounding_type": grounding_type})
    if url is not None:
        event = event.model_copy(update={"primary_source": event.primary_source.model_copy(update={"url": url})})
    report = make_report(event_id, relevance_score=relevance, confidence_score=confidence)
    service.persist_pipeline_result(
        articles=[article], evidence=[evidence], research_items=[item], events=[event], reports=[report],
    )
    return event


def _run(
    run_id="cur1", *, status=RunStatus.SUCCESS, digest_persistence_status=DigestPersistenceStatus.PERSISTED,
    digest_id=None, period_start=D - timedelta(hours=24), period_end=D, counts=None, started_at=None, finished_at=None,
):
    return PipelineRunResult(
        run_id=run_id, digest_type=DigestType.CURRENT, started_at=started_at or period_start, finished_at=finished_at or period_end,
        status=status, diagnostics=[], counts=counts or {}, period_start=period_start, period_end=period_end,
        digest_persistence_status=digest_persistence_status, digest_id=digest_id,
    )


# ---------------------------------------------------------------------------
# 1-2: latest CURRENT run selection, strict CURRENT-only isolation
# ---------------------------------------------------------------------------


def test_latest_current_run_is_selected_not_an_older_one(persistence_service: PersistenceService):
    event_old = _seed_event(persistence_service, event_id="ev-old")
    event_new = _seed_event(persistence_service, event_id="ev-new")
    digest_old = make_digest("dg-old", event_ids=["ev-old"], digest_type=DigestType.CURRENT)
    digest_new = make_digest("dg-new", event_ids=["ev-new"], digest_type=DigestType.CURRENT)
    persistence_service.persist_pipeline_result(digest=digest_old)
    persistence_service.persist_pipeline_result(digest=digest_new)
    persistence_service.persist_run(_run("cur-old", digest_id="dg-old", period_start=D - timedelta(days=2), period_end=D - timedelta(days=1)))
    persistence_service.persist_run(_run("cur-new", digest_id="dg-new"))

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.run_id == "cur-new"
    assert [c.event_id for c in view.events] == ["ev-new"]


def test_morning_digest_never_leaks_into_current(persistence_service: PersistenceService):
    _seed_event(persistence_service, event_id="ev1")
    digest = make_digest("dg1", event_ids=["ev1"], digest_type=DigestType.MORNING)
    persistence_service.persist_pipeline_result(digest=digest)
    # No CURRENT run has ever completed.

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.status == dd.CURRENT_STATUS_NO_RUN
    assert view.events == []


def test_evening_digest_never_leaks_into_current(persistence_service: PersistenceService):
    _seed_event(persistence_service, event_id="ev1")
    digest = make_digest("dg1", event_ids=["ev1"], digest_type=DigestType.EVENING)
    persistence_service.persist_pipeline_result(digest=digest)

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.status == dd.CURRENT_STATUS_NO_RUN
    assert view.events == []


def test_historical_inventory_articles_never_leak_into_current(persistence_service: PersistenceService):
    """`get_latest_current_intelligence` never queries articles directly by
    date -- only via the latest CURRENT run's own digest -- so an
    unrelated, undigested article sitting in the historical inventory can
    never surface here."""
    make_article("historical-art")  # just constructing it, never persisted into any CURRENT digest
    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)
    assert view.status == dd.CURRENT_STATUS_NO_RUN


def test_older_current_runs_are_never_merged_into_the_latest(persistence_service: PersistenceService):
    _seed_event(persistence_service, event_id="ev-old")
    _seed_event(persistence_service, event_id="ev-new")
    persistence_service.persist_pipeline_result(digest=make_digest("dg-old", event_ids=["ev-old"], digest_type=DigestType.CURRENT))
    persistence_service.persist_pipeline_result(digest=make_digest("dg-new", event_ids=["ev-new"], digest_type=DigestType.CURRENT))
    persistence_service.persist_run(_run("cur-old", digest_id="dg-old", period_start=D - timedelta(days=2), period_end=D - timedelta(days=1)))
    persistence_service.persist_run(_run("cur-new", digest_id="dg-new"))

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert len(view.events) == 1
    assert view.events[0].event_id == "ev-new"


# ---------------------------------------------------------------------------
# 7-9: exact period display, empty state, failed state (no silent fallback)
# ---------------------------------------------------------------------------


def test_exact_persisted_period_is_returned(persistence_service: PersistenceService):
    _seed_event(persistence_service, event_id="ev1")
    persistence_service.persist_pipeline_result(digest=make_digest("dg1", event_ids=["ev1"], digest_type=DigestType.CURRENT))
    period_start = D - timedelta(hours=24)
    persistence_service.persist_run(_run("cur1", digest_id="dg1", period_start=period_start, period_end=D))

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.period_start == period_start
    assert view.period_end == D


def test_successful_empty_current_result_renders_as_empty_not_failed(persistence_service: PersistenceService):
    persistence_service.persist_run(_run("cur1", digest_persistence_status=DigestPersistenceStatus.NO_QUALIFYING_EVENTS, digest_id=None))

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.status == dd.CURRENT_STATUS_EMPTY
    assert view.events == []


def test_failed_current_run_falls_back_to_the_older_successful_snapshot(persistence_service: PersistenceService):
    """Phase 17 (snapshot-fallback UX): a failed newest CURRENT attempt
    must not blank the user page -- the previous usable snapshot keeps
    being displayed, with its OWN run_id/events, never merged or
    reattributed to the new (failed) run."""
    _seed_event(persistence_service, event_id="ev-old")
    persistence_service.persist_pipeline_result(digest=make_digest("dg-old", event_ids=["ev-old"], digest_type=DigestType.CURRENT))
    persistence_service.persist_run(_run("cur-old", digest_id="dg-old", period_start=D - timedelta(days=2), period_end=D - timedelta(days=1)))
    persistence_service.persist_run(_run("cur-new", status=RunStatus.FAILED, digest_id=None, digest_persistence_status=DigestPersistenceStatus.NOT_ATTEMPTED))

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.status == dd.CURRENT_STATUS_OK
    assert view.run_id == "cur-old"
    assert [c.event_id for c in view.events] == ["ev-old"]
    assert view.latest_attempt_outcome == "failed"
    assert view.is_newer_scan_running is False


def test_persistence_failed_run_falls_back_to_the_older_successful_snapshot(persistence_service: PersistenceService):
    """Reproduces the real forensic finding: a run can be status=SUCCESS at
    the pipeline level yet have digest_persistence_status=PERSISTENCE_FAILED
    (the event_articles UNIQUE-constraint bug). The user page must still
    show the previous usable CURRENT snapshot (never merging the two
    runs), flagged via `latest_attempt_outcome`."""
    _seed_event(persistence_service, event_id="ev-old")
    persistence_service.persist_pipeline_result(digest=make_digest("dg-old", event_ids=["ev-old"], digest_type=DigestType.CURRENT))
    persistence_service.persist_run(_run("cur-old", digest_id="dg-old", period_start=D - timedelta(days=2), period_end=D - timedelta(days=1)))
    persistence_service.persist_run(
        _run("cur-new", status=RunStatus.PARTIAL_SUCCESS, digest_id=None, digest_persistence_status=DigestPersistenceStatus.PERSISTENCE_FAILED)
    )

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.status == dd.CURRENT_STATUS_OK
    assert view.run_id == "cur-old"
    assert [c.event_id for c in view.events] == ["ev-old"]
    assert view.latest_attempt_outcome == "failed"


def test_failed_current_run_with_no_older_snapshot_at_all_shows_empty_state(persistence_service: PersistenceService):
    """When there genuinely is no usable snapshot anywhere (not even an
    older one), the view must still be FAILED/empty -- fallback only
    applies when an older usable snapshot actually exists."""
    persistence_service.persist_run(_run("cur-new", status=RunStatus.FAILED, digest_id=None, digest_persistence_status=DigestPersistenceStatus.NOT_ATTEMPTED))

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.status == dd.CURRENT_STATUS_FAILED
    assert view.events == []


# ---------------------------------------------------------------------------
# 10-11: individual article URL preferred; general/source URL fallback only
# ---------------------------------------------------------------------------


def test_individual_article_url_is_preferred_when_available(persistence_service: PersistenceService):
    _seed_event(persistence_service, event_id="ev1", url="https://www.jpmorganchase.com/newsroom/ai-partnership-story", publisher="JPMorgan Chase")
    persistence_service.persist_pipeline_result(digest=make_digest("dg1", event_ids=["ev1"], digest_type=DigestType.CURRENT))
    persistence_service.persist_run(_run("cur1", digest_id="dg1"))

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    card = view.events[0]
    assert card.source_url == "https://www.jpmorganchase.com/newsroom/ai-partnership-story"
    assert card.source_link_label == "Read Original"


def test_general_source_url_used_only_as_fallback_when_no_individual_url(persistence_service: PersistenceService):
    event = _seed_event(persistence_service, event_id="ev1", publisher="JPMorgan Chase")
    # Simulate the (in-practice-rare) case where the primary source URL is empty.
    event = event.model_copy(update={"primary_source": event.primary_source.model_copy(update={"url": ""})})
    conn = persistence_service.database.connect()
    persistence_service.events.upsert(conn, event)
    conn.commit()
    conn.close()
    persistence_service.persist_pipeline_result(digest=make_digest("dg1", event_ids=["ev1"], digest_type=DigestType.CURRENT))
    persistence_service.persist_run(_run("cur1", digest_id="dg1"))

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    card = view.events[0]
    configured_names = {s.name for s in CONFIG.sources.peers}
    if "JPMorgan Chase" in configured_names:
        assert card.source_link_label == "Open Source"
        assert card.source_url != ""
    else:
        assert card.source_url is None


# ---------------------------------------------------------------------------
# 12: LINK-grounded events remain truthful and clickable
# ---------------------------------------------------------------------------


def test_link_grounded_event_is_flagged_and_still_clickable(persistence_service: PersistenceService):
    _seed_event(
        persistence_service, event_id="ev1", grounding_type=GroundingType.LINK,
        url="https://www.ft.com/content/undated-article",
    )
    persistence_service.persist_pipeline_result(digest=make_digest("dg1", event_ids=["ev1"], digest_type=DigestType.CURRENT))
    persistence_service.persist_run(_run("cur1", digest_id="dg1"))

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    card = view.events[0]
    assert card.is_link_only is True
    assert card.source_url == "https://www.ft.com/content/undated-article"
    assert view.link_only_count == 1


# ---------------------------------------------------------------------------
# 13-16: newsletter selection persistence, no duplicates, removal, exact ids
# ---------------------------------------------------------------------------


def test_add_to_newsletter_persists_across_separate_dashboard_data_calls(persistence_service: PersistenceService):
    _seed_event(persistence_service, event_id="ev1")

    dd.toggle_newsletter_selection(persistence_service, "ev1", now=D)

    # Simulates a fresh Streamlit rerun: nothing but the DB-backed read remains.
    assert dd.get_newsletter_selection(persistence_service) == ["ev1"]


def test_duplicate_newsletter_selections_cannot_occur(persistence_service: PersistenceService):
    _seed_event(persistence_service, event_id="ev1")

    conn = persistence_service.database.connect()
    persistence_service.newsletter.add(conn, "ev1", selected_at=D)
    persistence_service.newsletter.add(conn, "ev1", selected_at=D)
    conn.close()

    assert dd.get_newsletter_selection(persistence_service) == ["ev1"]


def test_toggling_a_selected_event_removes_it(persistence_service: PersistenceService):
    _seed_event(persistence_service, event_id="ev1")

    first = dd.toggle_newsletter_selection(persistence_service, "ev1", now=D)
    second = dd.toggle_newsletter_selection(persistence_service, "ev1", now=D)

    assert first is True
    assert second is False
    assert dd.get_newsletter_selection(persistence_service) == []


def test_generate_newsletter_receives_only_the_selected_event_ids(persistence_service: PersistenceService):
    _seed_event(persistence_service, event_id="ev1")
    _seed_event(persistence_service, event_id="ev2")
    dd.toggle_newsletter_selection(persistence_service, "ev1", now=D)
    # ev2 is deliberately never selected.

    selected = dd.get_newsletter_selection(persistence_service)
    markdown = dd.render_newsletter_markdown(persistence_service, selected)

    assert "ev1" not in markdown  # the title is rendered, not the raw id -- just confirm scope
    assert selected == ["ev1"]
    assert markdown.count("###") == 1  # exactly one event section


def test_newsletter_markdown_renders_key_insights_when_present(persistence_service: PersistenceService):
    """2026-10-15 quality fix: Tier A reports gained a bounded
    `key_insights` list -- confirms it renders as its own bulleted
    section, distinct from `opening_context`/`strategic_implications`."""
    from tests.storage.conftest import make_report

    event = _seed_event(persistence_service, event_id="ev1")
    report = make_report("ev1").model_copy(
        update={"details": make_report("ev1").details.model_copy(
            update={"key_insights": ["First concrete takeaway.", "Second concrete takeaway."]},
        )},
    )
    conn = persistence_service.database.connect()
    persistence_service.reports.upsert_report(conn, report)
    conn.commit()
    conn.close()

    dd.toggle_newsletter_selection(persistence_service, "ev1", now=D)
    markdown = dd.render_newsletter_markdown(persistence_service, dd.get_newsletter_selection(persistence_service))

    assert "**Key insights**" in markdown
    assert "- First concrete takeaway." in markdown
    assert "- Second concrete takeaway." in markdown


def test_newsletter_markdown_omits_key_insights_section_for_an_old_report_without_it(persistence_service: PersistenceService):
    """Backward compatibility: a Report persisted before `key_insights`
    existed deserializes with an empty list (the model's default) and
    must render exactly as before -- no fabricated section, no crash."""
    _seed_event(persistence_service, event_id="ev1")  # make_report() defaults key_insights to []

    dd.toggle_newsletter_selection(persistence_service, "ev1", now=D)
    markdown = dd.render_newsletter_markdown(persistence_service, dd.get_newsletter_selection(persistence_service))

    assert "**Key insights**" not in markdown


def test_newsletter_markdown_de_emphasizes_and_bounds_a_long_entity_list(persistence_service: PersistenceService):
    """User feedback (2026-10-15): a long flat entity dump added little
    value -- it must now be bounded and clearly secondary ("Also
    mentioned"), never removed outright (the underlying facts are real)."""
    from research_assistant.models.report import ReportDetails
    from tests.storage.conftest import make_report

    event = _seed_event(persistence_service, event_id="ev1")
    many_companies = [f"Company{i}" for i in range(10)]
    report = make_report("ev1").model_copy(
        update={"details": ReportDetails(companies=many_companies, narrative="Details narrative.")},
    )
    conn = persistence_service.database.connect()
    persistence_service.reports.upsert_report(conn, report)
    conn.commit()
    conn.close()

    dd.toggle_newsletter_selection(persistence_service, "ev1", now=D)
    markdown = dd.render_newsletter_markdown(persistence_service, dd.get_newsletter_selection(persistence_service))

    assert "Entities / participants" not in markdown
    assert "**Also mentioned**" in markdown
    assert "(+4 more)" in markdown  # 10 companies, 6 shown


def test_newsletter_selection_does_not_mutate_the_event(persistence_service: PersistenceService):
    event_before = _seed_event(persistence_service, event_id="ev1")

    dd.toggle_newsletter_selection(persistence_service, "ev1", now=D)

    conn = persistence_service.database.connect()
    event_after = persistence_service.events.get(conn, "ev1")
    conn.close()
    assert event_after.canonical_title == event_before.canonical_title
    assert event_after.relevance_score == event_before.relevance_score


# ---------------------------------------------------------------------------
# 17: no user UI action triggers a CURRENT scan
# ---------------------------------------------------------------------------


def test_reading_current_intelligence_with_no_run_returns_instantly_never_blocking_on_network(persistence_service: PersistenceService):
    started = time.monotonic()
    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)
    elapsed = time.monotonic() - started

    assert view.status == dd.CURRENT_STATUS_NO_RUN
    assert elapsed < 1.0  # a real scan/network/LLM call would take far longer than this


def test_dashboard_data_module_still_never_imports_retrieval_llm_or_scan_machinery():
    import research_assistant.ui.dashboard_data as dashboard_data_module
    from pathlib import Path

    lines = Path(dashboard_data_module.__file__).read_text(encoding="utf-8").splitlines()
    import_lines = [line.strip() for line in lines if line.strip().startswith(("import ", "from "))]
    assert not any("research_assistant.retrieval" in line for line in import_lines)
    assert not any("research_assistant.llm" in line for line in import_lines)
    assert not any("pipeline.graph" in line or "pipeline.nodes" in line for line in import_lines)
    assert not any("scheduling.jobs" in line for line in import_lines)
