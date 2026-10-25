"""`reporting.tier_a_report_regeneration.regenerate_tier_a_report` -- an
explicit, user-triggered refresh of exactly ONE already-persisted Tier A
(Event) report, reusing the existing `ReportAgent`/link-grounded path.
Never called from selection/deselection or from rendering; a UI concern
mirroring the Tier B "Fetch full article" pattern. No real Ollama."""

from __future__ import annotations

from research_assistant.config import get_config
from research_assistant.llm.errors import LLMResponseError
from research_assistant.models.enums import GroundingType
from research_assistant.models.report import Report, ReportFailure
from research_assistant.reporting.tier_a_report_regeneration import regenerate_tier_a_report
from research_assistant.storage.persistence_service import PersistenceService

from tests.reporting.conftest import FakeLLMClient, make_event, make_narrative, make_research_item
from tests.storage.conftest import make_article


def _seed_event(service: PersistenceService, *, grounding_type=GroundingType.CONTENT):
    item = make_research_item("ri1")
    article = make_article(item.article_id)
    event = make_event("ev1", research_items=[item])
    event = event.model_copy(update={"grounding_type": grounding_type})
    service.persist_pipeline_result(articles=[article], research_items=[item], events=[event])
    return event


def test_regeneration_overwrites_the_existing_report_row(persistence_service: PersistenceService):
    config = get_config()
    _seed_event(persistence_service)

    from research_assistant.agents.report import ReportAgent

    agent = ReportAgent(FakeLLMClient([make_narrative(opening_context="FRESH CONCRETE TEXT")]))
    result = regenerate_tier_a_report(persistence_service, config, "ev1", report_agent=agent)

    assert isinstance(result, Report)
    assert result.opening_context == "FRESH CONCRETE TEXT"

    conn = persistence_service.database.connect()
    persisted = persistence_service.reports.get(conn, "ev1")
    assert persisted is not None
    assert persisted.opening_context == "FRESH CONCRETE TEXT"


def test_regeneration_picks_up_a_fixed_prompt_guardrail_without_any_migration(persistence_service: PersistenceService):
    """The whole point of this feature: an event whose report was
    generated/sanitized under an OLDER guardrail can be refreshed to the
    CURRENT sanitizer/prompt purely by calling this function again --
    no backfill, no schema change, just a fresh LLM call + the existing
    `upsert_report` overwrite by `event_id`."""
    config = get_config()
    _seed_event(persistence_service)

    from research_assistant.agents.report import ReportAgent

    # Simulates an LLM response that the CURRENT (already-fixed) sanitizer
    # would clean up -- proves regeneration re-runs sanitization, not just
    # a raw overwrite.
    agent = ReportAgent(FakeLLMClient([make_narrative(opening_context="NVIDIA, a leading provider of AI, announced X.")]))
    result = regenerate_tier_a_report(persistence_service, config, "ev1", report_agent=agent)

    assert isinstance(result, Report)
    assert ", a of AI" not in result.opening_context
    assert "announced" in result.opening_context.lower() or "X" in result.opening_context


def test_regeneration_of_a_missing_event_returns_none(persistence_service: PersistenceService):
    config = get_config()
    result = regenerate_tier_a_report(persistence_service, config, "no-such-event", report_agent=None)
    assert result is None


def test_regeneration_failure_persists_a_report_failure_and_is_distinguishable(persistence_service: PersistenceService):
    config = get_config()
    _seed_event(persistence_service)

    from research_assistant.agents.report import ReportAgent

    agent = ReportAgent(FakeLLMClient([LLMResponseError("bad schema")]))
    result = regenerate_tier_a_report(persistence_service, config, "ev1", report_agent=agent)

    assert isinstance(result, ReportFailure)
    assert result.event_id == "ev1"

    conn = persistence_service.database.connect()
    persisted = persistence_service.reports.get(conn, "ev1")
    # A failure is persisted via the same upsert_failure path a first-time
    # generation failure already uses -- never silently dropped, and never
    # mistaken for a successful report.
    assert isinstance(persisted, ReportFailure)


def test_link_grounded_event_regenerates_without_any_llm_call(persistence_service: PersistenceService):
    config = get_config()
    _seed_event(persistence_service, grounding_type=GroundingType.LINK)

    from research_assistant.agents.report import ReportAgent

    agent = ReportAgent(FakeLLMClient([]))  # any call would raise AssertionError
    result = regenerate_tier_a_report(persistence_service, config, "ev1", report_agent=agent)

    assert isinstance(result, Report)


def test_regeneration_never_touches_tier_b_tables_or_backlog(persistence_service: PersistenceService):
    """This is purely a Tier A (Event-scoped) operation -- it must never
    create, read, or alter anything in the Tier B backlog/report tables."""
    config = get_config()
    _seed_event(persistence_service)

    from research_assistant.agents.report import ReportAgent

    agent = ReportAgent(FakeLLMClient([make_narrative()]))
    regenerate_tier_a_report(persistence_service, config, "ev1", report_agent=agent)

    conn = persistence_service.database.connect()
    row = conn.execute("SELECT COUNT(*) FROM tier_b_reports").fetchone()
    assert row[0] == 0
