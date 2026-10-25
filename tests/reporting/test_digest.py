"""DigestService: morning/evening digests, theme generation, empty sets,
source preservation, deterministic metadata. No LLM call anywhere."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from research_assistant.events.clock import FixedClock
from research_assistant.models.digest import DigestType
from research_assistant.reporting.digest_service import DigestService

from tests.reporting.conftest import make_event, make_research_item

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def test_morning_digest_is_built_correctly(config):
    item = make_research_item("ri1", publication_date=D)
    event = make_event("ev1", research_items=[item], event_date=D)
    service = DigestService(config, clock=FixedClock(D))

    digest = service.build_digest(
        [event], digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D
    )

    assert digest.digest_type == DigestType.MORNING
    assert digest.events == ["ev1"]
    assert digest.generated_at == D


def test_evening_digest_is_built_correctly(config):
    item = make_research_item("ri1", publication_date=D)
    event = make_event("ev1", research_items=[item], event_date=D)
    service = DigestService(config, clock=FixedClock(D))

    digest = service.build_digest(
        [event], digest_type=DigestType.EVENING, period_start=D - timedelta(hours=12), period_end=D
    )

    assert digest.digest_type == DigestType.EVENING


def test_multiple_events_are_all_included_in_order():
    from research_assistant.config import get_config

    config = get_config()
    item1 = make_research_item("ri1", publication_date=D)
    item2 = make_research_item("ri2", company="OpenAI", publisher="OpenAI", publication_date=D)
    event1 = make_event("ev1", research_items=[item1], event_date=D, relevance_score=0.9)
    event2 = make_event("ev2", research_items=[item2], event_date=D, primary_company="OpenAI", relevance_score=0.5)
    service = DigestService(config, clock=FixedClock(D))

    digest = service.build_digest(
        [event1, event2], digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D
    )

    assert digest.events == ["ev1", "ev2"]  # preserves the given (already-ranked) order


def test_theme_generation_matches_configured_strategic_themes(config):
    item = make_research_item(
        "ri1", publication_date=D, title="JPMorgan expands enterprise AI infrastructure",
        summary="JPMorgan is investing in AI infrastructure and enterprise AI across its technology stack.",
    )
    event = make_event(
        "ev1", research_items=[item], event_date=D,
        canonical_title="JPMorgan expands enterprise AI infrastructure",
        description="JPMorgan is investing in AI infrastructure and enterprise AI across its technology stack.",
    )
    service = DigestService(config, clock=FixedClock(D))

    digest = service.build_digest([event], digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    labels = {theme.label for theme in digest.themes}
    assert "enterprise AI" in labels
    assert "AI infrastructure" in labels
    for theme in digest.themes:
        assert theme.event_ids == ["ev1"]


def test_themes_never_include_events_unrelated_to_the_theme(config):
    item = make_research_item("ri1", publication_date=D, title="Unrelated announcement", summary="A routine corporate announcement with no AI theme keywords present.")
    event = make_event("ev1", research_items=[item], event_date=D, canonical_title="Unrelated announcement", description="A routine corporate announcement with no AI theme keywords present.")
    service = DigestService(config, clock=FixedClock(D))

    digest = service.build_digest([event], digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert digest.themes == []  # no configured strategic theme keyword appears -> no themes invented


def test_empty_event_set_produces_a_valid_empty_digest(config):
    service = DigestService(config, clock=FixedClock(D))

    digest = service.build_digest([], digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert digest.events == []
    assert digest.themes == []


def test_source_urls_are_preserved_through_digest_construction(config):
    item = make_research_item("ri1", publication_date=D, url="https://www.jpmorganchase.com/newsroom/ai?utm_source=x")
    event = make_event("ev1", research_items=[item], event_date=D)
    service = DigestService(config, clock=FixedClock(D))

    digest = service.build_digest([event], digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    # Digest references events by id; the Event object itself (unmodified,
    # not embedded/flattened) still carries the exact original source URL.
    assert digest.events == [event.id]
    assert event.primary_source.url == "https://www.jpmorganchase.com/newsroom/ai?utm_source=x"


def test_digest_metadata_is_deterministic_for_the_same_input(config):
    item = make_research_item("ri1", publication_date=D)
    event = make_event("ev1", research_items=[item], event_date=D)
    service = DigestService(config, clock=FixedClock(D))

    first = service.build_digest([event], digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)
    second = service.build_digest([event], digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert first.id == second.id


def test_digest_id_changes_when_event_set_changes(config):
    item1 = make_research_item("ri1", publication_date=D)
    item2 = make_research_item("ri2", company="OpenAI", publisher="OpenAI", publication_date=D)
    event1 = make_event("ev1", research_items=[item1], event_date=D)
    event2 = make_event("ev2", research_items=[item2], event_date=D, primary_company="OpenAI")
    service = DigestService(config, clock=FixedClock(D))

    only_one = service.build_digest([event1], digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)
    both = service.build_digest([event1, event2], digest_type=DigestType.MORNING, period_start=D - timedelta(hours=12), period_end=D)

    assert only_one.id != both.id


def test_digest_service_never_imports_the_llm_layer():
    import research_assistant.reporting.digest_service as digest_service_module

    lines = Path(digest_service_module.__file__).read_text(encoding="utf-8").splitlines()
    import_lines = [line.strip() for line in lines if line.strip().startswith(("import ", "from "))]
    assert not any("research_assistant.llm" in line for line in import_lines)
