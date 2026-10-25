"""End-to-end EventService: cluster -> score -> rank, config-driven weights,
no LLM/retrieval coupling, deterministic across repeated runs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import research_assistant.events.service as service_module
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.models.enums import EventType, SourceType

from tests.events.conftest import make_item

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def test_build_events_clusters_scores_and_ranks_end_to_end(config):
    same_event_a = make_item("a1", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                              event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["AI assistant"],
                              source_type=SourceType.PRIMARY)
    same_event_b = make_item("a2", company="Reuters", title="Reuters reports JPMorgan AI assistant launch",
                              event_type=EventType.PRODUCT_LAUNCH, publication_date=D,
                              entities=["JPMorgan Chase", "AI assistant"], source_type=SourceType.SECONDARY)
    separate_event = make_item("a3", company="OpenAI", title="OpenAI releases a new model",
                                event_type=EventType.MODEL_RELEASE, publication_date=D - timedelta(days=40))

    service = EventService(config, clock=FixedClock(D))
    events = service.build_events([same_event_a, same_event_b, separate_event])

    assert len(events) == 2
    jpmorgan_event = next(e for e in events if e.primary_company == "JPMorgan Chase")
    assert set(jpmorgan_event.research_items) == {"a1", "a2"}
    assert jpmorgan_event.primary_source.research_item_id == "a1"
    assert len(jpmorgan_event.secondary_sources) == 1
    # More recent + more corroborated -> ranked ahead of the older single-source event.
    assert events[0].primary_company == "JPMorgan Chase"


def test_event_service_filters_agentic_ai_out_of_related_companies_end_to_end(config):
    """Integration-level confirmation of the entity-typing fix: EventService
    threads the real configured topic vocabulary through to the builder, so
    a concept the LLM mislabels as a partner (e.g. "Agentic AI") never
    surfaces as a related company via the real, non-test code path."""
    item = make_item(
        "a1", company="NVIDIA", title="NVIDIA and CoreWeave close the loop on agentic AI",
        publication_date=D, partnerships=["NVIDIA", "CoreWeave", "Agentic AI", "Ennoble Care"],
    )
    service = EventService(config, clock=FixedClock(D))

    events = service.build_events([item])

    assert len(events) == 1
    assert "Agentic AI" not in events[0].related_companies
    assert "CoreWeave" in events[0].related_companies
    assert "Ennoble Care" in events[0].related_companies


def test_build_events_returns_empty_list_for_no_input(config):
    service = EventService(config, clock=FixedClock(D))
    assert service.build_events([]) == []


def test_weights_come_from_config_not_hardcoded(config):
    item = make_item("a1", company="X", title="t", publication_date=D)
    service_default = EventService(config, clock=FixedClock(D))
    events_default = service_default.build_events([item])

    modified_config = config.model_copy(deep=True)
    modified_config.settings.ranking.relevance = modified_config.settings.ranking.relevance.model_copy(
        update={"ai_relevance": 1.0, "financial_services_relevance": 0.0, "strategic_significance": 0.0, "recency": 0.0, "novelty": 0.0}
    )
    service_modified = EventService(modified_config, clock=FixedClock(D))
    events_modified = service_modified.build_events([item])

    assert events_default[0].relevance_score != events_modified[0].relevance_score
    assert events_modified[0].relevance_score == events_modified[0].scoring_breakdown.relevance.ai_relevance


def test_deterministic_output_across_repeated_runs(config):
    items = [
        make_item("a1", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["AI assistant"]),
        make_item("a2", company="JPMorgan Chase", title="JPMorgan introduces new AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["AI assistant"]),
        make_item("a3", company="OpenAI", title="OpenAI releases Model A",
                  event_type=EventType.MODEL_RELEASE, publication_date=D, products=["Model A"]),
    ]
    service = EventService(config, clock=FixedClock(D))

    first_run = service.build_events(list(items))
    second_run = service.build_events(list(reversed(items)))

    assert [e.id for e in first_run] == [e.id for e in second_run]
    assert [e.relevance_score for e in first_run] == [e.relevance_score for e in second_run]
    assert [e.confidence_score for e in first_run] == [e.confidence_score for e in second_run]


def test_events_module_never_imports_the_llm_or_retrieval_layers():
    for module in (service_module,):
        lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
        import_lines = [line.strip() for line in lines if line.strip().startswith(("import ", "from "))]
        assert not any("research_assistant.llm" in line for line in import_lines)
        assert not any("research_assistant.retrieval" in line for line in import_lines)


def test_no_llm_call_is_used_anywhere_in_the_events_package():
    """Structural guarantee: the events package has no dependency on the LLM
    client at all, so it is architecturally impossible for it to make an
    LLM call — scoring is Python arithmetic over already-produced fields."""
    import research_assistant.events as events_package

    package_dir = Path(events_package.__file__).parent
    for py_file in package_dir.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        import_lines = [line.strip() for line in source.splitlines() if line.strip().startswith(("import ", "from "))]
        assert not any("research_assistant.llm" in line for line in import_lines), py_file
