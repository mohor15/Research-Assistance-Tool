"""ReportAgent: the 12 required scenarios, all via a scripted FakeLLMClient."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from research_assistant.agents.report import ReportAgent
from research_assistant.llm.errors import LLMResponseError, LLMUnavailableError
from research_assistant.models.enums import SourcePriority, SourceType

from tests.reporting.conftest import FakeLLMClient, make_event, make_narrative, make_research_item

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def test_normal_event_report_generates_correctly():
    item = make_research_item("ri1", publication_date=D, products=["AI copilot"])
    event = make_event("ev1", research_items=[item], event_date=D)
    agent = ReportAgent(FakeLLMClient([make_narrative()]))

    report = agent.generate(event, [item])

    assert report.event_id == "ev1"
    assert report.title == "JPMorgan launches AI copilot"
    assert report.opening_context
    assert report.details.narrative
    assert report.relevance_score == event.relevance_score
    assert report.confidence_score == event.confidence_score


def test_missing_optional_information_is_left_null_not_guessed():
    item = make_research_item(
        "ri1", publication_date=D, people=[], products=[], technologies=[], partnerships=[],
        geography=[], deal_value=None,
    )
    event = make_event("ev1", research_items=[item], event_date=D)
    agent = ReportAgent(FakeLLMClient([make_narrative()]))

    report = agent.generate(event, [item])

    assert report.details.people == []
    assert report.details.products == []
    assert report.details.technologies == []
    assert report.details.partnerships == []
    assert report.details.geography == []
    assert report.details.deal_value is None


def test_multiple_source_event_aggregates_facts_from_all_items():
    primary = make_research_item("ri1", publication_date=D, products=["AI copilot"], source_type=SourceType.PRIMARY)
    secondary = make_research_item(
        "ri2", company="Reuters", publisher="Reuters", publication_date=D, source_type=SourceType.SECONDARY,
        entities=["JPMorgan Chase"], people=["Jane Doe"],
    )
    event = make_event("ev1", research_items=[primary, secondary], event_date=D)
    agent = ReportAgent(FakeLLMClient([make_narrative()]))

    report = agent.generate(event, [primary, secondary])

    assert "JPMorgan Chase" in report.details.companies
    assert "Reuters" in report.details.companies
    assert report.details.people == ["Jane Doe"]
    assert len(report.sources) == 2


def test_primary_and_secondary_sources_remain_distinguishable():
    primary = make_research_item("ri1", publication_date=D, source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH)
    secondary = make_research_item(
        "ri2", company="Reuters", publisher="Reuters", publication_date=D,
        source_type=SourceType.SECONDARY, source_priority=SourcePriority.HIGH,
    )
    event = make_event("ev1", research_items=[primary, secondary], event_date=D)
    agent = ReportAgent(FakeLLMClient([make_narrative()]))

    report = agent.generate(event, [primary, secondary])

    source_types = {source.source_type for source in report.sources}
    assert SourceType.PRIMARY in source_types
    assert SourceType.SECONDARY in source_types


def test_no_deal_value_is_fabricated_when_none_disclosed():
    item = make_research_item("ri1", publication_date=D, deal_value=None)
    event = make_event("ev1", research_items=[item], event_date=D)
    agent = ReportAgent(FakeLLMClient([make_narrative()]))

    report = agent.generate(event, [item])

    assert report.details.deal_value is None


def test_deal_value_is_preserved_verbatim_when_disclosed():
    item = make_research_item("ri1", publication_date=D, deal_value="$50 million")
    event = make_event("ev1", research_items=[item], event_date=D)
    agent = ReportAgent(FakeLLMClient([make_narrative()]))

    report = agent.generate(event, [item])

    assert report.details.deal_value == "$50 million"


def test_no_people_are_fabricated_when_none_extracted():
    item = make_research_item("ri1", publication_date=D, people=[])
    event = make_event("ev1", research_items=[item], event_date=D)
    agent = ReportAgent(FakeLLMClient([make_narrative()]))

    report = agent.generate(event, [item])

    assert report.details.people == []


def test_no_technology_is_fabricated_when_none_extracted():
    item = make_research_item("ri1", publication_date=D, technologies=[])
    event = make_event("ev1", research_items=[item], event_date=D)
    agent = ReportAgent(FakeLLMClient([make_narrative()]))

    report = agent.generate(event, [item])

    assert report.details.technologies == []


def test_provenance_preserves_original_urls_exactly():
    item = make_research_item("ri1", publication_date=D, url="https://www.jpmorganchase.com/newsroom/ai-copilot?utm_source=x")
    event = make_event("ev1", research_items=[item], event_date=D)
    agent = ReportAgent(FakeLLMClient([make_narrative()]))

    report = agent.generate(event, [item])

    assert report.sources[0].url == "https://www.jpmorganchase.com/newsroom/ai-copilot?utm_source=x"
    assert report.event_id == event.id


def test_nomura_opportunity_uses_diplomatic_non_prescriptive_wording():
    item = make_research_item("ri1", publication_date=D)
    event = make_event("ev1", research_items=[item], event_date=D)
    agent = ReportAgent(FakeLLMClient([make_narrative()]))

    report = agent.generate(event, [item])

    lowered = report.nomura_opportunity.lower()
    assert "nomura should" not in lowered
    assert "nomura must" not in lowered
    assert "may be relevant" in lowered or "may warrant" in lowered or "could be relevant" in lowered


def test_prescriptive_nomura_wording_from_llm_is_rejected():
    item = make_research_item("ri1", publication_date=D)
    event = make_event("ev1", research_items=[item], event_date=D)
    bad_narrative_kwargs = dict(nomura_opportunity="Nomura should acquire this company immediately.")
    with pytest.raises(ValidationError):
        make_narrative(**bad_narrative_kwargs)


def test_malformed_llm_output_raises_llm_response_error():
    item = make_research_item("ri1", publication_date=D)
    event = make_event("ev1", research_items=[item], event_date=D)
    agent = ReportAgent(FakeLLMClient([LLMResponseError("model returned malformed JSON")]))

    with pytest.raises(LLMResponseError):
        agent.generate(event, [item])


def test_ollama_failure_propagates_cleanly():
    item = make_research_item("ri1", publication_date=D)
    event = make_event("ev1", research_items=[item], event_date=D)
    agent = ReportAgent(FakeLLMClient([LLMUnavailableError("Ollama not running")]))

    with pytest.raises(LLMUnavailableError):
        agent.generate(event, [item])


def test_empty_event_with_no_research_items_does_not_crash():
    item = make_research_item("ri1", publication_date=D)
    event = make_event("ev1", research_items=[item], event_date=D)
    agent = ReportAgent(FakeLLMClient([make_narrative()]))

    report = agent.generate(event, [])  # no ResearchItems supplied at all

    assert report.details.companies == []
    # The Event's own source references still produce a source entry (the
    # URL/publisher/source_type came from the Event, not the missing item) —
    # just without a title/publication_date, since no item lookup succeeded.
    assert len(report.sources) == 1
    assert report.sources[0].title is None


def test_generate_many_continues_after_one_event_fails():
    item_ok = make_research_item("ri1", publication_date=D)
    item_fail = make_research_item("ri2", publication_date=D)
    event_ok = make_event("ev1", research_items=[item_ok], event_date=D)
    event_fail = make_event("ev2", research_items=[item_fail], event_date=D)

    agent = ReportAgent(FakeLLMClient([LLMUnavailableError("down"), make_narrative()]))

    results = agent.generate_many([(event_fail, [item_fail]), (event_ok, [item_ok])])

    from research_assistant.models.report import Report, ReportFailure

    assert isinstance(results[0], ReportFailure)
    assert results[0].event_id == "ev2"
    assert isinstance(results[1], Report)
    assert results[1].event_id == "ev1"


def test_report_agent_never_imports_retrieval_or_makes_http_calls():
    from pathlib import Path

    import research_assistant.agents.report as report_module

    lines = Path(report_module.__file__).read_text(encoding="utf-8").splitlines()
    import_lines = [line.strip() for line in lines if line.strip().startswith(("import ", "from "))]
    assert not any("research_assistant.retrieval" in line for line in import_lines)
    assert not any(line == "import requests" for line in import_lines)


def test_aggregated_facts_are_sorted_deterministically_regardless_of_item_order():
    item_a = make_research_item("z1", publication_date=D, people=["Zed"])
    item_b = make_research_item("a1", publication_date=D, people=["Abe"])
    event = make_event("ev1", research_items=[item_a, item_b], event_date=D)
    agent = ReportAgent(FakeLLMClient([make_narrative(), make_narrative()]))

    first = agent.generate(event, [item_a, item_b])
    second = agent.generate(event, [item_b, item_a])

    assert first.details.people == second.details.people == ["Abe", "Zed"]
