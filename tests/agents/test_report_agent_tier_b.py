"""ReportAgent.generate_tier_b / generate_tier_b_many and TierBReportNarrative
-- the on-demand Tier B newsletter report generation path. All scripted
via FakeLLMClient; never touches real Ollama."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from research_assistant.agents.report import ReportAgent, TierBGenerationInput
from research_assistant.agents.schemas import TierBReportNarrative
from research_assistant.llm.errors import LLMResponseError, LLMUnavailableError
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.models.report import TierBReportFailure

from tests.reporting.conftest import FakeLLMClient

D = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)


def _narrative(**overrides) -> TierBReportNarrative:
    kwargs = dict(
        opening_context="NatWest and NVIDIA worked together to improve speech recognition accuracy.",
        key_insights=[
            "The system uses NVIDIA Nemotron automatic speech recognition models.",
            "Testing included 90 colleagues with a range of UK regional accents.",
        ],
        strategic_implications="This may be relevant to institutions evaluating voice AI for customer service.",
        nomura_opportunity="This may be relevant to Nomura's consideration of voice AI applications.",
    )
    kwargs.update(overrides)
    return TierBReportNarrative(**kwargs)


def _item(**overrides) -> TierBGenerationInput:
    kwargs = dict(
        article_id="art1", title="NatWest improves AI speech recognition",
        publisher="NatWest Group", publication_date=D, url="https://natwestgroup.example/a1",
        source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
        text="NatWest and NVIDIA today described work on speech recognition. " * 10,
        content_partial=False,
    )
    kwargs.update(overrides)
    return TierBGenerationInput(**kwargs)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_valid_narrative_parses():
    narrative = _narrative()
    assert len(narrative.key_insights) == 2


def test_key_insights_below_minimum_rejected():
    with pytest.raises(ValidationError):
        _narrative(key_insights=["only one insight"])


def test_key_insights_above_maximum_rejected():
    with pytest.raises(ValidationError):
        _narrative(key_insights=["one", "two", "three", "four", "five"])


def test_empty_key_insight_item_rejected():
    with pytest.raises(ValidationError):
        _narrative(key_insights=["", "a real insight that is long enough"])


def test_overlong_key_insight_item_rejected():
    with pytest.raises(ValidationError):
        _narrative(key_insights=["x" * 300, "a real insight that is long enough"])


def test_overlong_opening_context_rejected():
    with pytest.raises(ValidationError):
        _narrative(opening_context="x" * 2000)


def test_prescriptive_nomura_opportunity_rejected():
    with pytest.raises(ValidationError):
        _narrative(nomura_opportunity="Nomura should immediately invest in this technology.")


# ---------------------------------------------------------------------------
# ReportAgent.generate_tier_b
# ---------------------------------------------------------------------------


def test_generate_tier_b_builds_a_report_from_the_narrative_and_provenance():
    agent = ReportAgent(FakeLLMClient([_narrative()]))

    report = agent.generate_tier_b(_item())

    assert report.article_id == "art1"
    assert report.title == "NatWest improves AI speech recognition"
    assert report.opening_context
    assert len(report.key_insights) == 2
    assert report.source.publisher == "NatWest Group"
    assert report.source.url == "https://natwestgroup.example/a1"
    assert report.content_partial is False


def test_generate_tier_b_sets_content_partial_flag_from_input():
    agent = ReportAgent(FakeLLMClient([_narrative()]))

    report = agent.generate_tier_b(_item(content_partial=True))

    assert report.content_partial is True


def test_generate_tier_b_never_sends_a_fact_list_the_agent_did_not_receive():
    """No ResearchItem/Event facts exist for Tier B -- the prompt must be
    built purely from the TierBGenerationInput's own text/provenance."""
    client = FakeLLMClient([_narrative()])
    agent = ReportAgent(client)

    agent.generate_tier_b(_item())

    _, prompt, schema = client.calls[0]
    assert schema is TierBReportNarrative
    assert "Companies:" not in prompt  # Tier A's deterministic fact block never appears here
    assert "NatWest and NVIDIA today described" in prompt


def test_generate_tier_b_bounds_an_unusually_long_source_text():
    long_text = "word " * 5000  # far beyond the bounded cap
    client = FakeLLMClient([_narrative()])
    agent = ReportAgent(client)

    agent.generate_tier_b(_item(text=long_text))

    _, prompt, _ = client.calls[0]
    assert "[...text truncated for length...]" in prompt


def test_generate_tier_b_drops_unsupported_partnership_language():
    """No ResearchItem exists to validate a partnership claim against --
    every partnership-framed sentence must be dropped, exactly like Tier
    A's own guardrail when there are no validated partners."""
    narrative = _narrative(
        opening_context="NatWest partnered with Acme Corp on this initiative.",
    )
    agent = ReportAgent(FakeLLMClient([narrative]))

    report = agent.generate_tier_b(_item())

    assert "partnered" not in report.opening_context.lower()
    assert "Acme Corp" not in report.opening_context


def test_generate_tier_b_strips_inflated_language():
    narrative = _narrative(
        strategic_implications="NatWest, a key player in UK banking, is exploring voice AI.",
    )
    agent = ReportAgent(FakeLLMClient([narrative]))

    report = agent.generate_tier_b(_item())

    assert "key player" not in report.strategic_implications.lower()


def test_generate_tier_b_raises_llm_error_uncaught():
    agent = ReportAgent(FakeLLMClient([LLMUnavailableError("ollama down")]))

    with pytest.raises(LLMUnavailableError):
        agent.generate_tier_b(_item())


# ---------------------------------------------------------------------------
# ReportAgent.generate_tier_b_many -- sequential, per-item isolation
# ---------------------------------------------------------------------------


def test_generate_tier_b_many_processes_sequentially_and_isolates_failures():
    client = FakeLLMClient([_narrative(), LLMResponseError("bad json"), _narrative()])
    agent = ReportAgent(client)
    items = [_item(article_id="a1"), _item(article_id="a2"), _item(article_id="a3")]

    results = agent.generate_tier_b_many(items)

    assert results[0].article_id == "a1"
    assert not isinstance(results[0], TierBReportFailure)
    assert isinstance(results[1], TierBReportFailure)
    assert results[1].article_id == "a2"
    assert results[1].reason == "LLMResponseError"
    assert not isinstance(results[2], TierBReportFailure)
    assert results[2].article_id == "a3"
    assert len(client.calls) == 3  # every item attempted, strictly in order
