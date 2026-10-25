"""Phase 13B grounding regression tests, deterministic half (no live Ollama):

- every extraction agent's system prompt explicitly forbids treating
  "available through X" as a partnership with X, and explains the
  dated-development vs. evergreen distinction with worked examples (tests
  1-3's prompt-level guard -- the model's actual compliance is verified
  separately, with live Ollama, in the Phase 13B E2E validation);
- the ReportAgent's own deterministic fact-aggregation (`_aggregate_facts`/
  `_build_prompt`) faithfully reflects an empty `partnerships` list as
  "none stated", so a report generated from correctly-extracted facts has
  nothing in its prompt suggesting a partnership exists (tests 12-13's
  deterministic half: the Python aggregation layer, which is what this
  project's architecture actually controls, is provably not the source of
  a fabricated partnership).
"""

from __future__ import annotations

from research_assistant.agents.ai_product import AIProductAgent
from research_assistant.agents.analyst import AnalystAgent
from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.report import _aggregate_facts, _build_prompt
from research_assistant.agents.startup import StartupAgent
from research_assistant.events.clock import FixedClock
from research_assistant.events.models import ClusterMetadata, Event

from tests.agents.conftest import FakeLLMClient
from tests.events.conftest import make_item
from tests.storage.conftest import D, make_event


def _collapsed_prompt(agent) -> str:
    return " ".join(agent._system_prompt.lower().split())


def test_ai_product_prompt_forbids_availability_through_as_partnership(config):
    text = _collapsed_prompt(AIProductAgent(FakeLLMClient([]), config))
    assert "available through" in text
    assert "is not a partnership" in text or "not a partnership by itself" in text
    assert "aws" in text and "google cloud" in text and "microsoft foundry" in text


def test_peer_prompt_forbids_availability_through_as_partnership(config):
    text = _collapsed_prompt(PeerAgent(FakeLLMClient([]), config))
    assert "available through" in text
    assert "not a partnership" in text


def test_startup_prompt_forbids_availability_through_as_partnership(config):
    text = _collapsed_prompt(StartupAgent(FakeLLMClient([]), config))
    assert "available through" in text or "built on another company" in text
    assert "not a partnership" in text


def test_analyst_prompt_forbids_mention_as_partnership(config):
    text = _collapsed_prompt(AnalystAgent(FakeLLMClient([]), config))
    assert "not a partnership" in text


def test_every_extraction_agent_prompt_explains_dated_vs_evergreen(config):
    for agent in (
        AIProductAgent(FakeLLMClient([]), config),
        PeerAgent(FakeLLMClient([]), config),
        AnalystAgent(FakeLLMClient([]), config),
        StartupAgent(FakeLLMClient([]), config),
    ):
        text = _collapsed_prompt(agent)
        assert "is_dated_development" in text
        assert "evergreen" in text


def test_every_extraction_agent_prompt_explains_entity_typing(config):
    for agent in (
        AIProductAgent(FakeLLMClient([]), config),
        PeerAgent(FakeLLMClient([]), config),
        AnalystAgent(FakeLLMClient([]), config),
        StartupAgent(FakeLLMClient([]), config),
    ):
        text = _collapsed_prompt(agent)
        assert '"company"' in text or "`company`" in text
        assert "`product`" in text or '"product"' in text
        assert "`technology`" in text or '"technology"' in text


def test_report_system_prompt_distinguishes_fact_from_qualified_implication():
    from research_assistant.agents.report import ReportAgent

    agent = ReportAgent(FakeLLMClient([]))
    text = " ".join(agent._system_prompt.lower().split())
    assert "qualified interpretation" in text or "framed as interpretation" in text
    assert "will transform investment analysis" in text  # the task's own "bad" example, verbatim
    assert "none stated" in text


def test_report_system_prompt_forbids_date_fabrication_when_unknown():
    from research_assistant.agents.report import ReportAgent

    agent = ReportAgent(FakeLLMClient([]))
    text = " ".join(agent._system_prompt.lower().split())
    assert "unknown" in text
    assert "specific date" in text


# ---------------------------------------------------------------------------
# Deterministic facts-aggregation (the Python layer ReportAgent actually
# controls, as opposed to the LLM's own compliance)
# ---------------------------------------------------------------------------


def test_empty_partnerships_are_aggregated_as_none_stated_never_invented():
    item = make_item(
        "a1", company="Bank of America", title="t", partnerships=[],
        entities=[],
    )
    facts = _aggregate_facts([item])

    assert facts["partnerships"] == []

    event = make_event("ev1", research_item_ids=["a1"], article_ids=[item.article_id])
    prompt = _build_prompt(event, facts)

    assert "Partnerships: none stated" in prompt
    assert "AWS" not in prompt
    assert "Google Cloud" not in prompt


def test_unknown_event_date_is_passed_through_to_the_report_prompt_as_unknown():
    item = make_item("a1", company="Anthropic", title="t", is_dated_development=False)
    facts = _aggregate_facts([item])
    event = make_event("ev1", research_item_ids=["a1"], article_ids=[item.article_id], event_date=None)

    prompt = _build_prompt(event, facts)

    assert "Event date: unknown" in prompt
