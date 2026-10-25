"""2026-10-15 quality fix: both ReportAgent prompts (Tier A and Tier B)
push for concrete development details and reasoned implications rather
than defaulting to a generic hedge, while explicitly preserving the
existing grounding guardrail's conservatism. Deterministic prompt-content
checks only -- actual model compliance needs live Ollama and is validated
separately, as with the existing grounding-prompt tests."""

from __future__ import annotations

from research_assistant.agents.prompt_loader import load_prompt

_TIER_A_PROMPT_FILE = "report_system.md"
_TIER_B_PROMPT_FILE = "report_tier_b_system.md"


def _collapsed(text: str) -> str:
    return " ".join(text.lower().split())


def test_tier_a_prompt_instructs_concrete_opening_context():
    text = _collapsed(load_prompt(_TIER_A_PROMPT_FILE))
    assert "be concrete" in text


def test_tier_a_prompt_has_a_key_insights_section_distinct_from_opening_context():
    text = _collapsed(load_prompt(_TIER_A_PROMPT_FILE))
    assert "key_insights" in text
    assert "not a restatement of `opening_context`" in text or "not a restatement of opening_context" in text


def test_tier_a_prompt_discourages_defaulting_to_a_generic_hedge():
    text = _collapsed(load_prompt(_TIER_A_PROMPT_FILE))
    assert "reason about the development" in text
    assert "rather than defaulting to a generic hedge" in text


def test_tier_a_prompt_still_forbids_inventing_facts_and_unqualified_outcomes():
    """The concreteness push must not have weakened the existing
    grounding guardrail instructions."""
    text = _collapsed(load_prompt(_TIER_A_PROMPT_FILE))
    assert "never invent a fact of any kind" in text
    assert "do not invent a future outcome or state one as a certainty" in text
    assert "qualified interpretation" in text


def test_tier_a_prompt_key_insights_rule_reuses_the_existing_guardrails():
    text = _collapsed(load_prompt(_TIER_A_PROMPT_FILE))
    assert "key_insights` items are factual narration" in text
    assert "never repeat `opening_context` verbatim as a key insight" in text or \
        "never repeat opening_context verbatim as a key insight" in text


def test_tier_b_prompt_discourages_defaulting_to_a_generic_hedge():
    text = _collapsed(load_prompt(_TIER_B_PROMPT_FILE))
    assert "reason about the development" in text
    assert "do not retreat to a vague hedge" in text


def test_tier_b_prompt_still_requires_qualified_language_for_implications():
    text = _collapsed(load_prompt(_TIER_B_PROMPT_FILE))
    assert "may suggest" in text and "could indicate" in text
    assert "never a declarative claim" in text
