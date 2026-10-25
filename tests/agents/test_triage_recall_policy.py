"""Relevance-policy fix (forensic investigation of run `8c605b0881e8edf0`):
triage's own prompt was biased toward REJECTING a candidate whenever its
text was "too thin or ambiguous" -- the opposite of a recall-oriented first
pass, with no "ambiguous" escape valve (`TriageItemResult.relevant` is
strictly binary). Separately, `StartupAgent`'s closed "Relevant event
types" list didn't name "research/survey findings" as a qualifying event,
even though its own "Scope" section lists AI security as in-scope -- a
JumpCloud-style "we released a study finding X" release doesn't map onto
any of funding/acquisition/partnership/product-launch/model-release/
enterprise-deal/strategic-initiative.

These are prompt-wording fixes only; no schema, guardrail, or deterministic
logic changed. This file is a structural regression guard against ever
silently reverting either fix -- actual model compliance is a model-
behavior question these tests cannot and do not claim to prove.
"""

from __future__ import annotations

from research_assistant.agents.prompt_loader import load_prompt


def test_triage_prompt_is_recall_oriented_not_precision_oriented():
    prompt = load_prompt("triage_system.md")
    lowered = prompt.lower()
    assert "recall" in lowered
    assert "be conservative" not in lowered
    assert "too thin or ambiguous to\n   support a confident relevant judgment, mark it not relevant" not in prompt


def test_triage_prompt_tells_the_model_to_defer_the_stronger_judgment():
    prompt = load_prompt("triage_system.md")
    assert "deep analysis" in prompt.lower()
    assert "let deep analysis" in prompt.lower() or "stronger" in prompt.lower()


def test_startup_prompt_names_research_findings_as_a_qualifying_event_type():
    prompt = load_prompt("startup_system.md")
    assert "research, survey, or report findings" in prompt.lower()
