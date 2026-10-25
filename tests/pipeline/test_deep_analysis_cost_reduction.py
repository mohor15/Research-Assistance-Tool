"""CPU-only qwen2.5:3b deep-analysis cost-reduction fix (forensic benchmark
follow-up): compact `AgentAssessment` output, a domain-fair cut to a smaller
configurable `max_deep_analysis` budget, and the `127.0.0.1` Ollama default.
No live Ollama/network anywhere -- FakeLLMClient/FakeTriageLLMClient only;
the real-Qwen benchmark lives outside the test suite (see the final report).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from research_assistant.agents.base import _derive_nomura_relevance
from research_assistant.agents.schemas import AgentAssessment, AnalystAssessment
from research_assistant.config import get_config
from research_assistant.llm.client import create_ollama_client
from research_assistant.models.enums import AgentName
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
)
from research_assistant.agents.triage import TriageItemResult
from research_assistant.pipeline.nodes import _select_deep_analysis_winners

from tests.pipeline.test_current_llm_funnel import _triage_candidate


def _valid_assessment_kwargs(**overrides):
    kwargs = dict(
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="Concrete GenAI product launch."),
        category=TopicCategory.ENTERPRISE_AI,
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="Directly applies to FS."),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="Signals adoption trend."),
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# A. Compact assessment
# ---------------------------------------------------------------------------


def test_compact_assessment_parses_without_nomura_relevance():
    """The model is no longer asked for `nomura_relevance` -- a valid
    compact response (everything else) still parses cleanly."""
    assessment = AgentAssessment(**_valid_assessment_kwargs())
    assert assessment.relevance.label == RelevanceLabel.RELEVANT
    assert not hasattr(assessment, "nomura_relevance")


def test_nomura_relevance_not_in_the_schema_sent_to_ollama():
    """The actual JSON Schema handed to Ollama's `format` field (the real
    generation-cost driver) must not request a third free-text rationale
    paragraph nothing downstream reads."""
    schema = AgentAssessment.model_json_schema()
    assert "nomura_relevance" not in schema["properties"]
    assert "financial_services_relevance" in schema["properties"]
    assert "strategic_significance" in schema["properties"]

    analyst_schema = AnalystAssessment.model_json_schema()
    assert "nomura_relevance" not in analyst_schema["properties"]


def test_required_downstream_fields_remain_on_the_compact_schema():
    """Every field `events/scoring.py`, `events/builder.py`,
    `events/similarity.py`, and the UI actually read must stay required/
    present -- the compaction only removed a field with zero downstream
    consumers, never a field something depends on."""
    schema = AgentAssessment.model_json_schema()
    required = set(schema["required"])
    for field in (
        "relevance", "category", "is_dated_development",
        "financial_services_relevance", "strategic_significance",
    ):
        assert field in required, f"{field} must stay required -- downstream code reads it"


def test_derived_nomura_relevance_mirrors_strategic_significance():
    """`ResearchItem.nomura_relevance` and the `research_items` table's
    NOT NULL columns still get a value -- deterministically derived in
    Python (zero extra LLM cost), not requested from the model."""
    strategic = RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="Major strategic move.")
    derived = _derive_nomura_relevance(strategic)
    assert derived.level == RelevanceLevel.HIGH
    assert derived.rationale  # non-empty -- satisfies the NOT NULL DB column


def test_invalid_structured_output_still_fails_safely():
    """A response missing a still-required field (e.g. the model omitting
    `financial_services_relevance`) must raise, exactly as before -- the
    compaction never loosened validation for genuinely required fields."""
    kwargs = _valid_assessment_kwargs()
    del kwargs["financial_services_relevance"]
    with pytest.raises(ValidationError):
        AgentAssessment(**kwargs)


def test_extra_unrequested_fields_still_do_not_attach_new_data():
    raw = _valid_assessment_kwargs()
    raw = {
        "relevance": raw["relevance"].model_dump(),
        "category": raw["category"].value,
        "is_dated_development": raw["is_dated_development"],
        "financial_services_relevance": raw["financial_services_relevance"].model_dump(),
        "strategic_significance": raw["strategic_significance"].model_dump(),
        "nomura_relevance": {"level": "medium", "rationale": "a model might still emit this; must be ignored"},
    }
    assessment = AgentAssessment.model_validate(raw)
    assert not hasattr(assessment, "nomura_relevance")
    assert "nomura_relevance" not in assessment.model_dump()


# ---------------------------------------------------------------------------
# B. Winner selection: domain-fair cut to the (smaller) deep-analysis budget
# ---------------------------------------------------------------------------


def _winner(candidate_id: str, agent: AgentName):
    candidate = _triage_candidate(candidate_id)
    candidate = replace(candidate, primary_agent=agent)
    return candidate, None


def test_configured_maximum_is_respected():
    winners = [_winner(f"c{i}", AgentName.PEER) for i in range(10)]
    selected, dropped = _select_deep_analysis_winners(winners, budget=4)
    assert len(selected) == 4
    assert dropped == 6


def test_fewer_winners_than_budget_are_all_selected_unchanged():
    winners = [_winner(f"c{i}", AgentName.PEER) for i in range(3)]
    selected, dropped = _select_deep_analysis_winners(winners, budget=4)
    assert selected == winners
    assert dropped == 0


def test_domain_aware_coverage_is_preserved_under_a_small_budget():
    """5 PEER + 5 AI_PRODUCT winners, budget=4: a plain global-rank cutoff
    could let one domain consume the entire budget -- domain-fair
    round-robin selection must instead give each domain a share."""
    peer_winners = [_winner(f"p{i}", AgentName.PEER) for i in range(5)]
    ai_winners = [_winner(f"a{i}", AgentName.AI_PRODUCT) for i in range(5)]
    selected, dropped = _select_deep_analysis_winners(peer_winners + ai_winners, budget=4)

    assert len(selected) == 4
    assert dropped == 6
    selected_agents = {pair[0].primary_agent for pair in selected}
    assert selected_agents == {AgentName.PEER, AgentName.AI_PRODUCT}, (
        "both domains must be represented in the bounded deep-analysis selection"
    )


def test_selection_is_deterministic():
    winners = [_winner(f"c{i}", AgentName.STARTUP) for i in range(9)]
    selected1, _ = _select_deep_analysis_winners(winners, budget=4)
    selected2, _ = _select_deep_analysis_winners(winners, budget=4)
    assert [c.candidate_id for c, _ in selected1] == [c.candidate_id for c, _ in selected2]


def test_dropped_winners_each_emit_a_budget_skip_diagnostic():
    """Instrumentation task (forensic finding, CURRENT run
    `e818e08adc2e4dd1`): the domain-fair-cut-dropped triage winners must
    each get their own `budget_skip:deep_analysis_selection` diagnostic --
    optional, purely additive, never changing which candidates are
    selected."""
    winners = [_winner(f"c{i}", AgentName.PEER) for i in range(10)]
    diagnostics = []
    selected, dropped = _select_deep_analysis_winners(winners, budget=4, diagnostics=diagnostics)

    assert len(selected) == 4
    assert dropped == 6
    skip_messages = [d.message for d in diagnostics if d.message.startswith("budget_skip:deep_analysis_selection")]
    assert len(skip_messages) == 6
    selected_ids = {c.candidate_id for c, _ in selected}
    for candidate, _result in winners:
        has_skip = any(f"candidate={candidate.candidate_id} " in m for m in skip_messages)
        assert has_skip != (candidate.candidate_id in selected_ids)


def test_diagnostics_omitted_by_default_never_crashes_and_selection_unaffected():
    winners = [_winner(f"c{i}", AgentName.PEER) for i in range(10)]
    selected, dropped = _select_deep_analysis_winners(winners, budget=4)
    assert len(selected) == 4
    assert dropped == 6


def test_no_duplicate_candidates_in_the_selection():
    winners = [_winner(f"c{i}", AgentName.ANALYST) for i in range(10)]
    selected, _ = _select_deep_analysis_winners(winners, budget=4)
    ids = [c.candidate_id for c, _ in selected]
    assert len(ids) == len(set(ids))


def _winner_with_confidence(candidate_id: str, agent: AgentName, confidence: float):
    candidate = _triage_candidate(candidate_id)
    candidate = replace(candidate, primary_agent=agent)
    return candidate, TriageItemResult(candidate_id=candidate.candidate_id, relevant=True, confidence=confidence)


def test_stronger_candidates_are_selected_ahead_of_weaker_ones_in_the_same_domain():
    """Real-run defect (CURRENT run `beeb562aba597399`): within one domain's
    allocated cap, selection previously took candidates in pre-triage rank
    order, so a weaker triage-confirmed candidate (confidence 0.85, a
    Faraday Future investor-report mention) won a "startup" deep-analysis
    slot ahead of a stronger one from the SAME domain (confidence 0.95, the
    real dated SWI Group capital-increase announcement). Within-domain
    selection must now rank by the triage model's own confidence."""
    weak_first = _winner_with_confidence("weak", AgentName.STARTUP, 0.85)
    strong_second = _winner_with_confidence("strong", AgentName.STARTUP, 0.95)
    selected, dropped = _select_deep_analysis_winners([weak_first, strong_second], budget=1)

    assert dropped == 1
    assert [c.candidate_id for c, _ in selected] == [strong_second[0].candidate_id]


def test_within_domain_confidence_ranking_is_stable_for_ties():
    """Equal-confidence candidates keep their original (pre-triage rank)
    relative order -- the fix only breaks ties by confidence, it never
    reshuffles genuinely tied candidates."""
    a = _winner_with_confidence("a", AgentName.STARTUP, 0.9)
    b = _winner_with_confidence("b", AgentName.STARTUP, 0.9)
    c = _winner_with_confidence("c", AgentName.STARTUP, 0.9)
    selected, _ = _select_deep_analysis_winners([a, b, c], budget=2)
    assert [cand.candidate_id for cand, _ in selected] == [a[0].candidate_id, b[0].candidate_id]


def test_domain_fairness_unaffected_by_within_domain_confidence_ranking():
    """The fix changes WHICH candidates win within a domain's own cap, never
    HOW MANY slots each domain gets -- domain-fair coverage must still hold
    exactly as before."""
    peer_winners = [_winner_with_confidence(f"p{i}", AgentName.PEER, 0.5 + i * 0.01) for i in range(5)]
    ai_winners = [_winner_with_confidence(f"a{i}", AgentName.AI_PRODUCT, 0.9 - i * 0.01) for i in range(5)]
    selected, dropped = _select_deep_analysis_winners(peer_winners + ai_winners, budget=4)

    assert len(selected) == 4
    assert dropped == 6
    selected_agents = {pair[0].primary_agent for pair in selected}
    assert selected_agents == {AgentName.PEER, AgentName.AI_PRODUCT}


# ---------------------------------------------------------------------------
# D. Ollama URL default
# ---------------------------------------------------------------------------


def test_configured_default_ollama_base_url_is_127_0_0_1():
    """`config/settings.yaml`'s own default (not an environment override)
    must be `127.0.0.1`, not `localhost` -- a direct benchmark on this
    machine measured ~2.1s of pure DNS/connection overhead per call when
    the literal string "localhost" is resolved."""
    import yaml

    from research_assistant.utils.paths import SETTINGS_PATH

    raw = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))
    assert raw["system"]["llm"]["base_url"] == "http://127.0.0.1:11434"


def test_create_ollama_client_still_reads_base_url_from_config(monkeypatch):
    """Whatever `config.settings.system.llm.base_url` resolves to (file
    default or an env override -- the existing, unchanged override
    mechanism), `create_ollama_client` must still use it exactly, with no
    hardcoded fallback anywhere in `llm/client.py`."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    config = get_config.__wrapped__()  # fresh load, bypassing the process-wide cache
    client = create_ollama_client(config)
    assert client.base_url == config.settings.system.llm.base_url.rstrip("/")
    assert client.base_url == "http://127.0.0.1:11434"
