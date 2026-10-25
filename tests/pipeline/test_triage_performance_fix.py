"""Focused regression tests for the CURRENT triage performance fix:
smaller batches (8->3), same 24-candidate bound (now 3*8 instead of
8*3), a longer timeout (60->90), and a trimmed structured-output schema
(no more free-text `reason`). No live Ollama/network anywhere -- fakes
only."""

from __future__ import annotations

from research_assistant.agents.triage import TriageCandidate, TriageItemResult
from research_assistant.config import get_config
from research_assistant.models.enums import AgentName

from tests.pipeline.conftest import FakeTriageLLMClient
from tests.pipeline.test_current_llm_funnel import _article_and_evidence
from research_assistant.pipeline.nodes import _run_current_triage

CONFIG = get_config()


def _triage_candidate(candidate_id: str) -> TriageCandidate:
    article, evidence = _article_and_evidence(
        url=f"https://jpmorganchase.com/{candidate_id}",
        title=f"JPMorgan Chase story {candidate_id}: AI partnership announced",
        text="JPMorgan Chase today announced a new AI partnership and deployed a GenAI platform. " * 10,
    )
    return TriageCandidate(candidate_id=article.id, article=article, evidence=evidence, primary_agent=AgentName.PEER)


# ---------------------------------------------------------------------------
# Configuration values actually in effect
# ---------------------------------------------------------------------------


def test_real_config_triage_batch_size_is_3():
    assert CONFIG.settings.current_llm.triage_batch_size == 3


def test_real_config_max_triage_batches_is_24():
    """Tripled 2026-10-15 (see config/settings.yaml's own comment): the
    discovery fixes (CNBC sitemap scavenging, Yahoo Finance, TradingView)
    grew the qualifying candidate pool well beyond what the old 24-slot
    triage sample could meaningfully cover."""
    assert CONFIG.settings.current_llm.max_triage_batches == 24


def test_real_config_candidate_budget_is_now_72():
    assert (
        CONFIG.settings.current_llm.triage_batch_size * CONFIG.settings.current_llm.max_triage_batches == 72
    )


def test_real_config_triage_timeout_is_90():
    assert CONFIG.settings.current_llm.triage_timeout_seconds == 90


# ---------------------------------------------------------------------------
# Triage schema: `reason` removed, required fields kept
# ---------------------------------------------------------------------------


def test_triage_item_result_no_longer_has_a_reason_field():
    assert "reason" not in TriageItemResult.model_fields


def test_triage_item_result_still_has_every_field_downstream_relies_on():
    required_or_present = TriageItemResult.model_fields
    for field in ("candidate_id", "relevant", "event_type", "category", "confidence"):
        assert field in required_or_present


def test_triage_item_result_constructs_without_a_reason_argument():
    # Downstream (run_triage_batch's schema.model_validate) must not need
    # a `reason` key in the model's JSON output anymore.
    result = TriageItemResult(candidate_id="c1", relevant=True, confidence=0.9)
    assert result.relevant is True


# ---------------------------------------------------------------------------
# Batch count with the real production batch size (24 candidates -> at
# most 8 sequential batches; fewer -> fewer batches; zero -> zero batches)
# ---------------------------------------------------------------------------


def test_24_candidates_at_production_batch_size_yields_at_most_8_batches():
    pool = [_triage_candidate(f"c{i}") for i in range(24)]
    client = FakeTriageLLMClient()
    diagnostics = []

    winners, metrics = _run_current_triage(
        pool,
        triage_client=client,
        batch_size=CONFIG.settings.current_llm.triage_batch_size,
        max_batches=CONFIG.settings.current_llm.max_triage_batches,
        diagnostics=diagnostics,
    )

    assert metrics["candidates_sent_to_triage"] == 24
    assert metrics["triage_batches_attempted"] == 8  # 24 / 3 == 8 exactly
    assert len(winners) == 24


def test_fewer_than_24_candidates_at_production_batch_size():
    pool = [_triage_candidate(f"c{i}") for i in range(7)]
    client = FakeTriageLLMClient()
    diagnostics = []

    winners, metrics = _run_current_triage(
        pool,
        triage_client=client,
        batch_size=CONFIG.settings.current_llm.triage_batch_size,
        max_batches=CONFIG.settings.current_llm.max_triage_batches,
        diagnostics=diagnostics,
    )

    assert metrics["candidates_sent_to_triage"] == 7
    assert metrics["triage_batches_attempted"] == 3  # ceil(7/3)
    assert len(winners) == 7


def test_zero_candidates_at_production_batch_size():
    client = FakeTriageLLMClient()
    diagnostics = []

    winners, metrics = _run_current_triage(
        [],
        triage_client=client,
        batch_size=CONFIG.settings.current_llm.triage_batch_size,
        max_batches=CONFIG.settings.current_llm.max_triage_batches,
        diagnostics=diagnostics,
    )

    assert winners == []
    assert metrics["triage_batches_attempted"] == 0
    assert metrics["candidates_sent_to_triage"] == 0
    assert client.calls == []
