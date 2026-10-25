"""Deterministic relevance scoring: all-HIGH/all-LOW/mixed inputs, missing fields,
weight sensitivity, weight-sum validation, determinism, breakdown transparency,
no-LLM, no-hardcoded-weights."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import research_assistant.events.scoring as scoring_module
from research_assistant.events.builder import build_event_draft
from research_assistant.events.clock import FixedClock
from research_assistant.events.clustering import Cluster
from research_assistant.events.models import ClusterMetadata
from research_assistant.events.scoring import compute_relevance_breakdown
from research_assistant.models.enums import EventType
from research_assistant.models.research_item import RelevanceLabel, RelevanceLevel, TopicCategory

from tests.events.conftest import make_item

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def _draft_and_items(items):
    cluster = Cluster(items=items, metadata=ClusterMetadata(member_count=len(items)))
    return build_event_draft(cluster), items


def test_all_high_inputs_yield_maximum_relevance(config):
    item = make_item(
        "a1", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
        event_type=EventType.PRODUCT_LAUNCH, publication_date=D, category=TopicCategory.ENTERPRISE_AI,
        relevance_label=RelevanceLabel.RELEVANT, fs_level=RelevanceLevel.HIGH, strategic_level=RelevanceLevel.HIGH,
    )
    draft, items = _draft_and_items([item])
    clock = FixedClock(D)  # same instant -> maximum recency

    breakdown = compute_relevance_breakdown(draft, items, [draft], weights=config.settings.ranking.relevance, clock=clock)

    assert breakdown.ai_relevance == 1.0
    assert breakdown.financial_services_relevance == 0.75  # HIGH, per the documented mapping
    assert breakdown.strategic_significance == 0.75
    assert breakdown.recency == 1.0
    assert breakdown.novelty == 1.0
    # HIGH tops out at 0.75 (there is no VERY_HIGH tier in the 3-level enum), so
    # the achievable ceiling with all-HIGH qualitative inputs is below 1.0 —
    # verify against the actual weighted formula, not an assumed maximum.
    w = config.settings.ranking.relevance
    expected = w.ai_relevance * 1.0 + w.financial_services_relevance * 0.75 + w.strategic_significance * 0.75 + w.recency * 1.0 + w.novelty * 1.0
    assert breakdown.weighted_total == round(expected, 4)


def test_all_low_inputs_yield_low_relevance(config):
    item = make_item(
        "a1", company="JPMorgan Chase", title="t", event_type=None, publication_date=D - _days(200),
        category=TopicCategory.OTHER, relevance_label=RelevanceLabel.AMBIGUOUS,
        fs_level=RelevanceLevel.LOW, strategic_level=RelevanceLevel.LOW,
    )
    draft, items = _draft_and_items([item])
    clock = FixedClock(D)

    breakdown = compute_relevance_breakdown(draft, items, [draft], weights=config.settings.ranking.relevance, clock=clock)

    assert breakdown.ai_relevance == round(0.4 * 0.6, 4)  # OTHER category multiplier * AMBIGUOUS label score
    assert breakdown.financial_services_relevance == 0.25
    assert breakdown.strategic_significance == 0.25
    assert breakdown.recency == 0.0  # 200 days old, beyond every bucket
    assert breakdown.weighted_total < 0.3


def test_mixed_inputs_across_cluster_members_take_the_strongest_signal(config):
    weak = make_item("a1", company="X", title="t", fs_level=RelevanceLevel.LOW, strategic_level=RelevanceLevel.LOW)
    strong = make_item("a2", company="X", title="t", fs_level=RelevanceLevel.HIGH, strategic_level=RelevanceLevel.HIGH)
    draft, items = _draft_and_items([weak, strong])
    clock = FixedClock(D)

    breakdown = compute_relevance_breakdown(draft, items, [draft], weights=config.settings.ranking.relevance, clock=clock)

    assert breakdown.financial_services_relevance == 0.75
    assert breakdown.strategic_significance == 0.75


def test_missing_event_type_and_other_category_are_handled_without_crashing(config):
    item = make_item("a1", company="X", title="t", event_type=None, category=TopicCategory.OTHER, publication_date=D)
    draft, items = _draft_and_items([item])
    clock = FixedClock(D)

    breakdown = compute_relevance_breakdown(draft, items, [draft], weights=config.settings.ranking.relevance, clock=clock)

    assert breakdown.ai_relevance == round(0.4 * 1.0, 4)


def test_configured_weight_changes_are_reflected_in_the_score(config):
    item = make_item("a1", company="X", title="t", publication_date=D, fs_level=RelevanceLevel.HIGH)
    draft, items = _draft_and_items([item])
    clock = FixedClock(D)

    default_breakdown = compute_relevance_breakdown(draft, items, [draft], weights=config.settings.ranking.relevance, clock=clock)

    boosted_weights = config.settings.ranking.relevance.model_copy(
        update={"ai_relevance": 0.0, "financial_services_relevance": 1.0, "strategic_significance": 0.0, "recency": 0.0, "novelty": 0.0}
    )
    boosted_breakdown = compute_relevance_breakdown(draft, items, [draft], weights=boosted_weights, clock=clock)

    assert boosted_breakdown.weighted_total == boosted_breakdown.financial_services_relevance
    assert boosted_breakdown.weighted_total != default_breakdown.weighted_total


def test_relevance_weights_sum_to_one(config):
    weights = config.settings.ranking.relevance.model_dump()
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_scoring_is_deterministic_across_repeated_calls(config):
    item = make_item("a1", company="X", title="t", publication_date=D)
    draft, items = _draft_and_items([item])
    clock = FixedClock(D)

    first = compute_relevance_breakdown(draft, items, [draft], weights=config.settings.ranking.relevance, clock=clock)
    second = compute_relevance_breakdown(draft, items, [draft], weights=config.settings.ranking.relevance, clock=clock)

    assert first == second


def test_breakdown_exposes_every_component_not_just_the_total(config):
    item = make_item("a1", company="X", title="t", publication_date=D)
    draft, items = _draft_and_items([item])
    clock = FixedClock(D)

    breakdown = compute_relevance_breakdown(draft, items, [draft], weights=config.settings.ranking.relevance, clock=clock)

    for field_name in ("ai_relevance", "financial_services_relevance", "strategic_significance", "recency", "novelty", "weighted_total"):
        assert hasattr(breakdown, field_name)


def test_scoring_module_never_imports_the_llm_layer():
    source = Path(scoring_module.__file__).read_text(encoding="utf-8")
    import_lines = [line.strip() for line in source.splitlines() if line.strip().startswith(("import ", "from "))]
    assert not any("research_assistant.llm" in line for line in import_lines)
    assert not any("research_assistant.retrieval" in line for line in import_lines)


def test_weights_parameter_has_no_default_forcing_config_as_the_source_of_truth():
    import inspect

    signature = inspect.signature(compute_relevance_breakdown)
    assert signature.parameters["weights"].default is inspect._empty


def _days(n: int):
    from datetime import timedelta

    return timedelta(days=n)
