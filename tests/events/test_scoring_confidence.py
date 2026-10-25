"""Deterministic confidence scoring: primary-only, primary+secondary, multiple
secondary, uncertain dates, complete/incomplete evidence, entity/classification
confidence, determinism, breakdown transparency."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.events.builder import build_event_draft
from research_assistant.events.clustering import Cluster
from research_assistant.events.models import ClusterMetadata
from research_assistant.events.scoring import compute_confidence_breakdown
from research_assistant.models.enums import EventType, GroundingType, SourceType

from tests.events.conftest import make_item

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def _draft_and_items(items):
    cluster = Cluster(items=items, metadata=ClusterMetadata(member_count=len(items)))
    return build_event_draft(cluster), items


def test_primary_source_only_gives_moderate_corroboration_but_full_confirmation(config):
    item = make_item("a1", company="X", title="t", source_type=SourceType.PRIMARY, publication_date=D)
    draft, items = _draft_and_items([item])

    breakdown = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    assert breakdown.primary_source_confirmation == 1.0
    assert breakdown.multiple_source_corroboration == 0.3  # only 1 source total


def test_primary_plus_one_secondary_increases_corroboration(config):
    primary = make_item("a1", company="X", title="t", source_type=SourceType.PRIMARY, publication_date=D)
    secondary = make_item("a2", company="X", title="t", source_type=SourceType.SECONDARY, publication_date=D)
    draft, items = _draft_and_items([primary, secondary])

    breakdown = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    assert breakdown.primary_source_confirmation == 1.0  # primary still selected as the primary_source
    assert breakdown.multiple_source_corroboration == 0.6


def test_multiple_secondary_sources_gives_full_corroboration(config):
    primary = make_item("a1", company="X", title="t", source_type=SourceType.PRIMARY, publication_date=D)
    s1 = make_item("a2", company="Reuters", title="t", source_type=SourceType.SECONDARY, publication_date=D, entities=["X"])
    s2 = make_item("a3", company="Bloomberg", title="t", source_type=SourceType.SECONDARY, publication_date=D, entities=["X"])
    draft, items = _draft_and_items([primary, s1, s2])

    breakdown = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    assert breakdown.multiple_source_corroboration == 1.0


def test_no_primary_source_only_secondary_scores_lower_confirmation(config):
    s1 = make_item("a1", company="Reuters", title="t", source_type=SourceType.SECONDARY, publication_date=D)
    draft, items = _draft_and_items([s1])

    breakdown = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    assert breakdown.primary_source_confirmation == 0.5  # SECONDARY, per the documented mapping


def test_press_distribution_scores_lowest_confirmation(config):
    p = make_item("a1", company="X", title="t", source_type=SourceType.PRESS_DISTRIBUTION, publication_date=D)
    draft, items = _draft_and_items([p])

    breakdown = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    assert breakdown.primary_source_confirmation == 0.3


def test_uncertain_missing_publication_date_scores_zero_certainty(config):
    item = make_item("a1", company="X", title="t", publication_date=None)
    draft, items = _draft_and_items([item])

    breakdown = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    assert breakdown.publication_date_certainty == 0.0


def test_known_publication_date_scores_full_certainty(config):
    item = make_item("a1", company="X", title="t", publication_date=D)
    draft, items = _draft_and_items([item])

    breakdown = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    assert breakdown.publication_date_certainty == 1.0


def test_complete_evidence_scores_full_content_completeness(config):
    item = make_item(
        "a1", company="X", title="t", summary="A summary.", event_type=EventType.PRODUCT_LAUNCH,
        entities=["X"], technologies=["GenAI"], geography=["United States"],
    )
    draft, items = _draft_and_items([item])

    breakdown = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    assert breakdown.content_completeness == 1.0


def test_incomplete_evidence_scores_partial_content_completeness(config):
    item = make_item("a1", company="X", title=None, summary=None, event_type=None, entities=[], technologies=[], geography=[])
    draft, items = _draft_and_items([item])

    breakdown = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    assert breakdown.content_completeness == 0.0


def test_missing_entity_information_scores_zero_entity_confidence(config):
    item = make_item("a1", company="X", title="t", entities=[], people=[], products=[], technologies=[])
    draft, items = _draft_and_items([item])

    breakdown = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    assert breakdown.entity_extraction_confidence == 0.0


def test_three_or_more_distinct_extracted_names_gives_full_entity_confidence(config):
    item = make_item("a1", company="X", title="t", entities=["A", "B"], people=["Jane Doe"])
    draft, items = _draft_and_items([item])

    breakdown = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    assert breakdown.entity_extraction_confidence == 1.0


def test_event_classification_confidence_reflects_whether_type_is_known(config):
    known = make_item("a1", company="X", title="t", event_type=EventType.FUNDING)
    unknown = make_item("a2", company="X", title="t", event_type=None)

    known_draft, known_items = _draft_and_items([known])
    unknown_draft, unknown_items = _draft_and_items([unknown])

    known_breakdown = compute_confidence_breakdown(known_draft, known_items, weights=config.settings.ranking.confidence)
    unknown_breakdown = compute_confidence_breakdown(unknown_draft, unknown_items, weights=config.settings.ranking.confidence)

    assert known_breakdown.event_classification_confidence == 1.0
    assert unknown_breakdown.event_classification_confidence == 0.5


def test_confidence_weights_sum_to_one(config):
    weights = config.settings.ranking.confidence.model_dump()
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_confidence_scoring_is_deterministic_across_repeated_calls(config):
    item = make_item("a1", company="X", title="t", publication_date=D, entities=["A", "B", "C"])
    draft, items = _draft_and_items([item])

    first = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)
    second = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    assert first == second


def test_confidence_breakdown_exposes_every_component():
    item = make_item("a1", company="X", title="t")
    from research_assistant.config import get_config

    draft, items = _draft_and_items([item])
    breakdown = compute_confidence_breakdown(draft, items, weights=get_config().settings.ranking.confidence)

    for field_name in (
        "primary_source_confirmation", "multiple_source_corroboration", "publication_date_certainty",
        "entity_extraction_confidence", "event_classification_confidence", "content_completeness", "weighted_total",
    ):
        assert hasattr(breakdown, field_name)


# ---------------------------------------------------------------------------
# LINK-grounded confidence ceiling: a cluster made ENTIRELY of link-grounded
# (metadata-only) items can never score above _LINK_GROUNDED_CONFIDENCE_CEILING
# (0.6), but a cluster with EVEN ONE content-grounded item is unaffected.
# ---------------------------------------------------------------------------


def test_all_link_grounded_cluster_is_capped_at_the_confidence_ceiling(config):
    item = make_item(
        "a1", company="Financial Times", title="t", source_type=SourceType.SECONDARY,
        publication_date=D, is_dated_development=True,
    ).model_copy(update={"grounding_type": GroundingType.LINK})
    draft, items = _draft_and_items([item])

    breakdown = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    assert breakdown.weighted_total <= 0.6


def test_mixed_grounding_cluster_is_not_capped(config):
    content_item = make_item(
        "a1", company="X", title="t", source_type=SourceType.PRIMARY, publication_date=D,
        entities=["X"],
    )  # default grounding_type is CONTENT
    link_item = make_item(
        "a2", company="X", title="t", source_type=SourceType.SECONDARY, publication_date=D, entities=["X"],
    ).model_copy(update={"grounding_type": GroundingType.LINK})
    draft, items = _draft_and_items([content_item, link_item])

    breakdown = compute_confidence_breakdown(draft, items, weights=config.settings.ranking.confidence)

    # Full-signal cluster (primary + secondary corroboration, entities present)
    # should clear the 0.6 ceiling that applies only to all-LINK clusters.
    assert breakdown.weighted_total > 0.6
