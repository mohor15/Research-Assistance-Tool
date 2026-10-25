"""Event clustering: the 14 required scenarios, deterministic IDs, and provenance."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.events.builder import build_event_draft, generate_event_id
from research_assistant.events.clustering import cluster_research_items
from research_assistant.models.enums import EventType, SourcePriority, SourceType

from tests.events.conftest import make_item

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def _cluster_titles(clusters):
    return [{item.id for item in c.items} for c in clusters]


def test_same_event_same_company_different_urls_clusters():
    a = make_item("a1", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["AI assistant"],
                  url="https://jpmorganchase.com/a")
    b = make_item("a2", company="JPMorgan Chase", title="JPMorgan introduces new AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["AI assistant"],
                  url="https://jpmorganchase.com/b")

    clusters = cluster_research_items([a, b])

    assert len(clusters) == 1
    assert {item.id for item in clusters[0].items} == {"a1", "a2"}


def test_same_event_different_publishers_clusters():
    a = make_item("a1", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["AI assistant"],
                  source_type=SourceType.PRIMARY)
    b = make_item("a2", company="Reuters", title="Reuters reports JPMorgan AI assistant launch",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D,
                  entities=["JPMorgan Chase", "AI assistant"], source_type=SourceType.SECONDARY)

    clusters = cluster_research_items([a, b])

    assert len(clusters) == 1
    assert {item.id for item in clusters[0].items} == {"a1", "a2"}


def test_same_company_different_events_does_not_cluster():
    a = make_item("a1", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D)
    b = make_item("a2", company="JPMorgan Chase", title="JPMorgan expands AI hiring",
                  event_type=EventType.HIRING, publication_date=D)

    clusters = cluster_research_items([a, b])

    assert len(clusters) == 2


def test_different_companies_same_topic_does_not_cluster():
    a = make_item("a1", company="JPMorgan Chase", title="JPMorgan discusses enterprise AI adoption",
                  event_type=EventType.STRATEGIC_INITIATIVE, publication_date=D)
    b = make_item("a2", company="Citi", title="Citi discusses enterprise AI adoption",
                  event_type=EventType.STRATEGIC_INITIATIVE, publication_date=D)

    clusters = cluster_research_items([a, b])

    assert len(clusters) == 2


def test_different_event_types_does_not_cluster():
    a = make_item("a1", company="Acme AI", title="Acme AI raises funding", event_type=EventType.FUNDING, publication_date=D)
    b = make_item("a2", company="Acme AI", title="Acme AI announces funding partnership", event_type=EventType.PARTNERSHIP, publication_date=D)

    clusters = cluster_research_items([a, b])

    assert len(clusters) == 2


def test_far_apart_dates_does_not_cluster():
    a = make_item("a1", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["AI assistant"])
    b = make_item("a2", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D + timedelta(days=60), entities=["AI assistant"])

    clusters = cluster_research_items([a, b])

    assert len(clusters) == 2


def test_acquisition_vs_partnership_does_not_cluster():
    a = make_item("a1", company="Bank X", title="Bank X partners with AI Company Y",
                  event_type=EventType.PARTNERSHIP, publication_date=D, partnerships=["AI Company Y"])
    b = make_item("a2", company="Bank X", title="Bank X acquires AI Company Y",
                  event_type=EventType.ACQUISITION, publication_date=D, partnerships=["AI Company Y"])

    clusters = cluster_research_items([a, b])

    assert len(clusters) == 2


def test_model_release_a_vs_model_release_b_does_not_cluster():
    a = make_item("a1", company="OpenAI", title="OpenAI releases Model A",
                  event_type=EventType.MODEL_RELEASE, publication_date=D, products=["Model A"])
    b = make_item("a2", company="OpenAI", title="OpenAI releases Model B",
                  event_type=EventType.MODEL_RELEASE, publication_date=D, products=["Model B"])

    clusters = cluster_research_items([a, b])

    assert len(clusters) == 2


def test_transitive_clustering_groups_all_three():
    """A~B and B~C pass threshold, A~C alone might not — all three still group."""
    a = make_item("a1", company="JPMorgan Chase", title="JPMorgan launches new AI assistant for advisors",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["AI assistant"],
                  summary="JPMorgan launched a GenAI assistant for financial advisors today.")
    b = make_item("a2", company="JPMorgan Chase", title="JPMorgan AI assistant rolls out to advisors nationwide",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["AI assistant"],
                  summary="The GenAI assistant for advisors is now rolling out nationwide at JPMorgan.")
    c = make_item("a3", company="Reuters", title="Reuters: JPMorgan advisor AI tool now live nationwide",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["JPMorgan Chase", "AI assistant"],
                  summary="JPMorgan's advisor AI tool is now live nationwide, Reuters reports.")

    clusters = cluster_research_items([a, b, c])

    assert len(clusters) == 1
    assert {item.id for item in clusters[0].items} == {"a1", "a2", "a3"}


def test_duplicate_article_objects_handled_gracefully():
    a = make_item("a1", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, article_id="same-article")
    b = make_item("a2", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, article_id="same-article")

    clusters = cluster_research_items([a, b])

    assert len(clusters) == 1
    assert len(clusters[0].items) == 2


def test_empty_missing_metadata_does_not_crash():
    a = make_item("a1", company="JPMorgan Chase", title=None, event_type=None, publication_date=None, summary=None)
    b = make_item("a2", company="Citi", title=None, event_type=None, publication_date=None, summary=None)

    clusters = cluster_research_items([a, b])

    assert len(clusters) == 2  # different companies, no other signal to merge them


def test_single_item_with_no_metadata_forms_its_own_cluster():
    a = make_item("a1", company="JPMorgan Chase", title=None, event_type=None, publication_date=None, summary=None)

    clusters = cluster_research_items([a])

    assert len(clusters) == 1
    assert clusters[0].items[0].id == "a1"


def test_deterministic_cluster_ids_are_stable_regardless_of_input_order():
    a = make_item("a1", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["AI assistant"])
    b = make_item("a2", company="JPMorgan Chase", title="JPMorgan introduces new AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["AI assistant"])

    clusters_1 = cluster_research_items([a, b])
    clusters_2 = cluster_research_items([b, a])

    draft_1 = build_event_draft(clusters_1[0])
    draft_2 = build_event_draft(clusters_2[0])

    assert draft_1.id == draft_2.id
    assert draft_1.id == generate_event_id(["a1", "a2"])
    assert generate_event_id(["a1", "a2"]) == generate_event_id(["a2", "a1"])  # order-independent


def test_cluster_provenance_records_matched_signals_and_scores():
    a = make_item("a1", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["AI assistant"])
    b = make_item("a2", company="JPMorgan Chase", title="JPMorgan introduces new AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D, entities=["AI assistant"])

    clusters = cluster_research_items([a, b])
    metadata = clusters[0].metadata

    assert metadata.member_count == 2
    assert len(metadata.matches) == 1
    match = metadata.matches[0]
    assert {match.research_item_id_a, match.research_item_id_b} == {"a1", "a2"}
    assert match.combined_score > 0
    assert len(match.matched_signals) > 0


def test_singleton_cluster_has_empty_matches_and_member_count_one():
    a = make_item("a1", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                  event_type=EventType.PRODUCT_LAUNCH, publication_date=D)

    clusters = cluster_research_items([a])

    assert clusters[0].metadata.member_count == 1
    assert clusters[0].metadata.matches == []


def test_primary_secondary_source_selection_after_clustering():
    primary_item = make_item("a1", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                              event_type=EventType.PRODUCT_LAUNCH, publication_date=D,
                              source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
                              entities=["AI assistant"])
    press_item = make_item("a2", company="JPMorgan Chase", title="JPMorgan launches new AI assistant (press release)",
                            event_type=EventType.PRODUCT_LAUNCH, publication_date=D,
                            source_type=SourceType.PRESS_DISTRIBUTION, source_priority=SourcePriority.MEDIUM,
                            entities=["AI assistant"])
    secondary_item = make_item("a3", company="Reuters", title="Reuters reports JPMorgan AI assistant launch",
                                event_type=EventType.PRODUCT_LAUNCH, publication_date=D,
                                source_type=SourceType.SECONDARY, source_priority=SourcePriority.HIGH,
                                entities=["JPMorgan Chase", "AI assistant"])

    clusters = cluster_research_items([press_item, secondary_item, primary_item])
    draft = build_event_draft(clusters[0])

    # The genuine primary source outranks both press distribution and secondary,
    # regardless of input order — press distribution is never treated as
    # independent journalism, let alone promoted above a real primary source.
    assert draft.primary_source.research_item_id == "a1"
    assert {ref.research_item_id for ref in draft.secondary_sources} == {"a2", "a3"}


def test_empty_input_list_returns_no_clusters():
    assert cluster_research_items([]) == []
