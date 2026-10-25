"""Near-duplicate-title same-day clustering override (confirmed forensic
finding, CURRENT run `5bac1b659bc8cceb`): the same PR Newswire release,
syndicated under two distinct URLs, produced two ResearchItems with an
identical title and the same exact publication timestamp, but the
weighted `combined_score` (0.51) fell just under `CLUSTER_THRESHOLD`
(0.55) because both items' `summary` was None (content_similarity lost
entirely) and the two independent Qwen calls extracted different
entity/product lists from near-identical text. A near-verbatim title
match on the same day is treated as a direct override, additive only --
it can never defeat an existing hard veto, and never fires for a
materially different title."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.events.clustering import cluster_research_items
from research_assistant.events.similarity import is_near_duplicate_title_same_day, should_cluster
from research_assistant.models.enums import EventType

from tests.events.conftest import make_item

D = datetime(2026, 10, 12, 15, 30, tzinfo=timezone.utc)

_ASUS_TITLE = "ASUS Unveils XA P2N-E2 Powered by NVIDIA Vera CPU for Agentic AI Workloads at OCP 2026"


def test_real_asus_pair_now_clusters():
    """Reproduces the exact real-run scenario: identical title, same
    timestamp, no summary, sparse/divergent entity extraction -- the
    weighted score alone (0.51) is below threshold (0.55), but the
    override makes them cluster."""
    a = make_item(
        "a1", company="PR Newswire", title=_ASUS_TITLE, publication_date=D,
        entities=["ASUS", "NVIDIA", "NVIDIA Vera CPU", "NVIDIA Vera Rubin NVL72", "NVIDIA HGX Rubin NVL8"],
        products=["XA P2N-E2", "XA VR721-E3", "XR R1I12B-M1BHFS"], summary=None,
        url="https://www.prnewswire.com/news-releases/asus-...-302905781.html",
    )
    b = make_item(
        "a2", company="PR Newswire", title=_ASUS_TITLE, publication_date=D,
        entities=["ASUS", "NVIDIA", "Intel"], products=[], summary=None,
        url="https://www.prnewswire.com/news-releases/asus-...-302905807.html",
    )

    assert should_cluster(a, b) is True
    clusters = cluster_research_items([a, b])
    assert len(clusters) == 1
    assert {item.id for item in clusters[0].items} == {"a1", "a2"}


def test_near_duplicate_title_predicate_fires_for_identical_title_same_day():
    a = make_item("a1", company="Acme", title=_ASUS_TITLE, publication_date=D)
    b = make_item("a2", company="Acme", title=_ASUS_TITLE, publication_date=D)
    assert is_near_duplicate_title_same_day(a, b) is True


def test_near_duplicate_title_predicate_tolerates_a_few_hours_gap():
    a = make_item("a1", company="Acme", title=_ASUS_TITLE, publication_date=D)
    b = make_item("a2", company="Acme", title=_ASUS_TITLE, publication_date=D + timedelta(hours=6))
    assert is_near_duplicate_title_same_day(a, b) is True


def test_near_duplicate_title_predicate_requires_close_dates():
    a = make_item("a1", company="Acme", title=_ASUS_TITLE, publication_date=D)
    b = make_item("a2", company="Acme", title=_ASUS_TITLE, publication_date=D + timedelta(days=3))
    assert is_near_duplicate_title_same_day(a, b) is False


def test_near_duplicate_title_predicate_requires_missing_dates_to_not_fire():
    a = make_item("a1", company="Acme", title=_ASUS_TITLE, publication_date=None)
    b = make_item("a2", company="Acme", title=_ASUS_TITLE, publication_date=D)
    assert is_near_duplicate_title_same_day(a, b) is False


def test_materially_different_titles_never_trigger_the_override():
    a = make_item("a1", company="Acme", title="Acme launches a new product today", publication_date=D)
    b = make_item("a2", company="Acme", title="Acme announces quarterly earnings results", publication_date=D)
    assert is_near_duplicate_title_same_day(a, b) is False


def test_override_never_defeats_an_existing_hard_veto():
    """Identical title + same day is NOT enough to override a genuine
    event-type conflict -- hard vetoes are still checked first."""
    a = make_item(
        "a1", company="Acme", title=_ASUS_TITLE, publication_date=D,
        event_type=EventType.ACQUISITION,
    )
    b = make_item(
        "a2", company="Acme", title=_ASUS_TITLE, publication_date=D,
        event_type=EventType.PARTNERSHIP,
    )
    assert is_near_duplicate_title_same_day(a, b) is True  # the predicate itself still fires
    assert should_cluster(a, b) is False  # but the event-type veto wins


def test_override_never_defeats_disjoint_entities_veto():
    a = make_item("a1", company="Acme", title=_ASUS_TITLE, publication_date=D, entities=["CompanyX"])
    b = make_item("a2", company="Acme2", title=_ASUS_TITLE, publication_date=D, entities=["CompanyY"])
    assert should_cluster(a, b) is False


def test_a_title_that_is_only_moderately_similar_does_not_trigger_the_override():
    """Below the 0.9 near-duplicate threshold -- normal weighted scoring
    applies, unaffected by this change."""
    a = make_item("a1", company="Acme", title="ASUS Unveils XA P2N-E2 at OCP 2026", publication_date=D)
    b = make_item("a2", company="Acme", title="ASUS discusses its roadmap at OCP 2026 conference", publication_date=D)
    assert is_near_duplicate_title_same_day(a, b) is False
