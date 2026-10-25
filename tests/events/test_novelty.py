"""Novelty scoring: batch-relative only, no external history required, and never
conflated with recency (an event isn't "novel" merely because it's recent)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.events.builder import EventDraft
from research_assistant.events.models import ClusterMetadata, SourceRef
from research_assistant.events.scoring import novelty_score
from research_assistant.models.enums import EventType, SourcePriority, SourceType

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def _draft(id_, *, company, event_type, event_date):
    ref = SourceRef(
        research_item_id=f"ri-{id_}", article_id=f"art-{id_}", url=f"https://example.com/{id_}",
        publisher=company, source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
    )
    return EventDraft(
        id=id_, canonical_title="t", event_type=event_type, primary_company=company, related_companies=[],
        event_date=event_date, description=None, research_items=[f"ri-{id_}"], articles=[f"art-{id_}"],
        primary_source=ref, secondary_sources=[], cluster_metadata=ClusterMetadata(member_count=1),
    )


def test_only_event_in_batch_is_fully_novel():
    draft = _draft("e1", company="X", event_type=EventType.FUNDING, event_date=D)
    assert novelty_score(draft, [draft]) == 1.0


def test_first_of_its_kind_in_batch_is_fully_novel_even_if_a_later_repeat_exists():
    earlier = _draft("e1", company="X", event_type=EventType.FUNDING, event_date=D)
    later = _draft("e2", company="X", event_type=EventType.FUNDING, event_date=D + timedelta(days=10))
    batch = [earlier, later]

    assert novelty_score(earlier, batch) == 1.0


def test_recurring_theme_already_covered_earlier_in_batch_scores_lower():
    earlier = _draft("e1", company="X", event_type=EventType.FUNDING, event_date=D)
    later = _draft("e2", company="X", event_type=EventType.FUNDING, event_date=D + timedelta(days=10))
    batch = [earlier, later]

    assert novelty_score(later, batch) == 0.4


def test_recency_alone_does_not_make_an_event_novel():
    """A brand-new article about an already-covered theme is still not novel —
    novelty and recency are independent axes."""
    old_first_report = _draft("e1", company="X", event_type=EventType.FUNDING, event_date=D)
    fresh_followup = _draft("e2", company="X", event_type=EventType.FUNDING, event_date=D + timedelta(days=89))
    batch = [old_first_report, fresh_followup]

    assert novelty_score(fresh_followup, batch) == 0.4  # still "not novel" despite being very recent


def test_different_company_does_not_reduce_novelty():
    a = _draft("e1", company="X", event_type=EventType.FUNDING, event_date=D)
    b = _draft("e2", company="Y", event_type=EventType.FUNDING, event_date=D + timedelta(days=1))
    batch = [a, b]

    assert novelty_score(b, batch) == 1.0


def test_different_event_type_does_not_reduce_novelty():
    a = _draft("e1", company="X", event_type=EventType.FUNDING, event_date=D)
    b = _draft("e2", company="X", event_type=EventType.ACQUISITION, event_date=D + timedelta(days=1))
    batch = [a, b]

    assert novelty_score(b, batch) == 1.0


def test_missing_event_date_defaults_to_fully_novel_not_penalized():
    draft = _draft("e1", company="X", event_type=EventType.FUNDING, event_date=None)
    other = _draft("e2", company="X", event_type=EventType.FUNDING, event_date=D)
    assert novelty_score(draft, [draft, other]) == 1.0


def test_novelty_ignores_events_with_missing_dates_in_the_comparison_batch():
    draft = _draft("e1", company="X", event_type=EventType.FUNDING, event_date=D)
    undated_other = _draft("e2", company="X", event_type=EventType.FUNDING, event_date=None)
    assert novelty_score(draft, [draft, undated_other]) == 1.0


def test_novelty_is_computed_against_a_controlled_batch_not_live_external_data():
    """Directly exercises the documented limitation: novelty is only ever
    relative to the batch passed in, never a call to any external service."""
    import inspect

    from research_assistant.events import scoring as scoring_module

    source = inspect.getsource(scoring_module.novelty_score)
    assert "requests" not in source
    assert "http" not in source.lower()
