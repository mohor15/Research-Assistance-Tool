"""Part 1 fix: a DATE_ONLY publication_date (stored as UTC midnight on the
known calendar day -- see evidence.normalizer) must not be compared as if
midnight were the literal observed publication instant for Morning/Evening
calendar-date digests. Confirmed root cause of the empty 2026-10-04 Evening
digest: FT articles dated that day (date_only, 00:00 UTC) were excluded
because Evening's period began at 02:30 UTC. CURRENT's exact [now-24h, now)
semantics are deliberately left unchanged."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.models.article import DatePrecision
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.models import ExtractedArticle, ExtractionStatus, SourceCategory, SourceDefinition
from research_assistant.evidence.normalizer import normalize_article
from research_assistant.pipeline.nodes import _date_only_calendar_day_overlaps_period, _filter_by_period


def _source() -> SourceDefinition:
    return SourceDefinition(
        name="Financial Times", domain="ft.com", urls=["https://www.ft.com/artificial-intelligence"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
    )


def _exact_article(published_at):
    extracted = ExtractedArticle(
        url="https://www.ft.com/content/a", published_at=published_at, status=ExtractionStatus.SUCCESS, text="x" * 50,
    )
    return normalize_article(extracted, source=_source())


def _date_only_article(published_at_midnight_utc):
    extracted = ExtractedArticle(
        url="https://www.ft.com/content/b", published_at=published_at_midnight_utc,
        status=ExtractionStatus.SUCCESS, text="x" * 50,
    )
    article = normalize_article(extracted, source=_source())
    assert article.publication_date_precision == DatePrecision.DATE_ONLY  # sanity: midnight -> DATE_ONLY
    return article


_EVENING_START = datetime(2026, 10, 4, 2, 30, 0, tzinfo=timezone.utc)
_EVENING_END = datetime(2026, 10, 4, 14, 30, 0, tzinfo=timezone.utc)


# 1-3: exact timestamp precision is completely unchanged by this fix.


def test_exact_timestamp_inside_evening_window_is_eligible():
    article = _exact_article(datetime(2026, 10, 4, 9, 0, 0, tzinfo=timezone.utc))
    kept, excluded = _filter_by_period([article], _EVENING_START, _EVENING_END, calendar_date_aware=True)
    assert kept == [article]


def test_exact_timestamp_before_evening_window_is_excluded():
    article = _exact_article(datetime(2026, 10, 4, 1, 0, 0, tzinfo=timezone.utc))
    kept, excluded = _filter_by_period([article], _EVENING_START, _EVENING_END, calendar_date_aware=True)
    assert excluded == [article]


def test_exact_timestamp_at_period_end_is_still_eligible_at_the_article_filter_stage():
    """`_filter_by_period` itself is a closed interval (unchanged, pre-
    existing behavior); the strict half-open [start, end) exclusion is
    enforced one stage later, at event-level digest qualification (see
    tests/pipeline/test_digest_boundary_e2e.py) -- not re-tested here to
    avoid duplicating that existing coverage."""
    article = _exact_article(_EVENING_END)
    kept, excluded = _filter_by_period([article], _EVENING_START, _EVENING_END, calendar_date_aware=True)
    assert kept == [article]


# 4-5: the actual fix -- date-only calendar-day overlap.


def test_date_only_article_on_the_evening_digest_calendar_date_is_not_excluded():
    """The exact confirmed-defect scenario: FT article dated 2026-10-04
    (date_only, i.e. stored as 2026-10-04T00:00:00Z), Evening period
    starting at 02:30Z the same day -- must now be ELIGIBLE, not excluded
    merely because midnight < period_start."""
    article = _date_only_article(datetime(2026, 10, 4, 0, 0, 0, tzinfo=timezone.utc))
    kept, excluded = _filter_by_period([article], _EVENING_START, _EVENING_END, calendar_date_aware=True)
    assert kept == [article]
    # the fix never relabels precision or invents a time
    assert kept[0].publication_date_precision == DatePrecision.DATE_ONLY
    assert kept[0].publication_date == datetime(2026, 10, 4, 0, 0, 0, tzinfo=timezone.utc)


def test_date_only_article_from_the_previous_calendar_day_is_excluded():
    article = _date_only_article(datetime(2026, 10, 3, 0, 0, 0, tzinfo=timezone.utc))
    kept, excluded = _filter_by_period([article], _EVENING_START, _EVENING_END, calendar_date_aware=True)
    assert excluded == [article]


# 6-7: precision/date are never altered; no invented time.


def test_date_only_precision_and_original_date_are_preserved_after_qualification():
    article = _date_only_article(datetime(2026, 10, 4, 0, 0, 0, tzinfo=timezone.utc))
    kept, _ = _filter_by_period([article], _EVENING_START, _EVENING_END, calendar_date_aware=True)
    assert kept[0].publication_date.time().isoformat() == "00:00:00"  # never rewritten to 23:59:59 or any other instant
    assert kept[0].publication_date_precision == DatePrecision.DATE_ONLY


def test_overlap_helper_never_invents_a_specific_time_just_checks_interval_overlap():
    day = datetime(2026, 10, 4, 0, 0, 0, tzinfo=timezone.utc)
    assert _date_only_calendar_day_overlaps_period(day, _EVENING_START, _EVENING_END) is True
    # a period entirely within the NEXT calendar day does not overlap
    next_day_start = datetime(2026, 10, 5, 2, 30, 0, tzinfo=timezone.utc)
    next_day_end = datetime(2026, 10, 5, 14, 30, 0, tzinfo=timezone.utc)
    assert _date_only_calendar_day_overlaps_period(day, next_day_start, next_day_end) is False


# 8: CURRENT is not silently weakened.


def test_current_mode_does_not_use_calendar_date_awareness():
    """CURRENT's own period math (nodes.py call sites) passes
    calendar_date_aware=False -- reconfirmed directly here: with the flag
    off, a date_only article at midnight UTC on the correct calendar day is
    EXCLUDED from a CURRENT-shaped window that starts later the same day,
    exactly as before this fix (conservative, unchanged)."""
    article = _date_only_article(datetime(2026, 10, 4, 0, 0, 0, tzinfo=timezone.utc))
    current_start = datetime(2026, 10, 4, 2, 30, 0, tzinfo=timezone.utc)
    current_end = datetime(2026, 10, 5, 2, 30, 0, tzinfo=timezone.utc)
    kept, excluded = _filter_by_period([article], current_start, current_end, calendar_date_aware=False)
    assert excluded == [article]


def test_agent_analysis_and_normalize_pass_calendar_date_aware_based_on_digest_type():
    """End-to-end confirmation that the two real call sites in nodes.py
    actually wire digest_type -> calendar_date_aware correctly, not just
    the helper function in isolation."""
    from research_assistant.pipeline.nodes import make_agent_analysis_node
    from research_assistant.pipeline.state import PipelineState
    from tests.pipeline.conftest import jpmorgan_source
    from tests.pipeline.test_graph import _build_services

    from research_assistant.config import get_config

    config = get_config()
    article = _date_only_article(datetime(2026, 10, 4, 0, 0, 0, tzinfo=timezone.utc))
    services = _build_services(config, peer_llm_responses=[], report_llm_responses=[])
    node = make_agent_analysis_node(services)

    evening_state = PipelineState(
        digest_type=DigestType.EVENING, period_start=_EVENING_START, period_end=_EVENING_END,
        articles=[article], evidence=[],
    )
    result = node(evening_state)
    # articles_excluded_from_digest_period counts calendar_date_aware-filtered exclusions
    assert result["metrics"]["articles_excluded_from_digest_period"] == 0  # NOT excluded under Evening

    current_state = PipelineState(
        digest_type=DigestType.CURRENT, period_start=_EVENING_START, period_end=_EVENING_END,
        articles=[article], evidence=[],
    )
    result = node(current_state)
    assert result["metrics"]["articles_excluded_from_digest_period"] == 1  # excluded under CURRENT (unchanged)
