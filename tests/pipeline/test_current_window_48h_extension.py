"""Integration proof for the CURRENT-window widening (24h -> 48h,
management requirement): `scheduling.period.compute_current_period`'s
configured width feeds directly into `pipeline.nodes._apply_current_date_gate`
with no other eligibility rule touched. Fixed timestamps only -- no live
Ollama, no retrieval/crawling, no real database.

This file proves the SEAM between the two pieces the task asked about;
`tests/scheduling/test_current_period.py` covers `compute_current_period`'s
own math in isolation, and `tests/pipeline/test_current_hard_date_gate.py`
covers `_apply_current_date_gate`'s own generic (window-width-agnostic)
rules in isolation -- neither of those files needed to change for this
task, which is itself proof the gate and the period math are properly
decoupled from any specific window width."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.config import get_config
from research_assistant.evidence.service import NormalizationService
from research_assistant.models.article import DatePrecision
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.pipeline.nodes import _apply_current_date_gate
from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition
from research_assistant.scheduling.period import compute_current_period, compute_digest_period

NOW = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)


def _source() -> SourceDefinition:
    return SourceDefinition(
        name="Reuters", domain="reuters.com", urls=["https://www.reuters.com/technology/artificial-intelligence/"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
    )


def _article(published_at, *, precision=DatePrecision.EXACT, url="https://reuters.com/story"):
    extracted = ExtractedArticle(
        url=url, title="A real headline about a specific development",
        text="Reuters reported that a major bank today announced a new generative AI platform. " * 5,
        published_at=published_at,
    )
    article = NormalizationService().normalize(extracted, source=_source())
    if published_at is None:
        return article.model_copy(update={"publication_date": None, "publication_date_precision": DatePrecision.UNKNOWN})
    return article.model_copy(update={"publication_date_precision": precision})


# ---------------------------------------------------------------------------
# 1. The CURRENT period itself is 48 hours wide (default config)
# ---------------------------------------------------------------------------


def test_current_period_is_exactly_48_hours_wide():
    period_start, period_end = compute_current_period(NOW)
    assert period_end - period_start == timedelta(hours=48)


def test_current_period_bounds_are_timezone_aware_utc():
    period_start, period_end = compute_current_period(NOW)
    assert period_start.tzinfo is not None and period_end.tzinfo is not None
    assert period_start.utcoffset() == timedelta(0)
    assert period_end.utcoffset() == timedelta(0)


# ---------------------------------------------------------------------------
# 2. An article published inside the 48h window passes the date gate,
#    subject to the existing (unchanged) eligibility rules
# ---------------------------------------------------------------------------


def test_article_published_36_hours_ago_passes_the_date_gate_in_the_48h_window():
    """36h old would have been STALE under the old 24h window but is
    comfortably inside the new 48h one."""
    period_start, period_end = compute_current_period(NOW)
    article = _article(NOW - timedelta(hours=36))

    kept, counts = _apply_current_date_gate([article], period_start, period_end)

    assert kept == [article]
    assert counts == {"undated": 0, "stale": 0, "future": 0}


def test_article_published_exactly_at_period_start_is_includable_half_open():
    period_start, period_end = compute_current_period(NOW)
    article = _article(period_start)

    kept, counts = _apply_current_date_gate([article], period_start, period_end)

    assert kept == [article]
    assert counts == {"undated": 0, "stale": 0, "future": 0}


# ---------------------------------------------------------------------------
# 3. An article just before the 48h boundary is rejected (stale)
# ---------------------------------------------------------------------------


def test_article_one_second_before_the_48h_boundary_is_rejected_as_stale():
    period_start, period_end = compute_current_period(NOW)
    article = _article(period_start - timedelta(seconds=1))

    kept, counts = _apply_current_date_gate([article], period_start, period_end)

    assert kept == []
    assert counts == {"undated": 0, "stale": 1, "future": 0}


def test_article_49_hours_old_is_rejected_as_stale_under_the_48h_window():
    period_start, period_end = compute_current_period(NOW)
    article = _article(NOW - timedelta(hours=49))

    kept, counts = _apply_current_date_gate([article], period_start, period_end)

    assert kept == []
    assert counts == {"undated": 0, "stale": 1, "future": 0}


# ---------------------------------------------------------------------------
# 4. An undated article remains rejected
# ---------------------------------------------------------------------------


def test_undated_article_remains_rejected_under_the_48h_window():
    period_start, period_end = compute_current_period(NOW)
    article = _article(None)

    kept, counts = _apply_current_date_gate([article], period_start, period_end)

    assert kept == []
    assert counts == {"undated": 1, "stale": 0, "future": 0}


# ---------------------------------------------------------------------------
# 5. A future-dated article remains rejected
# ---------------------------------------------------------------------------


def test_future_dated_article_remains_rejected_under_the_48h_window():
    period_start, period_end = compute_current_period(NOW)
    article = _article(period_end)

    kept, counts = _apply_current_date_gate([article], period_start, period_end)

    assert kept == []
    assert counts == {"undated": 0, "stale": 0, "future": 1}


# ---------------------------------------------------------------------------
# 6. DATE_ONLY / EXACT precision behavior is unchanged by the wider window
# ---------------------------------------------------------------------------


def test_date_only_precision_inside_the_48h_window_is_trustworthy_and_accepted():
    period_start, period_end = compute_current_period(NOW)
    article = _article(NOW - timedelta(hours=40), precision=DatePrecision.DATE_ONLY)

    kept, counts = _apply_current_date_gate([article], period_start, period_end)

    assert kept == [article]
    assert counts == {"undated": 0, "stale": 0, "future": 0}


def test_month_precision_inside_the_48h_window_is_still_untrustworthy():
    """Unchanged rule: only EXACT/DATE_ONLY are trustworthy for CURRENT --
    a coarser precision is still treated as effectively undated, even
    though its raw timestamp lands inside the wider 48h window."""
    period_start, period_end = compute_current_period(NOW)
    article = _article(NOW - timedelta(hours=10), precision=DatePrecision.MONTH)

    kept, counts = _apply_current_date_gate([article], period_start, period_end)

    assert kept == []
    assert counts["undated"] == 1


# ---------------------------------------------------------------------------
# 7. Morning/Evening period math is completely unaffected
# ---------------------------------------------------------------------------


def test_morning_and_evening_periods_are_unaffected_by_the_current_window_change():
    config = get_config()
    morning_start, morning_end = compute_digest_period(DigestType.MORNING, NOW, config)
    evening_start, evening_end = compute_digest_period(DigestType.EVENING, NOW, config)

    assert morning_end - morning_start == timedelta(hours=12)
    assert evening_end - evening_start == timedelta(hours=12)
