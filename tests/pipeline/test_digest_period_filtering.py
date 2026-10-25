"""Article history retention (Phase 9): `normalize()` now retains a WIDER
window than any single digest's own reporting period -- the existing,
previously-unused `research.lookback_hours` config value -- so that
persisted article history accumulates across runs instead of behaving like
a digest-only system. The NARROW `period_start`/`period_end` window is
still enforced, just one stage later (see
`tests/pipeline/test_digest_candidate_selection.py` for the
`agent_analysis()` narrowing that keeps LLM call volume unaffected, and
`tests/pipeline/test_digest_vs_history_separation.py` for the full,
end-to-end proof that a historically-retained article never leaks into a
digest it doesn't belong to).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from research_assistant.config import get_config
from research_assistant.evidence.service import NormalizationService
from research_assistant.pipeline.nodes import _filter_by_period, make_normalize_node
from research_assistant.pipeline.state import ExtractedBundle, PipelineState
from research_assistant.retrieval.models import ExtractedArticle

from tests.pipeline.conftest import jpmorgan_source

CONFIG = get_config()
LOOKBACK_HOURS = CONFIG.settings.research.lookback_hours  # 24, per config/settings.yaml

# A narrow 12-hour digest window (mirrors a real morning/evening digest span).
PERIOD_START = datetime(2026, 3, 5, 8, 0, tzinfo=timezone.utc)
PERIOD_END = datetime(2026, 3, 5, 20, 0, tzinfo=timezone.utc)


def _article(url: str, *, published_at) -> ExtractedArticle:
    # Distinct text per URL: NormalizationService.deduplicate() clusters by
    # content similarity/hash, so identical bodies would collapse multiple
    # test articles into one regardless of their distinct URLs/dates.
    slug = url.rsplit("/", 1)[-1]
    return ExtractedArticle(
        url=url, title=f"Title {slug}", text=f"Article body about AI in financial services, story '{slug}'.",
        published_at=published_at,
    )


def _services():
    return SimpleNamespace(normalization_service=NormalizationService(), config=CONFIG)


def _state(*articles: ExtractedArticle, period_start=PERIOD_START, period_end=PERIOD_END) -> PipelineState:
    source = jpmorgan_source()
    return PipelineState(
        extracted_bundles=[ExtractedBundle(source=source, extracted=a) for a in articles],
        period_start=period_start, period_end=period_end,
    )


def test_article_inside_the_narrow_digest_period_is_retained():
    node = make_normalize_node(_services())
    inside = _article("https://jpmorganchase.com/a", published_at=PERIOD_START + timedelta(hours=1))

    result = node(_state(inside))

    assert len(result["articles"]) == 1
    assert result["metrics"]["articles_excluded_out_of_lookback"] == 0


def test_article_outside_the_narrow_digest_period_but_inside_the_lookback_window_is_still_retained_for_history():
    """The key Phase 9 behavior change: an article published before the
    digest's own period_start, but still within the wider configured
    lookback_hours window, is now retained as article history rather than
    being dropped -- it just won't be analyzed by any agent this run (see
    test_digest_candidate_selection.py)."""
    node = make_normalize_node(_services())
    # 2 hours before the narrow period_start, well within a 24h lookback.
    just_outside_narrow_period = _article(
        "https://jpmorganchase.com/b", published_at=PERIOD_START - timedelta(hours=2)
    )

    result = node(_state(just_outside_narrow_period))

    assert len(result["articles"]) == 1
    assert result["metrics"]["articles_excluded_out_of_lookback"] == 0


def test_article_after_the_period_end_is_still_excluded():
    """Widening only moves the LOWER bound back further -- period_end
    remains the upper bound for both history and digest purposes (a digest
    run has no business retaining content from the future)."""
    node = make_normalize_node(_services())
    after = _article("https://jpmorganchase.com/c", published_at=PERIOD_END + timedelta(hours=1))

    result = node(_state(after))

    assert result["articles"] == []
    assert result["metrics"]["articles_excluded_out_of_lookback"] == 1


def test_article_outside_the_lookback_boundary_entirely_is_excluded_from_history():
    node = make_normalize_node(_services())
    # Well beyond even the widened lookback window (24h before period_end,
    # and period_start is already within that span) -- 3 days before period_end.
    ancient = _article("https://jpmorganchase.com/ancient", published_at=PERIOD_END - timedelta(days=3))

    result = node(_state(ancient))

    assert result["articles"] == []
    assert result["metrics"]["articles_excluded_out_of_lookback"] == 1


# ---------------------------------------------------------------------------
# Diagnostic fix (forensic finding, CURRENT run `e818e08adc2e4dd1`):
# `articles_excluded_out_of_lookback` alone cannot distinguish "had no
# usable publication date at all" (never excluded here, by `_filter_by_
# period`'s own design) from "had a real, known publication_date that
# still landed outside the lookback window" -- exactly the invisible
# failure mode a date-precision bug produced for six real PR Newswire
# releases. `articles_excluded_out_of_lookback_with_known_date` makes that
# second population separately countable.
# ---------------------------------------------------------------------------


def test_excluded_with_known_date_counter_is_zero_when_nothing_dated_is_excluded():
    node = make_normalize_node(_services())
    inside = _article("https://jpmorganchase.com/a", published_at=PERIOD_START + timedelta(hours=1))

    result = node(_state(inside))

    assert result["metrics"]["articles_excluded_out_of_lookback_with_known_date"] == 0


def test_excluded_with_known_date_counter_counts_a_dated_article_excluded_for_being_too_old():
    node = make_normalize_node(_services())
    ancient = _article("https://jpmorganchase.com/ancient", published_at=PERIOD_END - timedelta(days=3))

    result = node(_state(ancient))

    assert result["metrics"]["articles_excluded_out_of_lookback"] == 1
    assert result["metrics"]["articles_excluded_out_of_lookback_with_known_date"] == 1


def test_article_exactly_at_the_lookback_boundary_is_included():
    node = make_normalize_node(_services())
    lookback_start = PERIOD_END - timedelta(hours=LOOKBACK_HOURS)
    # period_start (08:00) is earlier than lookback_start here (20:00 - 24h
    # = previous day 20:00 < 08:00), so the effective lower bound is
    # min(period_start, lookback_start) = lookback_start itself.
    at_boundary = _article("https://jpmorganchase.com/boundary", published_at=lookback_start)

    result = node(_state(at_boundary))

    assert len(result["articles"]) == 1


def test_missing_publication_date_is_retained_explicitly_not_dropped():
    node = make_normalize_node(_services())
    dateless = _article("https://jpmorganchase.com/dateless", published_at=None)

    result = node(_state(dateless))

    assert len(result["articles"]) == 1
    assert result["articles"][0].publication_date is None
    assert result["metrics"]["articles_excluded_out_of_lookback"] == 0


def test_retrieved_at_never_silently_substitutes_for_a_missing_publication_date():
    """The dateless article's `retrieved_at` will naturally fall inside the
    window (it's normalized "now"), but `publication_date` must stay None --
    proving inclusion here is due to the explicit "missing date is kept"
    rule, not retrieved_at being used as a stand-in publication time."""
    node = make_normalize_node(_services())
    dateless = _article("https://jpmorganchase.com/dateless2", published_at=None)

    result = node(_state(dateless))

    article = result["articles"][0]
    assert article.publication_date is None
    assert article.retrieved_at is not None
    assert article.publication_date != article.retrieved_at


def test_mixed_batch_retains_history_window_and_dateless_but_not_ancient_or_future():
    node = make_normalize_node(_services())
    in_narrow_period = _article("https://jpmorganchase.com/in", published_at=PERIOD_START + timedelta(hours=2))
    in_lookback_only = _article(
        "https://jpmorganchase.com/history-only", published_at=PERIOD_START - timedelta(hours=3)
    )
    ancient = _article("https://jpmorganchase.com/old", published_at=PERIOD_END - timedelta(days=3))
    future = _article("https://jpmorganchase.com/future", published_at=PERIOD_END + timedelta(days=1))
    dateless = _article("https://jpmorganchase.com/nodate", published_at=None)

    result = node(_state(in_narrow_period, in_lookback_only, ancient, future, dateless))

    kept_urls = {a.url for a in result["articles"]}
    assert kept_urls == {
        "https://jpmorganchase.com/in", "https://jpmorganchase.com/history-only", "https://jpmorganchase.com/nodate",
    }
    assert result["metrics"]["articles_excluded_out_of_lookback"] == 2


def test_no_period_configured_is_a_no_op_and_keeps_every_article():
    """Backward compatibility: a run with no period bounds at all (both
    None, as in every pre-Phase-8 test) must not filter anything."""
    node = make_normalize_node(_services())
    old = _article("https://jpmorganchase.com/anytime", published_at=datetime(2020, 1, 1, tzinfo=timezone.utc))

    result = node(_state(old, period_start=None, period_end=None))

    assert len(result["articles"]) == 1
    assert result["metrics"]["articles_excluded_out_of_lookback"] == 0


def test_filter_by_period_helper_directly_old_article_discovered_today_is_excluded():
    """The exact regression this fixes: an old article must not be included
    just because it was *discovered* (retrieved) today. `_filter_by_period`
    itself is unchanged -- only its callers' bounds differ now."""
    old_article_retrieved_today = NormalizationService().normalize(
        _article("https://jpmorganchase.com/stale", published_at=datetime(2024, 1, 1, tzinfo=timezone.utc)),
        source=jpmorgan_source(),
    )
    assert old_article_retrieved_today.retrieved_at > PERIOD_START  # "discovered" within/near the window

    kept, excluded = _filter_by_period([old_article_retrieved_today], PERIOD_START, PERIOD_END)

    assert kept == []
    assert len(excluded) == 1
