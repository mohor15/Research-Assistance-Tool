"""CURRENT-only candidate ranking/selection (Part B1-B4): a dedicated policy
that only ever applies when `state.digest_type == DigestType.CURRENT`,
never to Morning/Evening. Exercises the two pure helpers directly rather
than the full LangGraph wiring (covered end-to-end by the existing
tests/pipeline/test_graph.py and test_retrieve_zero_candidate_diagnostic.py
style tests, which prove Morning/Evening are unaffected)."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.models.article import DatePrecision
from research_assistant.pipeline.nodes import (
    _current_candidate_rank_key,
    _select_with_source_diversity,
)


class _FakeArticle:
    def __init__(self, *, title, publication_date, precision, source_name="Source"):
        self.title = title
        self.publication_date = publication_date
        self.publication_date_precision = precision
        self.source_name = source_name


class _FakeEvidence:
    def __init__(self, content=""):
        self.content = content


_NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
_SUBSTANTIVE_BODY = "x" * 500 + " today announced a major enterprise AI partnership with a bank."


def test_exact_date_candidate_outranks_month_precision_candidate():
    exact = (
        _FakeArticle(title="Bank announces AI deal", publication_date=_NOW, precision=DatePrecision.EXACT),
        _FakeEvidence(_SUBSTANTIVE_BODY),
    )
    month = (
        _FakeArticle(title="Bank announces AI deal", publication_date=_NOW, precision=DatePrecision.MONTH),
        _FakeEvidence(_SUBSTANTIVE_BODY),
    )
    ranked = sorted([month, exact], key=lambda p: _current_candidate_rank_key(p[0], p[1], [], []), reverse=True)
    assert ranked[0] is exact


def test_month_precision_outranks_missing_date():
    month = (
        _FakeArticle(title="Bank announces AI deal", publication_date=_NOW, precision=DatePrecision.MONTH),
        _FakeEvidence(_SUBSTANTIVE_BODY),
    )
    missing = (
        _FakeArticle(title="Bank announces AI deal", publication_date=None, precision=DatePrecision.UNKNOWN),
        _FakeEvidence(_SUBSTANTIVE_BODY),
    )
    ranked = sorted([missing, month], key=lambda p: _current_candidate_rank_key(p[0], p[1], [], []), reverse=True)
    assert ranked[0] is month


def test_date_only_is_treated_as_equally_trustworthy_as_exact():
    exact = (
        _FakeArticle(title="Bank announces AI deal", publication_date=_NOW, precision=DatePrecision.EXACT),
        _FakeEvidence(_SUBSTANTIVE_BODY),
    )
    date_only = (
        _FakeArticle(title="Bank announces AI deal", publication_date=_NOW, precision=DatePrecision.DATE_ONLY),
        _FakeEvidence(_SUBSTANTIVE_BODY),
    )
    assert (
        _current_candidate_rank_key(exact[0], exact[1], [], [])[0]
        == _current_candidate_rank_key(date_only[0], date_only[1], [], [])[0]
    )


def _candidate(source_name: str, rank_in_source: int):
    article = _FakeArticle(
        title=f"{source_name} story {rank_in_source}", publication_date=_NOW,
        precision=DatePrecision.EXACT, source_name=source_name,
    )
    return (article, _FakeEvidence(_SUBSTANTIVE_BODY))


def test_source_diversity_gives_every_source_its_best_pick_before_any_seconds():
    """One dominant source (Reuters, 5 candidates) must not exhaust the
    whole budget while two other sources (each with 1 candidate) go
    entirely unrepresented."""
    ranked = (
        [_candidate("Reuters", i) for i in range(5)]
        + [_candidate("FT", 0)]
        + [_candidate("CNBC", 0)]
    )
    selected = _select_with_source_diversity(ranked, top_k=3)
    sources_selected = {pair[0].source_name for pair in selected}
    assert sources_selected == {"Reuters", "FT", "CNBC"}


def test_source_diversity_falls_back_to_plain_top_k_with_a_single_source():
    ranked = [_candidate("Reuters", i) for i in range(5)]
    selected = _select_with_source_diversity(ranked, top_k=3)
    assert selected == ranked[:3]


def test_source_diversity_gives_a_second_pick_once_every_source_has_a_first():
    ranked = (
        [_candidate("Reuters", 0), _candidate("Reuters", 1)]
        + [_candidate("FT", 0)]
    )
    selected = _select_with_source_diversity(ranked, top_k=3)
    # Round 0: Reuters#0, FT#0 (2 picks); round 1: Reuters#1 (budget reached)
    assert selected == [ranked[0], ranked[2], ranked[1]]


def test_source_diversity_never_exceeds_top_k_even_with_many_sources():
    ranked = [_candidate(f"Source{i}", 0) for i in range(10)]
    selected = _select_with_source_diversity(ranked, top_k=4)
    assert len(selected) == 4
