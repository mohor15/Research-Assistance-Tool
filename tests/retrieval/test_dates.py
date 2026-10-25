"""`parse_date`'s Unix-epoch-seconds recognition (Phase 14 forensic fix):
needed to corroborate a real, observed, non-standard metadata field (a
JPMorganChase page's `<meta name="alg-search-date" content="1790697480"/>`
-- an Algolia-style search-indexing field, not a named date tag) against
trafilatura's own date guess. Narrowly guarded to a plausible calendar-year
range so an unrelated bare integer (a count, an ID) is never misread as a
date."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.retrieval.dates import parse_date


def test_a_real_observed_epoch_value_parses_to_the_correct_date():
    # 1790697480 -- the exact value from the real JPMorganChase
    # alg-search-date tag that corroborates its correct 2026-09-29 date.
    result = parse_date("1790697480")
    assert result == datetime(2026, 9, 29, 15, 58, 0, tzinfo=timezone.utc)


def test_epoch_value_with_surrounding_whitespace_still_parses():
    result = parse_date("  1790697480  ")
    assert result is not None
    assert result.year == 2026


def test_a_small_bare_integer_outside_the_epoch_range_is_not_a_date():
    assert parse_date("42") is None


def test_a_huge_bare_integer_outside_the_epoch_range_is_not_a_date():
    assert parse_date("99999999999999") is None


def test_year_2000_boundary_is_accepted():
    result = parse_date("946684800")  # exactly 2000-01-01T00:00:00Z
    assert result == datetime(2000, 1, 1, tzinfo=timezone.utc)


def test_normal_iso_date_strings_are_unaffected():
    result = parse_date("2025-09-01T09:00:00Z")
    assert result == datetime(2025, 9, 1, 9, 0, 0, tzinfo=timezone.utc)


def test_empty_and_none_values_remain_unparseable():
    assert parse_date(None) is None
    assert parse_date("") is None
