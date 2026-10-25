"""Phase 14 / forensic-diagnosis Part 6-7 fix: the entity monitoring pages'
article inventory must never use wording that can be read as "the current
digest's operational period" -- it is a rolling historical lookback. This
is a portable (no Streamlit import required) source-text check, consistent
with this project's existing convention of not requiring Streamlit to be
installed for the test suite (see test_dashboard_data.py)."""

from __future__ import annotations

from pathlib import Path

# Read by file path, not by `import research_assistant.ui.app` -- that
# module does `import streamlit as st` at the top level, and this project's
# own convention (see test_dashboard_data.py) is that the test suite must
# never require Streamlit to be installed.
_APP_PY_PATH = Path(__file__).resolve().parents[2] / "src" / "research_assistant" / "ui" / "app.py"
_SOURCE = _APP_PY_PATH.read_text(encoding="utf-8")


def test_the_old_ambiguous_window_phrase_is_gone():
    assert "retrieved in this window" not in _SOURCE


def test_the_entity_page_explicitly_says_historical_lookback():
    assert "historical lookback" in _SOURCE.lower()


def test_the_entity_page_renders_an_explicit_date_range_via_the_shared_formatter():
    # `_fmt` is the shared date-range formatter also used by the new
    # operational-status section -- both must display the actual configured
    # range, never a hardcoded date.
    assert "_fmt(since, config)" in _SOURCE
    assert "_fmt(now, config)" in _SOURCE


def test_the_entity_card_caption_distinguishes_from_the_operational_digest():
    assert "not the current Morning/Evening" in _SOURCE or "NOT limited to the current" in _SOURCE
