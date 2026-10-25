"""Admin CURRENT page / Scan Explorer timestamp display: must use the same
timezone-aware `_fmt_period()` helper the user-facing CURRENT page already
uses, instead of raw UTC `.isoformat()`. Presentation only -- storage stays
UTC, run selection/ordering is untouched (see `dashboard_data.py`)."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone

from research_assistant.config import get_config
from research_assistant.ui.app import (
    _fmt_period,
    _render_current_page,
    _render_current_pipeline_funnel,
    _render_current_scan_explorer,
)

CONFIG = get_config()


def test_fmt_period_converts_utc_to_the_configured_timezone():
    """Stored timestamps are always UTC; the display must convert to
    `config.settings.system.timezone` (Asia/Kolkata in this project's
    config) -- never shown as raw UTC."""
    utc_ts = datetime(2026, 10, 7, 16, 43, 27, tzinfo=timezone.utc)
    formatted = _fmt_period(utc_ts, CONFIG)
    # 16:43 UTC -> 22:13 IST (UTC+5:30)
    assert "10:13 PM" in formatted or "22:13" in formatted
    assert "IST" in formatted


def test_fmt_period_handles_none_without_raising():
    assert _fmt_period(None, CONFIG) == "—"


def test_admin_current_page_uses_fmt_period_not_raw_isoformat():
    """Structural check (same style as existing tests in this file's
    package): the admin CURRENT page's own source must call `_fmt_period`
    for its run timestamps, and must never call `.isoformat()` on a run's
    `finished_at`/`period_start`/`period_end` directly."""
    source = inspect.getsource(_render_current_page)
    assert "_fmt_period(last_run.finished_at" in source
    assert "_fmt_period(last_run.period_start" in source
    assert "_fmt_period(last_run.period_end" in source
    assert "last_run.finished_at.isoformat()" not in source
    assert "last_run.period_start.isoformat()" not in source
    assert "last_run.period_end.isoformat()" not in source


def test_admin_scan_explorer_uses_fmt_period_not_raw_isoformat():
    source = inspect.getsource(_render_current_scan_explorer)
    assert "_fmt_period(data.started_at" in source
    assert "_fmt_period(data.finished_at" in source
    assert "_fmt_period(data.period_start" in source
    assert "_fmt_period(data.period_end" in source
    assert "_fmt_period(r.started_at" in source
    assert "data.started_at.isoformat()" not in source
    assert "data.finished_at.isoformat()" not in source
    assert "data.period_start.isoformat()" not in source
    assert "data.period_end.isoformat()" not in source
    assert ".strftime('%Y-%m-%d %H:%M')} UTC" not in source


# ---------------------------------------------------------------------------
# Pipeline Funnel (dashboard observability task): a run that predates one
# of these counters must render "n/a", never crash on a KeyError and never
# silently show a fabricated 0.
# ---------------------------------------------------------------------------


def test_pipeline_funnel_is_wired_into_the_scan_explorer():
    source = inspect.getsource(_render_current_scan_explorer)
    assert "_render_current_pipeline_funnel(" in source


def test_pipeline_funnel_never_uses_direct_dict_indexing_on_counts():
    """Every counter must be read via the defensive `g()`/`.get()` helper,
    never `m["..."]` -- a run persisted before a given counter existed
    must render "n/a" for it, not raise a KeyError."""
    source = inspect.getsource(_render_current_pipeline_funnel)
    assert 'm["' not in source and "m['" not in source


def test_pipeline_funnel_handles_a_run_with_no_current_specific_counters_at_all():
    """A Morning/Evening-shaped (or very old) counts dict with none of the
    CURRENT-only funnel keys must not raise -- every lookup degrades to
    "n/a" instead."""
    _render_current_pipeline_funnel({"candidates_discovered": 10}, "not_attempted")
