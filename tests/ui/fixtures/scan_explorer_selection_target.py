"""AppTest target for `tests/ui/test_scan_explorer_selection.py` -- renders
`_render_current_scan_explorer` directly, with `dd.get_current_run_options`/
`dd.get_current_scan_explorer` monkeypatched to return hand-built, scenario-
controlled data -- never touching a real `PersistenceService`/`data/research.db`,
and never calling Ollama or the network.

`st.query_params["scenario"]` selects which canned run-option list this
render sees, simulating successive reruns of the SAME Streamlit session
(AppTest preserves `st.session_state` across `.run()` calls, exactly like a
real browser session issuing one rerun after another)."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from research_assistant.config import get_config
from research_assistant.ui import app as app_module
from research_assistant.ui import dashboard_data as dd

D = datetime(2026, 10, 12, tzinfo=timezone.utc)


def _run_option(run_id: str) -> dd.RunOption:
    return dd.RunOption(run_id=run_id, started_at=D, status="partial_success")


def _scan_data(run_id: str) -> dd.CurrentScanExplorerData:
    return dd.CurrentScanExplorerData(
        run_id=run_id, status="partial_success", started_at=D, finished_at=D,
        period_start=D, period_end=D, metrics={}, articles=[], events=[],
        coverage_evidence=0, coverage_analyzed=0, coverage_rejected_by_agent=0,
        coverage_not_analyzed=0, coverage_content_unavailable=0, coverage_relevant=0,
        coverage_irrelevant=0, coverage_ambiguous=0, reconstruction_note="test fixture",
    )


# Scenario A: only "RUN_OLD"/"RUN_OLDER" exist -- newest is RUN_OLD.
# Scenario B: unchanged from A (same newest) -- a plain rerun.
# Scenario C: "RUN_NEW" has appeared as the new newest run.
# Scenario D: unchanged from C (same newest, RUN_NEW) -- a plain rerun.
_SCENARIOS = {
    "A": ["RUN_OLD", "RUN_OLDER"],
    "B": ["RUN_OLD", "RUN_OLDER"],
    "C": ["RUN_NEW", "RUN_OLD", "RUN_OLDER"],
    "D": ["RUN_NEW", "RUN_OLD", "RUN_OLDER"],
}

scenario = st.query_params.get("scenario", "A")
run_ids = _SCENARIOS[scenario]

app_module.dd.get_current_run_options = lambda service, limit=20: [_run_option(rid) for rid in run_ids]
app_module.dd.get_current_scan_explorer = lambda service, config, run_id: _scan_data(run_id)
app_module.dd.get_run_diagnostic_messages = lambda service, run_id, contains=None: []

app_module._render_current_scan_explorer(service=None, config=get_config())
