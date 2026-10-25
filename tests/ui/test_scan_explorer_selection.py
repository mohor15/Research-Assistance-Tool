"""Scan Explorer default-run-selection refresh fix (forensic finding, run
`b9eb35cb5362b752`): `st.selectbox`'s `key`-bound session state previously
persisted the user's selection across every rerun, so a newly-completed
CURRENT run appearing at the top of the (freshly-requeried) run list never
became the default selection -- the stale, previously-selected older run
stayed "selected" instead.

Uses Streamlit's own `AppTest` harness against a tiny target script
(`tests/ui/fixtures/scan_explorer_selection_target.py`) with
`dd.get_current_run_options`/`dd.get_current_scan_explorer`/
`dd.get_run_diagnostic_messages` monkeypatched to hand-built, scenario-
controlled data -- never a real `PersistenceService`/`data/research.db`,
never Ollama, never the network. `AppTest` preserves `st.session_state`
across successive `.run()` calls on the same instance, exactly like a real
browser session issuing one rerun after another, so this proves the real
cross-rerun behavior, not just a single render."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

_TARGET = str(Path(__file__).parent / "fixtures" / "scan_explorer_selection_target.py")


def _selected_run_id(at) -> str:
    # The selectbox label is "{period} — {status} ({run_id})".
    value = at.selectbox(key="scan_explorer_run").value
    return value.split("(")[-1].rstrip(")")


def test_the_first_available_run_is_selected_by_default():
    at = AppTest.from_file(_TARGET, default_timeout=30)
    at.query_params["scenario"] = "A"
    at.run()

    assert at.exception == []
    assert _selected_run_id(at) == "RUN_OLD"  # newest of ["RUN_OLD", "RUN_OLDER"]


def test_selection_remains_stable_across_an_ordinary_rerun_when_newest_is_unchanged():
    at = AppTest.from_file(_TARGET, default_timeout=30)
    at.query_params["scenario"] = "A"
    at.run()
    assert _selected_run_id(at) == "RUN_OLD"

    # Plain rerun, same run list (same newest) -- selection must not move.
    at.query_params["scenario"] = "B"
    at.run()

    assert at.exception == []
    assert _selected_run_id(at) == "RUN_OLD"


def test_user_can_still_manually_select_an_older_run():
    at = AppTest.from_file(_TARGET, default_timeout=30)
    at.query_params["scenario"] = "A"
    at.run()

    at.selectbox(key="scan_explorer_run").set_value(
        [v for v in at.selectbox(key="scan_explorer_run").options if "RUN_OLDER" in v][0]
    )
    at.run()

    assert at.exception == []
    assert _selected_run_id(at) == "RUN_OLDER"


def test_manual_older_selection_survives_a_rerun_when_newest_is_still_unchanged():
    at = AppTest.from_file(_TARGET, default_timeout=30)
    at.query_params["scenario"] = "A"
    at.run()
    at.selectbox(key="scan_explorer_run").set_value(
        [v for v in at.selectbox(key="scan_explorer_run").options if "RUN_OLDER" in v][0]
    )
    at.run()
    assert _selected_run_id(at) == "RUN_OLDER"

    # Scenario B: same run list, same newest ("RUN_OLD") -- the user's
    # deliberate older selection must NOT be disturbed.
    at.query_params["scenario"] = "B"
    at.run()

    assert at.exception == []
    assert _selected_run_id(at) == "RUN_OLDER"


def test_a_newly_appearing_latest_run_becomes_the_default_selection():
    at = AppTest.from_file(_TARGET, default_timeout=30)
    at.query_params["scenario"] = "A"
    at.run()
    assert _selected_run_id(at) == "RUN_OLD"

    # Scenario C: "RUN_NEW" is now the newest run -- the default selection
    # must refresh to it, even though the user never touched the widget.
    at.query_params["scenario"] = "C"
    at.run()

    assert at.exception == []
    assert _selected_run_id(at) == "RUN_NEW"


def test_newest_run_refresh_does_not_repeat_on_a_subsequent_unchanged_rerun():
    at = AppTest.from_file(_TARGET, default_timeout=30)
    at.query_params["scenario"] = "A"
    at.run()
    at.query_params["scenario"] = "C"
    at.run()
    assert _selected_run_id(at) == "RUN_NEW"

    # The user deliberately picks an older run after the refresh...
    at.selectbox(key="scan_explorer_run").set_value(
        [v for v in at.selectbox(key="scan_explorer_run").options if "RUN_OLD)" in v][0]
    )
    at.run()
    assert _selected_run_id(at) == "RUN_OLD"

    # ...and an ordinary rerun with the SAME newest run ("RUN_NEW", scenario
    # D) must not force it back -- only an actual newest-run CHANGE resets
    # the selection.
    at.query_params["scenario"] = "D"
    at.run()

    assert at.exception == []
    assert _selected_run_id(at) == "RUN_OLD"


def test_the_selected_run_id_is_passed_to_the_existing_explorer_query():
    """The widget's selection must be exactly what reaches
    `dd.get_current_scan_explorer` -- proven here by the rendered Run ID
    line, which this test's fixture sources directly from that call's
    returned `CurrentScanExplorerData.run_id`."""
    at = AppTest.from_file(_TARGET, default_timeout=30)
    at.query_params["scenario"] = "C"
    at.run()

    assert at.exception == []
    text = "\n".join(str(el.value) for el in at.markdown)
    assert "**Run ID:** RUN_NEW" in text
