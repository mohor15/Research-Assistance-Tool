"""Regression guard for the exact runtime import `ui/app.py`'s
"Run Live Current Scan now" button performs. This is deliberately a
literal copy of that import statement -- if either name is ever renamed,
removed, or moved to a different module, this test fails immediately
instead of only surfacing as a runtime ImportError the first time someone
clicks the button in a live Streamlit session.

Note: this test (like every other test in the suite) always imports
`research_assistant.pipeline.runner` fresh into a brand-new interpreter
process, so it can catch a genuine missing/renamed symbol in the source --
but it CANNOT catch a long-lived Streamlit server process holding a stale,
already-imported copy of this module in `sys.modules` from before a code
change. That failure mode is a process-lifecycle issue (the fix is
restarting the server), not a code defect, and no test run in a fresh
process can reproduce or detect it.
"""

from __future__ import annotations


def test_app_py_current_scan_button_import_resolves():
    from research_assistant.pipeline.runner import (
        CurrentRunAlreadyInProgressError,
        PipelineRunner,
    )

    assert issubclass(CurrentRunAlreadyInProgressError, RuntimeError)
    assert hasattr(PipelineRunner, "run")
    assert hasattr(PipelineRunner, "build")


def test_current_run_already_in_progress_error_carries_the_holder_run_id():
    from research_assistant.pipeline.runner import CurrentRunAlreadyInProgressError

    err = CurrentRunAlreadyInProgressError("some-run-id")
    assert err.run_id == "some-run-id"
    assert "some-run-id" in str(err)
