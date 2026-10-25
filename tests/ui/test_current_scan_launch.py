"""Connection-reliability fix: the admin "Run Live Current Scan now"
button launches the scan as a detached subprocess (the existing
`python -m research_assistant.scheduling.cli --digest-type current` entry
point) instead of blocking the Streamlit script synchronously for the
scan's entire duration. These tests exercise the real button-handling
logic in `_render_current_page` via Streamlit's own `AppTest` harness
against a REAL temp-file database (so `current_run_lock`/`pipeline_runs`
reads are genuine, not mocked) -- `_launch_current_scan_subprocess` is
monkeypatched to a fake that records a call without ever spawning a real
process, calling Ollama, or touching the network.

Audit follow-up (stale-lock UI deadlock, Popen-failure handling, and
child-startup-failure detection): this file also covers the three
corrections from the read-only pre-flight audit -- see the fixture
(`fixtures/current_scan_launch_target.py`) for the `launch_mode`/
`poll_result_path` query params that drive the fake launch's behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from streamlit.testing.v1 import AppTest

from research_assistant.storage.repositories.current_run_lock import CurrentRunLockRepository

_TARGET = str(Path(__file__).parent / "fixtures" / "current_scan_launch_target.py")


def _run_app(database, tmp_path, *, launch_log_name="launches.log", launch_mode="success", poll_result=None):
    launch_log_path = tmp_path / launch_log_name
    poll_result_path = tmp_path / "poll_result.txt"
    if poll_result is not None:
        poll_result_path.write_text(str(poll_result), encoding="utf-8")
    at = AppTest.from_file(_TARGET, default_timeout=30)
    at.query_params["db_path"] = str(database.path)
    at.query_params["launch_log_path"] = str(launch_log_path)
    at.query_params["launch_mode"] = launch_mode
    at.query_params["poll_result_path"] = str(poll_result_path)
    at.run()
    return at, launch_log_path, poll_result_path


def _launch_count(launch_log_path: Path) -> int:
    if not launch_log_path.exists():
        return 0
    return launch_log_path.read_text(encoding="utf-8").count("launch\n")


def _buttons(at):
    return [b for b in at.button if b.label == "Run Live Current Scan now"]


# ---------------------------------------------------------------------------
# Baseline launch behavior (pre-existing, re-verified unchanged)
# ---------------------------------------------------------------------------


def test_run_button_launches_the_scan_without_blocking(persistence_service, tmp_path):
    at, log_path, _ = _run_app(persistence_service.database, tmp_path, poll_result="none")
    buttons = _buttons(at)
    assert len(buttons) == 1
    buttons[0].click().run()

    assert at.exception == []
    assert _launch_count(log_path) == 1


def test_a_streamlit_rerun_does_not_launch_a_duplicate_scan(persistence_service, tmp_path):
    at, log_path, _ = _run_app(persistence_service.database, tmp_path, poll_result="none")
    _buttons(at)[0].click().run()
    assert _launch_count(log_path) == 1

    at.run()  # an unrelated rerun

    assert _launch_count(log_path) == 1


def test_rapid_double_click_within_one_session_is_suppressed(persistence_service, tmp_path):
    at, log_path, _ = _run_app(persistence_service.database, tmp_path, poll_result="none")
    _buttons(at)[0].click().run()
    assert _launch_count(log_path) == 1

    assert _buttons(at)[0].disabled is True
    at.run()
    assert _launch_count(log_path) == 1


def test_existing_admin_current_run_summary_still_renders(persistence_service, tmp_path):
    at, _log_path, _ = _run_app(persistence_service.database, tmp_path)
    assert at.exception == []
    assert any("No CURRENT scan has ever been run" in str(el.value) for el in at.info)


def test_current_page_caption_states_the_configured_window_width_not_a_stale_24_hours(persistence_service, tmp_path):
    """The page's own wording must reflect
    `config.settings.research.current_window_hours` (48 by default, widened
    from the original fixed 24 per management's requirement) rather than a
    hardcoded, now-incorrect "24-hour" claim."""
    at, _log_path, _ = _run_app(persistence_service.database, tmp_path)
    assert at.exception == []
    caption_text = "\n".join(str(el.value) for el in at.caption)
    assert "48-hour" in caption_text
    assert "24-hour" not in caption_text


def test_a_genuine_worker_failure_is_visible_and_accurately_reported(persistence_service, tmp_path):
    """A worker (subprocess) that fails is recorded by the EXISTING,
    unchanged `PipelineRunner._run_locked` exception handling as a
    `FAILED` run with diagnostics -- never misreported as "Streamlit has
    stopped"."""
    from datetime import timedelta

    from research_assistant.models.digest import DigestType
    from research_assistant.models.enums import DigestPersistenceStatus
    from research_assistant.pipeline.runner import PipelineDiagnostic, PipelineRunResult, RunStatus

    now = datetime.now(timezone.utc)
    run = PipelineRunResult(
        run_id="failed_run", digest_type=DigestType.CURRENT, started_at=now - timedelta(minutes=5), finished_at=now,
        status=RunStatus.FAILED,
        diagnostics=[PipelineDiagnostic(stage="pipeline_run", message="Could not reach Ollama", severity="error")],
        counts={}, period_start=now - timedelta(hours=24), period_end=now,
        digest_persistence_status=DigestPersistenceStatus.NOT_ATTEMPTED,
    )
    persistence_service.persist_run(run)

    at, _log_path, _ = _run_app(persistence_service.database, tmp_path)

    assert at.exception == []
    page_text = "\n".join(str(el.value) for el in at.markdown)
    assert "failed" in page_text
    assert _buttons(at)[0].disabled is False


# ---------------------------------------------------------------------------
# A. Recoverable stale-lock UI: live / dead / indeterminate owner
# ---------------------------------------------------------------------------


def test_lock_held_by_a_live_process_disables_the_button(persistence_service, tmp_path):
    """A genuinely active run (owner pid is THIS test process, definitely
    alive) must still fully block the button -- mutual exclusion is
    never weakened."""
    import os

    conn = persistence_service.database.connect()
    CurrentRunLockRepository().try_acquire(
        conn, run_id="run_in_flight", started_at=datetime.now(timezone.utc), owner_pid=os.getpid(),
    )
    conn.close()

    at, log_path, _ = _run_app(persistence_service.database, tmp_path)

    assert at.exception == []
    assert any("already running" in str(el.value) for el in at.warning)
    assert _buttons(at)[0].disabled is True
    assert _launch_count(log_path) == 0


def test_lock_held_by_a_dead_process_does_not_permanently_disable_the_button(persistence_service, tmp_path):
    """The audit's core finding: a stale lock (confirmed-dead owner) must
    NOT make the button permanently unclickable -- the user must be able
    to click it, which lets the EXISTING `try_acquire` reclaim logic run.
    The lock row itself is never touched by this test or by the UI."""
    conn = persistence_service.database.connect()
    CurrentRunLockRepository().try_acquire(
        conn, run_id="abandoned_run", started_at=datetime.now(timezone.utc), owner_pid=-1,
    )
    conn.close()

    at, log_path, _ = _run_app(persistence_service.database, tmp_path, poll_result="none")

    assert at.exception == []
    assert any("abandoned_run" in str(el.value) and "stopped without releasing" in str(el.value) for el in at.warning)
    assert _buttons(at)[0].disabled is False

    # Clicking now actually attempts a new launch -- the UI never deletes
    # the lock row itself; it only stops blocking the attempt that would
    # let `try_acquire` do the real reclaiming.
    _buttons(at)[0].click().run()
    assert _launch_count(log_path) == 1


def test_lock_with_indeterminate_liveness_is_treated_as_held(persistence_service, tmp_path):
    """`owner_pid IS NULL` (a lock row predating the column) cannot be
    verified either way -- per the task's own rule, this must be treated
    as held, never as reclaimable, exactly like `try_acquire`'s own
    `existing_pid is None` special case."""
    conn = persistence_service.database.connect()
    conn.execute(
        "INSERT INTO current_run_lock (id, run_id, started_at, owner_pid) VALUES ('singleton', 'null_owner_run', ?, NULL)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()
    conn.close()

    at, log_path, _ = _run_app(persistence_service.database, tmp_path)

    assert at.exception == []
    assert any("already running" in str(el.value) and "could not be verified" in str(el.value) for el in at.warning)
    assert _buttons(at)[0].disabled is True
    assert _launch_count(log_path) == 0


# ---------------------------------------------------------------------------
# B. Popen failure handling
# ---------------------------------------------------------------------------


def test_popen_failure_shows_a_clean_error_and_never_claims_the_scan_started(persistence_service, tmp_path):
    at, log_path, _ = _run_app(persistence_service.database, tmp_path, launch_mode="fail")

    assert at.exception == []  # the exception is CAUGHT, never an uncaught Streamlit traceback
    _buttons(at)[0].click().run()

    assert at.exception == []
    assert any("Failed to launch the CURRENT scan" in str(el.value) for el in at.error)
    assert not any("launched in the background" in str(el.value) for el in at.success)
    # No lock was ever created and the next click is still possible (not
    # permanently disabled by a launch that never actually happened).
    conn = persistence_service.database.connect()
    lock = CurrentRunLockRepository().get(conn)
    conn.close()
    assert lock is None
    assert _buttons(at)[0].disabled is False


def test_successful_launch_still_shows_the_success_message(persistence_service, tmp_path):
    at, log_path, _ = _run_app(persistence_service.database, tmp_path, launch_mode="success", poll_result="none")
    _buttons(at)[0].click().run()

    assert at.exception == []
    assert _launch_count(log_path) == 1


# ---------------------------------------------------------------------------
# C. Child startup failure before lock acquisition
# ---------------------------------------------------------------------------


def test_child_still_initializing_is_distinguished_from_failure(persistence_service, tmp_path):
    """While the real OS child process is still running (`poll()` returns
    `None`) but has not yet acquired the lock, the UI must show
    "starting up," never a failure -- this is a real OS fact, not a
    timeout guess."""
    at, log_path, poll_result_path = _run_app(persistence_service.database, tmp_path, poll_result="none")
    _buttons(at)[0].click().run()
    assert _launch_count(log_path) == 1

    # A later rerun (e.g. the user refreshing) while the child is STILL
    # running and still hasn't taken the lock.
    at.run()

    assert at.exception == []
    assert any("starting up" in str(el.value) for el in at.info)
    assert not any("exited immediately" in str(el.value) for el in at.error)
    assert _buttons(at)[0].disabled is True  # suppresses a redundant second launch from this session


def test_child_startup_failure_is_detected_from_its_real_exit_code(persistence_service, tmp_path):
    """Once the real child process has actually exited with a non-zero
    code, and no lock ever appeared, that is reported as a genuine
    startup failure -- a durable OS fact (the exit code), never a guess
    from the mere presence of a log file."""
    at, log_path, poll_result_path = _run_app(persistence_service.database, tmp_path, poll_result="none")
    _buttons(at)[0].click().run()
    assert _launch_count(log_path) == 1

    # The real child process has now actually exited with a failure code.
    poll_result_path.write_text("1", encoding="utf-8")
    at.run()

    assert at.exception == []
    assert any("exited immediately with code 1" in str(el.value) for el in at.error)
    # (The button may still be disabled for the remainder of the unrelated,
    # separate short launch-cooldown window -- that is not permanent, and
    # is exercised independently by the cooldown tests above.)


def test_normal_lock_takeover_clears_the_pending_startup_tracking(persistence_service, tmp_path):
    """Once the child actually acquires the real lock, the lock/run
    status becomes the whole story again -- the pending-launch tracking
    is cleared, never fighting with or duplicating the authoritative
    status display."""
    at, log_path, poll_result_path = _run_app(persistence_service.database, tmp_path, poll_result="none")
    _buttons(at)[0].click().run()
    assert _launch_count(log_path) == 1

    # The child has now (really) acquired the lock -- using a genuinely
    # alive pid (this test process's own) so the real liveness probe
    # correctly reports it as active, not stale.
    import os

    conn = persistence_service.database.connect()
    CurrentRunLockRepository().try_acquire(conn, run_id="real_run", started_at=datetime.now(timezone.utc), owner_pid=os.getpid())
    conn.close()

    at.run()

    assert at.exception == []
    assert any("already running" in str(el.value) and "real_run" in str(el.value) for el in at.warning)
    assert not any("starting up" in str(el.value) for el in at.info)
    assert not any("exited immediately" in str(el.value) for el in at.error)


# ---------------------------------------------------------------------------
# D. Completion notification (watch while a lock is held, toast once released)
# ---------------------------------------------------------------------------


def test_an_active_lock_starts_watching_and_embeds_an_auto_refresh_tag(persistence_service, tmp_path):
    import os

    conn = persistence_service.database.connect()
    CurrentRunLockRepository().try_acquire(
        conn, run_id="run_in_flight", started_at=datetime.now(timezone.utc), owner_pid=os.getpid(),
    )
    conn.close()

    at, _log_path, _ = _run_app(persistence_service.database, tmp_path)

    assert at.exception == []
    assert at.session_state["current_scan_watching"] is True
    markdown_text = "\n".join(str(el.value) for el in at.markdown)
    assert 'http-equiv="refresh"' in markdown_text
    assert at.toast == []  # still running -- no completion toast yet


def test_a_released_lock_fires_a_one_shot_completion_toast(persistence_service, tmp_path):
    import os

    conn = persistence_service.database.connect()
    CurrentRunLockRepository().try_acquire(
        conn, run_id="run_in_flight", started_at=datetime.now(timezone.utc), owner_pid=os.getpid(),
    )
    conn.close()

    at, _log_path, _ = _run_app(persistence_service.database, tmp_path)
    assert at.session_state["current_scan_watching"] is True

    conn = persistence_service.database.connect()
    CurrentRunLockRepository().release(conn, run_id="run_in_flight")
    conn.close()

    at.run()

    assert at.exception == []
    assert any("CURRENT scan finished" in str(t.value) for t in at.toast)
    assert at.session_state.get("current_scan_watching") is not True
    markdown_text = "\n".join(str(el.value) for el in at.markdown)
    assert 'http-equiv="refresh"' not in markdown_text  # no longer watching -- stops auto-refreshing

    # The toast is one-shot -- a further unrelated rerun must not re-fire it.
    at.run()
    assert at.toast == []


def test_never_having_watched_never_shows_a_completion_toast(persistence_service, tmp_path):
    at, _log_path, _ = _run_app(persistence_service.database, tmp_path)

    assert at.exception == []
    assert at.toast == []
    markdown_text = "\n".join(str(el.value) for el in at.markdown)
    assert 'http-equiv="refresh"' not in markdown_text


def test_lock_acquisition_delayed_but_eventually_successful_is_never_misclassified_as_failed(persistence_service, tmp_path):
    """No fixed timeout anywhere in this logic -- an arbitrarily slow (but
    still running) child must never be misreported as failed just
    because the lock took a while to appear."""
    at, log_path, poll_result_path = _run_app(persistence_service.database, tmp_path, poll_result="none")
    _buttons(at)[0].click().run()

    for _ in range(5):  # several reruns, still running, still no lock
        at.run()
        assert not any("exited immediately" in str(el.value) for el in at.error)

    # It finally takes the lock -- success, not a delayed "failure".
    import os

    conn = persistence_service.database.connect()
    CurrentRunLockRepository().try_acquire(
        conn, run_id="finally_real_run", started_at=datetime.now(timezone.utc), owner_pid=os.getpid(),
    )
    conn.close()
    at.run()

    assert any("finally_real_run" in str(el.value) for el in at.warning)
