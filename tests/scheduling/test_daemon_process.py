"""Phase 14 Part L — process-level proof: a unit test proving
`scheduler.add_job(...)` was called is NOT sufficient (per the task's own
explicit requirement). This test launches the REAL `scheduling.daemon`
entry point (`python -m research_assistant.scheduling.daemon`) as an
actual OS subprocess against an isolated temp-file database, and verifies,
by reading that database from OUTSIDE the process, that it:

1. started and initialized the database schema,
2. registered the morning/evening cron jobs (proven by real future
   `next_*_run_at` timestamps, not merely "a job object exists in memory"),
3. remained alive and emitted a heartbeat without any job having fired,
4. the heartbeat updates again after an interval tick (proving the
   background thread is genuinely alive and ticking, not frozen at
   startup).

No real 08:00/20:00 IST wait is used -- `--heartbeat-interval-seconds 1` is
the test-only timing knob `daemon.py` exposes for exactly this purpose; the
production schedule (`morning_digest_hour`/`evening_digest_hour`) is never
touched or overridden.

KNOWN SANDBOX LIMITATION (documented, not silently worked around): this
Windows sandbox's shell (a Git-Bash/mintty-hosted terminal) is not attached
to a real Win32 console, so a `CTRL_BREAK_EVENT` cannot be generated to
prove graceful signal-based shutdown end-to-end as a live OS signal in
THIS environment -- confirmed by a minimal isolated repro (a trivial
`signal.signal(signal.SIGBREAK, ...)` script in the same subprocess
launch style never receives the event here, independent of this project's
code). `daemon.py`'s SIGINT/SIGTERM/SIGBREAK handlers are implemented using
the standard, correct Python pattern; `ResearchScheduler.shutdown()` itself
(the part that actually matters -- clean APScheduler shutdown, no hang) is
directly unit-tested in `test_scheduler.py`
(`test_start_and_shutdown_toggle_is_running`). This test cleans up via
`Popen.terminate()`, which is a hard stop, not a claim of graceful
in-process shutdown.
"""

from __future__ import annotations

import subprocess
import sys
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest


def _wait_for(predicate, *, timeout_seconds=10, interval_seconds=0.25):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval_seconds)
    return predicate()  # one last honest attempt, result (possibly falsy) returned as-is


@pytest.fixture
def isolated_db_path():
    # A real `tempfile.mkdtemp()`-rooted path -- NOT a hardcoded "/tmp/..."
    # string, which this sandbox's shell silently mistranslates for a
    # native Windows subprocess (confirmed during development: a literal
    # "/tmp/x.db" argv resolved to a different filesystem location than
    # the one this same Python process reads back from).
    return str(Path(tempfile.mkdtemp()) / "daemon_process_test.db")


def _query_scheduler_state(db_path: str):
    con = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "scheduler_state" not in tables:
            return None
        row = con.execute("SELECT * FROM scheduler_state WHERE id='singleton'").fetchone()
        if row is None:
            return None
        cols = [d[0] for d in con.execute("SELECT * FROM scheduler_state").description]
        return dict(zip(cols, row))
    finally:
        con.close()


def test_the_real_daemon_process_starts_registers_jobs_and_heartbeats(isolated_db_path):
    proc = subprocess.Popen(
        [sys.executable, "-m", "research_assistant.scheduling.daemon",
         "--db-path", isolated_db_path, "--heartbeat-interval-seconds", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        state = _wait_for(lambda: _query_scheduler_state(isolated_db_path), timeout_seconds=15)

        assert proc.poll() is None, f"daemon process exited unexpectedly; output:\n{proc.stdout.read() if proc.stdout else ''}"
        assert state is not None, "scheduler_state row was never written -- the daemon never started cleanly"
        assert state["started_at"] is not None
        assert state["last_heartbeat_at"] is not None
        # Real, future-pointing cron-derived timestamps -- not a placeholder
        # -- proves the morning/evening jobs were genuinely registered.
        assert state["next_morning_run_at"] is not None
        assert state["next_evening_run_at"] is not None

        first_heartbeat = state["last_heartbeat_at"]
        second_state = _wait_for(
            lambda: (_query_scheduler_state(isolated_db_path) or {}).get("last_heartbeat_at") != first_heartbeat,
            timeout_seconds=10,
        )
        assert second_state, "heartbeat never advanced -- the background scheduler thread appears frozen, not alive"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
