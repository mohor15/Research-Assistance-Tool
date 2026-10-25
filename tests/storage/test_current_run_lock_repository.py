"""Phase 17 stale-lock audit: `current_run_lock` must never permanently
brick CURRENT just because the process that held it died without reaching
`PipelineRunner.run()`'s `finally` release (crash, kill, machine
power-loss) -- but it must also never let a genuinely live holder be
stolen from. Repository-level only; no pipeline, no Ollama, no network."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from typing import List

import pytest

from research_assistant.storage.database import Database
from research_assistant.storage.persistence_service import PersistenceService

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def persistence_service(tmp_path: Path) -> PersistenceService:
    database = Database(tmp_path / "test.db")
    database.initialize()
    return PersistenceService(database)


def _dead_pid() -> int:
    """A real pid guaranteed to no longer be running: spawn a trivial
    child process, wait for it to exit, then return its (now-dead) pid."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = proc.pid
    proc.wait()
    return pid


# ---------------------------------------------------------------------------
# A/B: acquisition is atomic; two simultaneous callers cannot both succeed
# ---------------------------------------------------------------------------


def test_basic_acquire_and_release(persistence_service):
    conn = persistence_service.database.connect()
    try:
        assert persistence_service.current_run_lock.try_acquire(conn, run_id="r1", started_at=D) is True
        assert persistence_service.current_run_lock.get(conn).run_id == "r1"
        persistence_service.current_run_lock.release(conn, run_id="r1")
        assert persistence_service.current_run_lock.get(conn) is None
    finally:
        conn.close()


def test_second_acquire_fails_while_first_live_holder_still_holds_it(persistence_service):
    conn = persistence_service.database.connect()
    try:
        assert persistence_service.current_run_lock.try_acquire(
            conn, run_id="r1", started_at=D, owner_pid=os.getpid(),
        ) is True
        # Same connection's own live pid -- a second acquire must fail.
        assert persistence_service.current_run_lock.try_acquire(
            conn, run_id="r2", started_at=D, owner_pid=os.getpid(),
        ) is False
        assert persistence_service.current_run_lock.get(conn).run_id == "r1"
    finally:
        conn.close()


def test_two_simultaneous_try_acquire_calls_only_one_succeeds(tmp_path: Path):
    """Real concurrency, not simulated: two threads, two separate
    connections to the same DB file, racing `try_acquire` at the same
    time. Exactly one must win -- proves BEGIN IMMEDIATE actually
    serializes the check-then-write instead of both readers seeing 'no
    row' and both inserting."""
    database = Database(tmp_path / "race.db")
    database.initialize()
    service = PersistenceService(database)

    results: List[bool] = [None, None]  # type: ignore[list-item]

    def attempt(index: int, run_id: str) -> None:
        conn = database.connect()
        try:
            results[index] = service.current_run_lock.try_acquire(
                conn, run_id=run_id, started_at=D, owner_pid=os.getpid(),
            )
        finally:
            conn.close()

    t1 = Thread(target=attempt, args=(0, "race-a"))
    t2 = Thread(target=attempt, args=(1, "race-b"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert sorted(results) == [False, True]

    conn = database.connect()
    try:
        holder = service.current_run_lock.get(conn)
    finally:
        conn.close()
    assert holder is not None
    assert holder.run_id in ("race-a", "race-b")


# ---------------------------------------------------------------------------
# Stale-owner reclaim: the core of this audit
# ---------------------------------------------------------------------------


def test_lock_held_by_a_confirmed_dead_process_is_reclaimed(persistence_service):
    dead_pid = _dead_pid()
    conn = persistence_service.database.connect()
    try:
        # Simulates a CURRENT run that crashed: the lock row survives, its
        # owner_pid is real but that process has already exited.
        persistence_service.current_run_lock.try_acquire(conn, run_id="crashed-run", started_at=D, owner_pid=dead_pid)
        assert persistence_service.current_run_lock.get(conn).run_id == "crashed-run"

        acquired = persistence_service.current_run_lock.try_acquire(
            conn, run_id="new-run-after-restart", started_at=D + timedelta(minutes=1), owner_pid=os.getpid(),
        )
        assert acquired is True
        holder = persistence_service.current_run_lock.get(conn)
        assert holder.run_id == "new-run-after-restart"
    finally:
        conn.close()


def test_lock_held_by_a_confirmed_live_process_is_never_reclaimed(persistence_service):
    """The inverse of the above: a lock whose owner_pid is THIS test
    process (unquestionably alive) must never be stolen, no matter how
    long it has been held -- no timeout-based guessing."""
    conn = persistence_service.database.connect()
    try:
        persistence_service.current_run_lock.try_acquire(
            conn, run_id="genuinely-active-run", started_at=D - timedelta(hours=999), owner_pid=os.getpid(),
        )
        acquired = persistence_service.current_run_lock.try_acquire(
            conn, run_id="would-be-thief", started_at=D, owner_pid=os.getpid(),
        )
        assert acquired is False
        assert persistence_service.current_run_lock.get(conn).run_id == "genuinely-active-run"
    finally:
        conn.close()


def test_lock_row_with_no_recorded_owner_pid_is_treated_as_live_not_stale(persistence_service):
    """A lock row written before `owner_pid` existed (NULL) must NOT be
    guessed to be stale -- there is nothing to verify it against, and
    assuming it is dead is exactly the blind behavior this audit forbids."""
    conn = persistence_service.database.connect()
    try:
        conn.execute(
            "INSERT INTO current_run_lock (id, run_id, started_at, owner_pid) VALUES ('singleton', ?, ?, NULL)",
            ("legacy-run", D.isoformat()),
        )
        conn.commit()
        acquired = persistence_service.current_run_lock.try_acquire(
            conn, run_id="new-run", started_at=D, owner_pid=os.getpid(),
        )
        assert acquired is False
        assert persistence_service.current_run_lock.get(conn).run_id == "legacy-run"
    finally:
        conn.close()
