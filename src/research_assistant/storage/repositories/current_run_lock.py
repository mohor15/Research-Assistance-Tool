"""CURRENT's application-level run mutex (Phase 17, Section 3) -- a single
`id='singleton'` row whose presence means a CURRENT pipeline run is
currently in flight. This is a single-process (one Streamlit/CLI process,
one SQLite file) mutex, not a distributed lock -- sufficient for this
application's actual deployment, and consistent with the existing
`scheduler_state` singleton-row pattern.

Stale-lock audit (Phase 17 follow-up): the original implementation had no
way to tell a genuinely active holder apart from one whose owning process
had died without reaching `PipelineRunner.run()`'s `finally` release
(crash, `kill -9`, machine power-loss) -- such a row would permanently
brick every future CURRENT run, since nothing ever deletes it. `try_acquire`
now records the acquiring OS process's pid and, when it finds an existing
lock, checks whether that pid is still alive on THIS machine before
refusing. A lock whose owning process is confirmed dead can by definition
no longer be a genuinely active run, so reclaiming it never risks two live
runs overlapping -- this is a deterministic liveness check, not a blind
timeout (an arbitrarily slow-but-alive run is never stolen from, no matter
how long it has held the lock).
"""

from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from research_assistant.storage.serde import dt_from_str, dt_to_str

_SINGLETON_ID = "singleton"


@dataclass
class CurrentRunLockRecord:
    run_id: str
    started_at: datetime
    owner_pid: Optional[int] = None


def _is_process_alive(pid: Optional[int]) -> bool:
    """True iff `pid` identifies a process currently running on this
    machine. `None` (a lock row from before `owner_pid` existed) is
    treated as "cannot verify liveness" -- NOT as alive and NOT as dead;
    callers must decide explicitly rather than have this function guess
    (see the comment at its one call site)."""
    if pid is None or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Exists, just owned by another user -- still alive.
            return True
        return True


def probe_owner_liveness(pid: Optional[int]) -> str:
    """UI-facing, presentation-only tri-state liveness probe: `"alive"`,
    `"dead"`, or `"indeterminate"`. NEVER the authority for whether a lock
    can actually be reclaimed -- `try_acquire`'s own unchanged
    `_is_process_alive` check (above) remains the single source of truth
    for that; this function only lets a caller (the admin UI) distinguish
    a genuinely active run from a stale lock worth offering the user a
    chance to reclaim via a normal launch attempt, without ever directly
    deleting/modifying the lock row itself.

    Deliberately conservative per its one real behavioral difference from
    `_is_process_alive`: on Windows, `OpenProcess` failing with
    `ERROR_ACCESS_DENIED` (the process exists but this process lacks
    permission to query it) returns `"indeterminate"` here, never
    `"dead"` -- `_is_process_alive` collapses that same case to `False`
    (treated as not-alive), which is the correct, narrower contract for
    the reclaim decision itself (a stale lock should still be reclaimable
    even if the stale owner happens to be unqueryable), but is NOT safe to
    surface to a user as "definitely dead." `pid is None` (a lock row
    from before `owner_pid` existed) is `"indeterminate"`, matching
    `try_acquire`'s own special-cased treatment of that one case as
    conservatively live (`existing_pid is None or _is_process_alive(...)`
    short-circuits on `None` alone). A non-`None` `pid <= 0` is NOT the
    same case -- `_is_process_alive` does not special-case it, it falls
    through to that function's own `pid <= 0: return False`, so the real
    authority already treats it as reclaimable; reporting it as
    `"indeterminate"` here would contradict what `try_acquire` will
    actually do, so it is reported as `"dead"` instead, matching the
    authority exactly."""
    if pid is None:
        return "indeterminate"
    if pid <= 0:
        return "dead"
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        ERROR_ACCESS_DENIED = 5
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            last_error = ctypes.windll.kernel32.GetLastError()
            return "indeterminate" if last_error == ERROR_ACCESS_DENIED else "dead"
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return "indeterminate"
            return "alive" if exit_code.value == STILL_ACTIVE else "dead"
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "dead"
        except PermissionError:
            return "alive"
        except OSError:
            return "indeterminate"
        return "alive"


class CurrentRunLockRepository:
    def try_acquire(
        self, conn: sqlite3.Connection, *, run_id: str, started_at: datetime, owner_pid: Optional[int] = None,
    ) -> bool:
        """Atomic (wrapped in its own `BEGIN IMMEDIATE` transaction, so a
        concurrent caller's `try_acquire` genuinely blocks rather than
        racing on the read-then-write): acquires the lock if no row exists,
        OR if the existing row's `owner_pid` is confirmed dead (stale --
        see module docstring). A row with no recorded pid (`owner_pid IS
        NULL`, i.e. predates this column) is conservatively treated as
        LIVE, never stale -- there is nothing to verify it against, and
        guessing it is dead is exactly the kind of blind behavior this
        audit forbids. Returns True iff THIS call's `run_id` now holds the
        lock."""
        owner_pid = owner_pid if owner_pid is not None else os.getpid()
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT run_id, owner_pid FROM current_run_lock WHERE id = ?", (_SINGLETON_ID,)
            ).fetchone()
            if row is not None:
                existing_pid = row["owner_pid"]
                if existing_pid is None or _is_process_alive(existing_pid):
                    conn.rollback()
                    return False
                # existing_pid is recorded and confirmed dead: its holder
                # cannot possibly still be running, so this is a stale lock
                # left behind by an abnormal process exit, not a live run.
                conn.execute("DELETE FROM current_run_lock WHERE id = ?", (_SINGLETON_ID,))
            conn.execute(
                "INSERT INTO current_run_lock (id, run_id, started_at, owner_pid) VALUES (?, ?, ?, ?)",
                (_SINGLETON_ID, run_id, dt_to_str(started_at), owner_pid),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        row = conn.execute("SELECT run_id FROM current_run_lock WHERE id = ?", (_SINGLETON_ID,)).fetchone()
        return row is not None and row["run_id"] == run_id

    def release(self, conn: sqlite3.Connection, *, run_id: str) -> None:
        """Only releases the lock if `run_id` is the one actually holding
        it -- a late/duplicate release from a run that already lost the
        lock (or never held it) must never clear a DIFFERENT run's lock."""
        conn.execute("DELETE FROM current_run_lock WHERE id = ? AND run_id = ?", (_SINGLETON_ID, run_id))
        conn.commit()

    def get(self, conn: sqlite3.Connection) -> Optional[CurrentRunLockRecord]:
        row = conn.execute(
            "SELECT run_id, started_at, owner_pid FROM current_run_lock WHERE id = ?", (_SINGLETON_ID,)
        ).fetchone()
        if row is None:
            return None
        return CurrentRunLockRecord(
            run_id=row["run_id"], started_at=dt_from_str(row["started_at"]), owner_pid=row["owner_pid"],
        )
