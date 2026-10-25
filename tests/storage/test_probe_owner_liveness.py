"""Audit follow-up: `probe_owner_liveness` is a UI-facing, presentation-
only tri-state liveness probe ("alive"/"dead"/"indeterminate") -- it must
never be the authority for whether a lock can be reclaimed (that remains
`_is_process_alive`/`try_acquire`, unchanged), and must report
"indeterminate" (never "dead") whenever genuine liveness cannot be
confirmed, per the task's "treat as held" rule."""

from __future__ import annotations

import os
import sys

import pytest

from research_assistant.storage.repositories.current_run_lock import probe_owner_liveness


def test_a_genuinely_live_process_is_reported_alive():
    assert probe_owner_liveness(os.getpid()) == "alive"


def test_a_pid_with_no_matching_process_is_reported_dead():
    # A very large pid extremely unlikely to correspond to a real process.
    assert probe_owner_liveness(999_999_999) == "dead"


def test_none_pid_is_indeterminate_never_dead():
    """Matches `try_acquire`'s own special-cased treatment of `owner_pid
    IS NULL` as conservatively live -- `probe_owner_liveness` must agree,
    never report `None` as "dead"."""
    assert probe_owner_liveness(None) == "indeterminate"


def test_non_positive_pid_is_reported_dead_matching_the_real_authority():
    """`_is_process_alive` does NOT special-case a non-None `pid <= 0` --
    it falls through to `pid <= 0: return False`, which `try_acquire`
    then treats as reclaimable. `probe_owner_liveness` must agree (report
    "dead"), never "indeterminate", or the UI would contradict what the
    real mechanism actually does on the next launch attempt."""
    assert probe_owner_liveness(0) == "dead"
    assert probe_owner_liveness(-1) == "dead"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific ERROR_ACCESS_DENIED behavior")
def test_access_denied_is_indeterminate_never_dead(monkeypatch):
    """A process that exists but cannot be queried (permission denied)
    must never be reported as "dead" -- it might well still be alive."""
    import ctypes

    ERROR_ACCESS_DENIED = 5

    def fake_open_process(*_args, **_kwargs):
        return 0  # NULL handle -- OpenProcess "failed"

    def fake_get_last_error():
        return ERROR_ACCESS_DENIED

    monkeypatch.setattr(ctypes.windll.kernel32, "OpenProcess", fake_open_process)
    monkeypatch.setattr(ctypes.windll.kernel32, "GetLastError", fake_get_last_error)

    assert probe_owner_liveness(4) == "indeterminate"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific OpenProcess behavior")
def test_no_such_process_is_dead_not_indeterminate(monkeypatch):
    import ctypes

    ERROR_INVALID_PARAMETER = 87

    def fake_open_process(*_args, **_kwargs):
        return 0

    def fake_get_last_error():
        return ERROR_INVALID_PARAMETER

    monkeypatch.setattr(ctypes.windll.kernel32, "OpenProcess", fake_open_process)
    monkeypatch.setattr(ctypes.windll.kernel32, "GetLastError", fake_get_last_error)

    assert probe_owner_liveness(123456) == "dead"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific GetExitCodeProcess behavior")
def test_get_exit_code_failure_is_indeterminate(monkeypatch):
    import ctypes

    def fake_open_process(*_args, **_kwargs):
        return 12345  # a truthy "handle"

    def fake_get_exit_code_process(_handle, _ptr):
        return 0  # failure return from the Win32 call

    monkeypatch.setattr(ctypes.windll.kernel32, "OpenProcess", fake_open_process)
    monkeypatch.setattr(ctypes.windll.kernel32, "GetExitCodeProcess", fake_get_exit_code_process)
    monkeypatch.setattr(ctypes.windll.kernel32, "CloseHandle", lambda *_a, **_k: None)

    assert probe_owner_liveness(12345) == "indeterminate"
