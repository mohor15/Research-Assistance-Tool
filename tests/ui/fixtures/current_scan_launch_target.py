"""AppTest target for `tests/ui/test_current_scan_launch.py` -- renders
`_render_current_page` directly against a REAL temp-file SQLite database
(path supplied via `st.query_params["db_path"]`, built by the test's own
`persistence_service` fixture so both sides see the same file), with
`app_module._launch_current_scan_subprocess` monkeypatched to a fake that
never actually spawns a process, calls Ollama, or touches the network.

Query params:
  db_path            -- the real temp-file SQLite database to open.
  launch_log_path     -- fake-launch call counter (one "launch\\n" line per call).
  launch_mode         -- "success" (default) or "fail" (raises OSError).
  poll_result_path    -- a file whose content controls the fake child
                          process's `.poll()` return value: "none" (still
                          running) or an integer (the exit code). The test
                          rewrites this file between `.run()` calls to
                          simulate the child's progress over real wall time
                          -- never a sleep/timeout in this fixture itself.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from research_assistant.config import get_config
from research_assistant.storage import Database, PersistenceService
from research_assistant.ui import app as app_module

db_path = st.query_params["db_path"]
launch_log_path = st.query_params["launch_log_path"]
launch_mode = st.query_params.get("launch_mode", "success")
poll_result_path = st.query_params.get("poll_result_path")
database = Database(Path(db_path))
service = PersistenceService(database)
config = get_config()


class _FakeProc:
    def __init__(self, pid: int, poll_result_path: str):
        self.pid = pid
        self._poll_result_path = poll_result_path

    def poll(self):
        try:
            content = Path(self._poll_result_path).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return None if content == "none" else int(content)


def _fake_launch():
    with open(launch_log_path, "a", encoding="utf-8") as f:
        f.write("launch\n")
    if launch_mode == "fail":
        raise OSError("fake Popen failure (simulated -- no real process was spawned)")
    return 99999, "fake.log", _FakeProc(99999, poll_result_path)


app_module._launch_current_scan_subprocess = _fake_launch

app_module._render_current_page(service, config)
