"""AppTest target for `tests/ui/test_login_gate.py` -- exercises
`app_module._render_login_gate` directly, never `main()` (which connects
to the REAL production database via `_get_persistence_service` --
deliberately never touched by this fixture or its tests).

Query params:
  configured_password -- what `load_app_login_password` should return for
                          this run ("__none__" simulates it being unset).
  auth_username        -- `config.settings.auth.username` for this run.
  pre_authenticated     -- "true" seeds `st.session_state["authenticated"]`
                            before the gate renders, simulating a user who
                            already logged in earlier this session.
"""

from __future__ import annotations

import streamlit as st

from research_assistant.config import get_config
from research_assistant.ui import app as app_module

configured_password = st.query_params.get("configured_password", "")
if configured_password == "__none__":
    configured_password = None
auth_username = st.query_params.get("auth_username", "admin")
pre_authenticated = st.query_params.get("pre_authenticated") == "true"

config = get_config().model_copy(deep=True)
config.settings.auth.username = auth_username

app_module.load_app_login_password = lambda **_kwargs: configured_password

if pre_authenticated:
    st.session_state["authenticated"] = True

authenticated = app_module._render_login_gate(config)
st.write(f"GATE_RESULT:{authenticated}")
