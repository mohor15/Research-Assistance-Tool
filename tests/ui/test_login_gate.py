"""The single shared username/password login gate (`app._render_login_gate`
/ `app._credentials_are_valid`) added to the Streamlit UI. Exercises the
gate via Streamlit's own `AppTest` harness against the dedicated
`fixtures/login_gate_target.py` target -- never `main()` itself, which
connects to the REAL production database and is deliberately out of
scope for these tests. `load_app_login_password` is monkeypatched inside
the fixture to a fake value controlled by query params -- no real .env
file or environment variable is ever read."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from research_assistant.config import get_config
from research_assistant.ui.app import _credentials_are_valid

_TARGET = str(Path(__file__).parent / "fixtures" / "login_gate_target.py")


def _run_gate(*, configured_password="secret123", auth_username="admin", pre_authenticated=False):
    at = AppTest.from_file(_TARGET, default_timeout=30)
    at.query_params["configured_password"] = configured_password if configured_password is not None else "__none__"
    at.query_params["auth_username"] = auth_username
    at.query_params["pre_authenticated"] = "true" if pre_authenticated else "false"
    at.run()
    return at


def _gate_result_text(at) -> str:
    return "\n".join(str(el.value) for el in at.markdown)


# ---------------------------------------------------------------------------
# Pure comparison logic -- no Streamlit involved at all.
# ---------------------------------------------------------------------------


class TestCredentialsAreValid:
    def test_correct_username_and_password_is_valid(self):
        config = get_config().model_copy(deep=True)
        config.settings.auth.username = "admin"
        assert _credentials_are_valid("admin", "secret123", config, configured_password="secret123") is True

    def test_wrong_password_is_invalid(self):
        config = get_config().model_copy(deep=True)
        config.settings.auth.username = "admin"
        assert _credentials_are_valid("admin", "wrong", config, configured_password="secret123") is False

    def test_wrong_username_is_invalid(self):
        config = get_config().model_copy(deep=True)
        config.settings.auth.username = "admin"
        assert _credentials_are_valid("someone_else", "secret123", config, configured_password="secret123") is False

    def test_no_configured_password_is_always_invalid_even_with_an_empty_guess(self):
        config = get_config().model_copy(deep=True)
        config.settings.auth.username = "admin"
        assert _credentials_are_valid("admin", "", config, configured_password=None) is False
        assert _credentials_are_valid("admin", "secret123", config, configured_password=None) is False


# ---------------------------------------------------------------------------
# The rendered gate itself, via AppTest.
# ---------------------------------------------------------------------------


def test_an_unauthenticated_session_sees_the_login_form_and_the_gate_blocks(tmp_path):
    at = _run_gate()

    assert at.exception == []
    assert len(at.text_input) == 2  # username + password fields
    assert len(at.button) >= 1 or len(at.form) >= 1  # a submit control exists
    assert "GATE_RESULT:False" in _gate_result_text(at)


def test_submitting_correct_credentials_authenticates_and_the_gate_opens(tmp_path):
    at = _run_gate(configured_password="secret123")

    at.text_input[0].set_value("admin")
    at.text_input[1].set_value("secret123")
    at.button[0].click().run()

    assert at.exception == []
    assert at.session_state.get("authenticated") is True


def test_submitting_wrong_credentials_shows_an_error_and_stays_locked_out(tmp_path):
    at = _run_gate(configured_password="secret123")

    at.text_input[0].set_value("admin")
    at.text_input[1].set_value("totally-wrong")
    at.button[0].click().run()

    assert at.exception == []
    assert any("Invalid username or password" in str(el.value) for el in at.error)
    assert not st_authenticated(at)


def test_an_unconfigured_password_shows_a_clear_message_and_disables_submit(tmp_path):
    at = _run_gate(configured_password=None)

    assert at.exception == []
    assert any("not configured" in str(el.value) for el in at.error)
    assert at.button[0].disabled is True


def test_an_already_authenticated_session_skips_the_form_entirely(tmp_path):
    at = _run_gate(pre_authenticated=True)

    assert at.exception == []
    assert at.text_input == []  # no login form rendered at all
    assert "GATE_RESULT:True" in _gate_result_text(at)


def st_authenticated(at) -> bool:
    return at.session_state.get("authenticated") is True
