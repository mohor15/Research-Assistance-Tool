"""Manual run CLI (Phase 8 Part 15): correct job dispatch, no Streamlit
dependency, testable via an injected fake PipelineRunner."""

from __future__ import annotations

from research_assistant.scheduling.cli import main, parse_args, run

from tests.scheduling.conftest import FakePipelineRunner


def test_parse_args_requires_digest_type():
    args = parse_args(["--digest-type", "morning"])
    assert args.digest_type == "morning"


def test_parse_args_rejects_an_invalid_digest_type():
    import pytest

    with pytest.raises(SystemExit):
        parse_args(["--digest-type", "afternoon"])


def test_run_dispatches_to_the_morning_job(fake_runner):
    result = run("morning", runner=fake_runner)

    assert len(fake_runner.calls) == 1
    assert fake_runner.calls[0]["digest_type"].value == "morning"
    assert result.status.value == "success"


def test_run_dispatches_to_the_evening_job(fake_runner):
    result = run("evening", runner=fake_runner)

    assert fake_runner.calls[0]["digest_type"].value == "evening"
    assert result.status.value == "success"


def test_parse_args_accepts_current_digest_type():
    args = parse_args(["--digest-type", "current"])
    assert args.digest_type == "current"


def test_run_dispatches_to_the_current_job(fake_runner):
    """`python -m research_assistant.scheduling.cli --digest-type current`
    -- CURRENT's manual-trigger CLI entry point."""
    result = run("current", runner=fake_runner)

    assert len(fake_runner.calls) == 1
    assert fake_runner.calls[0]["digest_type"].value == "current"
    assert result.status.value == "success"


def test_main_returns_zero_on_success(fake_runner, monkeypatch, capsys):
    monkeypatch.setattr("research_assistant.scheduling.cli.PipelineRunner.build", lambda *a, **k: fake_runner)

    exit_code = main(["--digest-type", "morning"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "run_id=" in captured.out
    assert "status=success" in captured.out


def test_cli_module_never_puts_business_logic_in_streamlit():
    """The CLI is the manual-run mechanism explicitly required to live
    outside Streamlit (Part 15) — confirm it has no Streamlit import."""
    from pathlib import Path

    import research_assistant.scheduling.cli as cli_module

    lines = Path(cli_module.__file__).read_text(encoding="utf-8").splitlines()
    import_lines = [line.strip() for line in lines if line.strip().startswith(("import ", "from "))]
    assert not any("streamlit" in line.lower() for line in import_lines)
