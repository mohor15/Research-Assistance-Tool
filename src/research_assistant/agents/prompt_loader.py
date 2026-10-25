"""Shared system-prompt loading for every agent — keeps prompt text out of
Python source (see `agents/prompts/*.md`)."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")
