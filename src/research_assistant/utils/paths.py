"""Filesystem locations shared by every phase.

Resolving the project root from this file's own location (rather than from
the process's current working directory) means config loading, data storage,
and logging all agree on the same paths no matter where a script or test is
launched from.
"""

from __future__ import annotations

from pathlib import Path

# src/research_assistant/utils/paths.py -> src/research_assistant -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

SETTINGS_PATH = CONFIG_DIR / "settings.yaml"
SOURCES_PATH = CONFIG_DIR / "sources.yaml"
TOPICS_PATH = CONFIG_DIR / "topics.yaml"

ENV_PATH = PROJECT_ROOT / ".env"
