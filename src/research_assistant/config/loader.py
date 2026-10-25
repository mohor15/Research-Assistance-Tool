"""Loads and validates config/settings.yaml, config/sources.yaml, and config/topics.yaml.

This is the *only* place later phases should read YAML files or environment
variables for configuration purposes — agents, retrieval, and pipeline code
must consume the typed `AppConfig` object instead of touching files or
`os.environ` directly. The loader itself never performs retrieval, LLM
calls, or database access.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from research_assistant.config.schema import AppConfig, Settings, SourcesRegistry, TopicsRegistry
from research_assistant.utils.paths import ENV_PATH, SETTINGS_PATH, SOURCES_PATH, TOPICS_PATH


class ConfigError(Exception):
    """Raised whenever configuration is missing, malformed, or fails validation."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read configuration file {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        raise ConfigError(f"Configuration file is empty: {path}")
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a top-level mapping in {path}, got {type(data).__name__}")

    return data


def _apply_env_overrides(settings_data: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay OLLAMA_BASE_URL / OLLAMA_MODEL from the environment, if set.

    Environment variables take precedence over settings.yaml so a developer
    (or a future scheduled job) can point at a different Ollama instance or
    model without editing checked-in config.
    """
    llm = settings_data.setdefault("system", {}).setdefault("llm", {})

    base_url = os.environ.get("OLLAMA_BASE_URL")
    if base_url:
        llm["base_url"] = base_url

    model = os.environ.get("OLLAMA_MODEL")
    if model:
        llm["model"] = model

    return settings_data


def _wrap_validation_error(exc: ValidationError, source: Path) -> ConfigError:
    return ConfigError(f"Configuration validation failed for {source}:\n{exc}")


def load_settings(path: Path = SETTINGS_PATH, *, apply_env_overrides: bool = True) -> Settings:
    data = _load_yaml(path)
    if apply_env_overrides:
        data = _apply_env_overrides(data)
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise _wrap_validation_error(exc, path) from exc


def load_sources(path: Path = SOURCES_PATH) -> SourcesRegistry:
    data = _load_yaml(path)
    try:
        return SourcesRegistry.model_validate(data)
    except ValidationError as exc:
        raise _wrap_validation_error(exc, path) from exc


def load_topics(path: Path = TOPICS_PATH) -> TopicsRegistry:
    data = _load_yaml(path)
    try:
        return TopicsRegistry.model_validate(data)
    except ValidationError as exc:
        raise _wrap_validation_error(exc, path) from exc


def load_config(
    *,
    settings_path: Path = SETTINGS_PATH,
    sources_path: Path = SOURCES_PATH,
    topics_path: Path = TOPICS_PATH,
    env_path: Optional[Path] = ENV_PATH,
) -> AppConfig:
    """Load and validate the full configuration. Not cached — see `get_config()`."""
    if env_path is not None and env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)

    return AppConfig(
        settings=load_settings(settings_path),
        sources=load_sources(sources_path),
        topics=load_topics(topics_path),
    )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide cached configuration. Call `get_config.cache_clear()` to reload."""
    return load_config()


def load_smtp_password(*, env_path: Optional[Path] = ENV_PATH) -> Optional[str]:
    """The SMTP authentication secret -- read ONLY from the `SMTP_PASSWORD`
    environment variable, never from settings.yaml or any other checked-in
    file, and never cached on `AppConfig`/`Settings` (which later code may
    log or display). Returns `None` (never an empty string) when unset, so
    callers can treat "no password" and "not configured" identically."""
    if env_path is not None and env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    return os.environ.get("SMTP_PASSWORD") or None


def load_app_login_password(*, env_path: Optional[Path] = ENV_PATH) -> Optional[str]:
    """The shared UI login password -- read ONLY from the
    `APP_LOGIN_PASSWORD` environment variable, mirroring
    `load_smtp_password` exactly: never from settings.yaml, never cached
    on `AppConfig`, never logged. Returns `None` (never an empty string)
    when unset, so the login screen can show a clear "not configured"
    message instead of silently accepting a blank password."""
    if env_path is not None and env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    return os.environ.get("APP_LOGIN_PASSWORD") or None
