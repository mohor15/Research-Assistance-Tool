"""Ollama client with Pydantic-validated structured output.

Only `OllamaClient` talks HTTP; everything above it (agents) depends on the
`LLMClient` protocol, so tests inject a fake implementation and never need
a live Ollama server. `create_ollama_client(config)` is the only place that
reads `system.llm.provider/model/base_url` from configuration — agent code
never hardcodes a model name or reads config itself.

`generate_structured()` sends Ollama the *actual* JSON Schema derived from
the caller's Pydantic `schema` (via `schema.model_json_schema()`) as its
`format` field, rather than the bare string `"json"` — this constrains
generation to the real response shape instead of leaving the model to infer
field names from prose, and the schema is derived, never hand-duplicated,
so the generation contract and the `schema.model_validate(...)` validation
contract below cannot drift apart. `_generate_raw()` still accepts no
schema (falling back to the plain `"json"` mode) for any lower-level caller
that doesn't have one.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional, Protocol, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from research_assistant.llm.errors import LLMResponseError, LLMUnavailableError

if TYPE_CHECKING:
    from research_assistant.config.schema import AppConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_TIMEOUT_SECONDS = 120.0


class LLMClient(Protocol):
    """What agent code actually depends on — small enough to fake in tests."""

    def generate_structured(self, *, system: str, prompt: str, schema: Type[T]) -> T: ...


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._session = session or requests.Session()

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    def generate_structured(self, *, system: str, prompt: str, schema: Type[T]) -> T:
        raw_text = self._generate_raw(system=system, prompt=prompt, response_format=schema.model_json_schema())

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"Model output was not valid JSON: {exc}. Raw output (truncated): {raw_text[:300]!r}"
            ) from exc

        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMResponseError(f"Model output failed schema validation: {exc}") from exc

    def _generate_raw(self, *, system: str, prompt: str, response_format: Optional[dict] = None) -> str:
        payload = {
            "model": self._model,
            "system": system,
            "prompt": prompt,
            # A JSON Schema object grammar-constrains generation to the real
            # response shape; falling back to the bare "json" mode (valid
            # JSON syntax only, no shape constraint) when no schema is given.
            "format": response_format if response_format is not None else "json",
            "stream": False,
        }
        logger.debug("Calling Ollama model=%s at %s", self._model, self._base_url)

        try:
            response = self._session.post(f"{self._base_url}/api/generate", json=payload, timeout=self._timeout)
        except requests.RequestException as exc:
            raise LLMUnavailableError(f"Could not reach Ollama at {self._base_url}: {exc}") from exc

        if response.status_code != 200:
            raise LLMUnavailableError(
                f"Ollama returned HTTP {response.status_code} for model {self._model!r}"
            )

        try:
            envelope = response.json()
        except ValueError as exc:
            raise LLMResponseError(f"Ollama response envelope was not valid JSON: {exc}") from exc

        text = envelope.get("response")
        if not isinstance(text, str):
            raise LLMResponseError("Ollama response envelope is missing a string 'response' field")
        return text


def create_ollama_client(config: "AppConfig", **kwargs) -> OllamaClient:
    """Build an `OllamaClient` from `config.settings.system.llm` — the only
    place agent/pipeline code should read the LLM's model/base_url/timeout
    from."""
    llm_settings = config.settings.system.llm
    kwargs.setdefault("timeout_seconds", llm_settings.timeout_seconds)
    return OllamaClient(base_url=llm_settings.base_url, model=llm_settings.model, **kwargs)
