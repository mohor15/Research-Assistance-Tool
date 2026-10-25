"""A small, testable LLM abstraction over the configured local Ollama model.

Agent code (Phase 3+) depends on `LLMClient`/`OllamaClient` here rather than
making raw HTTP calls directly, and on `create_ollama_client(config)` rather
than reading `provider`/`model`/`base_url` itself — so the model can be
swapped via `config/settings.yaml` (or the `OLLAMA_MODEL`/`OLLAMA_BASE_URL`
env vars) with no agent code changes. No paid/cloud LLM API is supported.
"""

from research_assistant.llm.client import LLMClient, OllamaClient, create_ollama_client
from research_assistant.llm.errors import LLMError, LLMResponseError, LLMUnavailableError

__all__ = [
    "LLMClient",
    "OllamaClient",
    "create_ollama_client",
    "LLMError",
    "LLMResponseError",
    "LLMUnavailableError",
]
