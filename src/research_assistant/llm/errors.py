"""LLM-layer exceptions.

An agent's `analyze()` lets these propagate rather than swallowing them —
"graceful failure" here means a clear, typed, catchable exception instead
of a raw `requests` traceback or (worse) silently wrong data, not that
failures disappear.
"""


class LLMError(Exception):
    """Base class for LLM-layer errors."""


class LLMUnavailableError(LLMError):
    """The configured LLM backend could not be reached, or returned a server error."""


class LLMResponseError(LLMError):
    """The backend responded, but its output was not valid JSON or failed schema validation."""
