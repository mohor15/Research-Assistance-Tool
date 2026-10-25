"""Phase 8 Part 8 — retry/timeout bound review: a permanent failure must
terminate after exactly `max_retries + 1` attempts, never loop indefinitely,
and delays/timeouts must come from configuration, never be hardcoded or
unbounded. Complements test_client.py's existing retry/timeout coverage."""

from __future__ import annotations

import requests

from research_assistant.retrieval.client import HTTPClient

from tests.retrieval.fakes import FakeSession


class _CountingSleep:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


def test_a_permanently_failing_source_terminates_after_exactly_max_retries_plus_one_attempts():
    # Script far more failures than max_retries could ever consume — if the
    # loop were unbounded, this would raise "ran out of scripted responses"
    # well before completion; instead it must stop itself deterministically.
    session = FakeSession([requests.ConnectionError("permanently down")] * 1000)
    sleep = _CountingSleep()
    client = HTTPClient(timeout_seconds=5, max_retries=3, retry_delay_seconds=0.01, session=session, sleep_fn=sleep)

    page = client.get("http://permanently-down.example/a")

    assert page.success is False
    assert len(session.calls) == 4  # 1 initial attempt + 3 retries, never more
    assert len(sleep.calls) == 3  # a sleep between each retry, none after the final attempt


def test_zero_max_retries_still_makes_exactly_one_attempt_and_fails_cleanly():
    session = FakeSession([requests.Timeout("slow")])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session, sleep_fn=lambda s: None)

    page = client.get("http://example.com/a")

    assert page.success is False
    assert len(session.calls) == 1


def test_retry_delay_is_bounded_by_configuration_not_hardcoded():
    """The exact configured delay is used for every retry sleep — no
    exponential backoff or unbounded growth was introduced."""
    session = FakeSession([requests.Timeout("slow"), requests.Timeout("slow"), requests.Timeout("slow")])
    sleep = _CountingSleep()
    client = HTTPClient(timeout_seconds=5, max_retries=2, retry_delay_seconds=7.5, session=session, sleep_fn=sleep)

    client.get("http://example.com/a")

    assert sleep.calls == [7.5, 7.5]


def test_llm_client_makes_a_single_bounded_attempt_with_no_retry_loop():
    """OllamaClient has no retry loop at all — a single request with a
    bounded timeout, raising immediately on failure. Confirms there is no
    infinite-retry risk on the LLM side either."""
    from research_assistant.llm.client import OllamaClient
    from research_assistant.llm.errors import LLMUnavailableError

    class _FakeLLMSession:
        def __init__(self):
            self.call_count = 0

        def post(self, url, json=None, timeout=None):
            self.call_count += 1
            raise requests.ConnectionError("Ollama unreachable")

    fake_session = _FakeLLMSession()
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b", session=fake_session)

    from pydantic import BaseModel

    class _Schema(BaseModel):
        x: str

    try:
        client.generate_structured(system="s", prompt="p", schema=_Schema)
        assert False, "expected LLMUnavailableError"
    except LLMUnavailableError:
        pass

    assert fake_session.call_count == 1  # exactly one attempt, no retry loop
