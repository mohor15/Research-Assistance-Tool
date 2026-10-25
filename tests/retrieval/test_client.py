"""HTTP client: success, timeout+retry, HTTP error, redirect, no-secrets-logged."""

from __future__ import annotations

import requests

from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.models import RetrievalMethod

from tests.retrieval.fakes import FakeResponse, FakeSession


def _client(session, *, max_retries=3, retry_delay=0, sleep_fn=lambda s: None, **kwargs) -> HTTPClient:
    return HTTPClient(
        timeout_seconds=5,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay,
        session=session,
        sleep_fn=sleep_fn,
        **kwargs,
    )


def test_successful_request_returns_populated_page():
    session = FakeSession([FakeResponse(status_code=200, url="http://ex.com/a", text="<html>hi</html>")])
    client = _client(session)

    page = client.get("http://ex.com/a")

    assert page.success is True
    assert page.status_code == 200
    assert page.html == "<html>hi</html>"
    assert page.method == RetrievalMethod.HTTP
    assert page.final_url == "http://ex.com/a"


def test_redirect_is_reflected_in_final_url():
    session = FakeSession([FakeResponse(status_code=200, url="http://ex.com/final", text="ok")])
    client = _client(session)

    page = client.get("http://ex.com/original")

    assert page.url == "http://ex.com/original"
    assert page.final_url == "http://ex.com/final"


def test_timeout_is_retried_then_fails_cleanly():
    session = FakeSession([requests.Timeout("slow"), requests.Timeout("slow"), requests.Timeout("slow")])
    client = _client(session, max_retries=2)

    page = client.get("http://ex.com/slow")

    assert page.success is False
    assert page.failure.reason == "timeout"
    assert len(session.calls) == 3  # 1 initial + 2 retries


def test_transient_server_error_is_retried_until_success():
    session = FakeSession([
        FakeResponse(status_code=503, url="http://ex.com/a"),
        FakeResponse(status_code=200, url="http://ex.com/a", text="recovered"),
    ])
    client = _client(session, max_retries=2)

    page = client.get("http://ex.com/a")

    assert page.success is True
    assert page.html == "recovered"
    assert len(session.calls) == 2


def test_http_error_status_is_reported_as_failure_without_retry_exhaustion():
    session = FakeSession([FakeResponse(status_code=404, url="http://ex.com/missing", text="not found")])
    client = _client(session, max_retries=3)

    page = client.get("http://ex.com/missing")

    assert page.success is False
    assert page.status_code == 404
    assert page.failure.reason == "http_error"
    assert len(session.calls) == 1  # 4xx is not retried


def test_connection_error_is_reported_cleanly():
    session = FakeSession([requests.ConnectionError("dns failure")])
    client = _client(session, max_retries=0)

    page = client.get("http://unreachable.example")

    assert page.success is False
    assert page.failure.reason == "connection_error"


def test_sensitive_response_headers_are_never_stored():
    session = FakeSession([
        FakeResponse(
            status_code=200,
            url="http://ex.com/a",
            text="ok",
            headers={"Content-Type": "text/html", "Set-Cookie": "secret=1", "Authorization": "Bearer xyz"},
        )
    ])
    client = _client(session)

    page = client.get("http://ex.com/a")

    assert "Set-Cookie" not in page.headers
    assert "Authorization" not in page.headers
    assert page.headers["Content-Type"] == "text/html"


def test_user_agent_header_is_sent():
    session = FakeSession([FakeResponse(status_code=200, url="http://ex.com/a", text="ok")])
    client = _client(session, user_agent="TestBot/1.0")

    client.get("http://ex.com/a")

    assert session.calls[0]["headers"]["User-Agent"] == "TestBot/1.0"
