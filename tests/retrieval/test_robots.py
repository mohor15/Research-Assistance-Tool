"""robots.txt: allowed URL, disallowed URL, unreachable robots.txt, disabled checking."""

from __future__ import annotations

import requests

from research_assistant.retrieval.robots import RobotsChecker

ROBOTS_TXT = """
User-agent: *
Disallow: /private/
Allow: /
"""


class _FakeRobotsResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


def test_allowed_url_passes(monkeypatch):
    monkeypatch.setattr(
        "research_assistant.retrieval.robots.requests.get",
        lambda *a, **k: _FakeRobotsResponse(200, ROBOTS_TXT),
    )
    checker = RobotsChecker(enabled=True, user_agent="TestBot")

    assert checker.is_allowed("https://example.com/news/article") is True


def test_disallowed_url_is_blocked(monkeypatch):
    monkeypatch.setattr(
        "research_assistant.retrieval.robots.requests.get",
        lambda *a, **k: _FakeRobotsResponse(200, ROBOTS_TXT),
    )
    checker = RobotsChecker(enabled=True, user_agent="TestBot")

    assert checker.is_allowed("https://example.com/private/secret") is False


def test_missing_robots_txt_defaults_to_allowed(monkeypatch):
    monkeypatch.setattr(
        "research_assistant.retrieval.robots.requests.get",
        lambda *a, **k: _FakeRobotsResponse(404, ""),
    )
    checker = RobotsChecker(enabled=True, user_agent="TestBot")

    assert checker.is_allowed("https://example.com/private/secret") is True


def test_unreachable_robots_txt_defaults_to_allowed_not_crashed(monkeypatch):
    def _raise(*a, **k):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr("research_assistant.retrieval.robots.requests.get", _raise)
    checker = RobotsChecker(enabled=True, user_agent="TestBot")

    assert checker.is_allowed("https://example.com/anything") is True


def test_disabled_checking_always_allows_without_fetching(monkeypatch):
    def _fail_if_called(*a, **k):
        raise AssertionError("robots.txt should never be fetched when disabled")

    monkeypatch.setattr("research_assistant.retrieval.robots.requests.get", _fail_if_called)
    checker = RobotsChecker(enabled=False, user_agent="TestBot")

    assert checker.is_allowed("https://example.com/private/secret") is True


def test_robots_txt_is_only_fetched_once_per_origin(monkeypatch):
    calls = []

    def _get(*a, **k):
        calls.append(a)
        return _FakeRobotsResponse(200, ROBOTS_TXT)

    monkeypatch.setattr("research_assistant.retrieval.robots.requests.get", _get)
    checker = RobotsChecker(enabled=True, user_agent="TestBot")

    checker.is_allowed("https://example.com/news/a")
    checker.is_allowed("https://example.com/news/b")

    assert len(calls) == 1
