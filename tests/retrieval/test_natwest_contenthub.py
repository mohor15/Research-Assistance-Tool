"""Targeted content-hub API discovery for NatWest Group (retrieval.
natwest_contenthub) -- additive, NatWest-only, never touches the real
network in this suite (deterministic fixtures/mocks only, mirroring
test_ft_search.py's conventions for the project's other non-HTML-scraping
discovery adapter)."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.config import get_config
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval import natwest_contenthub as nw
from research_assistant.retrieval.discovery import discover_candidates
from research_assistant.retrieval.models import CachePolicy, DiscoveryMethod, SourceCategory, SourceDefinition
from research_assistant.retrieval.service import build_source_definitions


def _natwest_source() -> SourceDefinition:
    return SourceDefinition(
        name="NatWest Group", domain="natwestgroup.com",
        urls=["https://www.natwestgroup.com/news-and-insights/news-room/press-releases.html"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.MEDIUM, category=SourceCategory.PEER,
    )


def _page(articles, *, total_articles=None):
    return {
        "success": True,
        "responseObject": {
            "totalArticles": total_articles if total_articles is not None else len(articles),
            "paginationList": [{"totalResults": len(articles), "start": 0, "articles": articles}],
        },
    }


_SPEECH_ARTICLE = {
    "date": "14 Oct 2026",
    "description": "<p>Some description.</p>",
    "title": "NatWest improves AI speech recognition for UK banking customers with NVIDIA Nemotron",
    "url": "/content/natwestgroup_com/en_uk/natwestgroup/news-and-insights/news-room/press-releases/2026/"
           "natwest-improves-ai-speech-recognition-for-uk-banking-customers-",
    "theme": "Ai & data",
}


class _FakeResponse:
    def __init__(self, *, status_code=200, json_body=None, raise_json_error=False):
        self.status_code = status_code
        self._json_body = json_body
        self._raise_json_error = raise_json_error

    def json(self):
        if self._raise_json_error:
            raise ValueError("not json")
        return self._json_body


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_natwest_is_present_in_the_real_source_registry():
    config = get_config()
    names = {s.name for s in config.sources.peers}
    assert "NatWest Group" in names


def test_natwest_source_flows_through_to_the_retrieval_layer():
    config = get_config()
    defs = build_source_definitions(config)
    nat = next(d for d in defs if d.name == "NatWest Group")
    assert nat.domain == nw.NATWEST_DOMAIN


# ---------------------------------------------------------------------------
# discover_via_natwest_contenthub: parsing a real-shaped response
# ---------------------------------------------------------------------------


def test_real_article_is_parsed_into_a_candidate_url(monkeypatch):
    calls = []

    def fake_post(url, *, json, timeout, headers):
        calls.append((url, json))
        return _FakeResponse(json_body=_page([_SPEECH_ARTICLE]))

    monkeypatch.setattr(nw.requests, "post", fake_post)

    candidates = nw.discover_via_natwest_contenthub(_natwest_source())

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.url == (
        "https://www.natwestgroup.com/news-and-insights/news-room/press-releases/2026/"
        "natwest-improves-ai-speech-recognition-for-uk-banking-customers-.html"
    )
    assert candidate.title == "NatWest improves AI speech recognition for UK banking customers with NVIDIA Nemotron"
    assert candidate.published_at == datetime(2026, 10, 14, tzinfo=timezone.utc)
    assert candidate.discovery_method == DiscoveryMethod.CONTENT_HUB_API
    assert len(calls) == 1
    assert calls[0][1]["filters"] == [{"firstleveltags": "natwestgroup-content-hub/themes/ai-&-data"}]


def test_content_path_is_mapped_to_the_real_public_page_not_the_internal_aem_path():
    """Regression: the API's `url` field is an AEM internal JCR content path
    that 404s if used directly (confirmed live). The real public page drops
    the content-root prefix and has a `.html` extension."""
    assert nw._public_url(
        "/content/natwestgroup_com/en_uk/natwestgroup/news-and-insights/news-room/"
        "press-releases/2026/natwest-improves-ai-speech-recognition-for-uk-banking-customers-"
    ) == (
        "https://www.natwestgroup.com/news-and-insights/news-room/press-releases/2026/"
        "natwest-improves-ai-speech-recognition-for-uk-banking-customers-.html"
    )


def test_an_article_with_no_url_is_skipped(monkeypatch):
    broken = dict(_SPEECH_ARTICLE)
    broken.pop("url")
    monkeypatch.setattr(nw.requests, "post", lambda *a, **k: _FakeResponse(json_body=_page([broken])))

    candidates = nw.discover_via_natwest_contenthub(_natwest_source())

    assert candidates == []


def test_multiple_articles_across_one_page_entry_all_parsed(monkeypatch):
    second = dict(_SPEECH_ARTICLE, url="/content/.../second-article", title="Second AI article")
    monkeypatch.setattr(nw.requests, "post", lambda *a, **k: _FakeResponse(json_body=_page([_SPEECH_ARTICLE, second])))

    candidates = nw.discover_via_natwest_contenthub(_natwest_source())

    assert len(candidates) == 2
    assert {c.title for c in candidates} == {
        "NatWest improves AI speech recognition for UK banking customers with NVIDIA Nemotron",
        "Second AI article",
    }


# ---------------------------------------------------------------------------
# Failure handling -- every failure mode yields [], never raises
# ---------------------------------------------------------------------------


def test_non_200_response_yields_empty_list(monkeypatch):
    monkeypatch.setattr(nw.requests, "post", lambda *a, **k: _FakeResponse(status_code=503))
    assert nw.discover_via_natwest_contenthub(_natwest_source()) == []


def test_non_json_response_yields_empty_list(monkeypatch):
    monkeypatch.setattr(nw.requests, "post", lambda *a, **k: _FakeResponse(raise_json_error=True))
    assert nw.discover_via_natwest_contenthub(_natwest_source()) == []


def test_success_false_yields_empty_list(monkeypatch):
    monkeypatch.setattr(nw.requests, "post", lambda *a, **k: _FakeResponse(json_body={"success": False}))
    assert nw.discover_via_natwest_contenthub(_natwest_source()) == []


def test_network_exception_yields_empty_list_never_raises(monkeypatch):
    import requests as real_requests

    def raise_exc(*a, **k):
        raise real_requests.RequestException("connection failed")

    monkeypatch.setattr(nw.requests, "post", raise_exc)
    assert nw.discover_via_natwest_contenthub(_natwest_source()) == []


def test_robots_disallowed_skips_the_request_entirely(monkeypatch):
    calls = []
    monkeypatch.setattr(nw.requests, "post", lambda *a, **k: calls.append(1) or _FakeResponse(json_body=_page([_SPEECH_ARTICLE])))

    class _DenyingRobots:
        def is_allowed(self, url):
            return False

    candidates = nw.discover_via_natwest_contenthub(_natwest_source(), robots=_DenyingRobots())

    assert candidates == []
    assert calls == []  # the request itself was never made


# ---------------------------------------------------------------------------
# Bounded pagination
# ---------------------------------------------------------------------------


def test_pagination_stops_once_total_articles_is_reached(monkeypatch):
    first_batch = [dict(_SPEECH_ARTICLE, url=f"/content/.../article-{i}", title=f"Article {i}") for i in range(3)]
    second_batch = [dict(_SPEECH_ARTICLE, url=f"/content/.../article-{i}", title=f"Article {i}") for i in range(3, 5)]
    responses = [_page(first_batch, total_articles=5), _page(second_batch, total_articles=5)]

    def fake_post(url, *, json, timeout, headers):
        return _FakeResponse(json_body=responses.pop(0))

    monkeypatch.setattr(nw.requests, "post", fake_post)

    candidates = nw.discover_via_natwest_contenthub(_natwest_source())

    assert len(candidates) == 5


def test_pagination_is_bounded_and_never_loops_forever(monkeypatch):
    """A pathological server that always claims more articles exist than
    it ever actually returns must not cause an infinite request loop."""
    call_count = {"n": 0}

    def fake_post(url, *, json, timeout, headers):
        call_count["n"] += 1
        article = dict(_SPEECH_ARTICLE, url=f"/content/.../article-{call_count['n']}", title=f"Article {call_count['n']}")
        return _FakeResponse(json_body=_page([article], total_articles=999999))

    monkeypatch.setattr(nw.requests, "post", fake_post)

    candidates = nw.discover_via_natwest_contenthub(_natwest_source())

    assert call_count["n"] == nw._MAX_REQUESTS
    assert len(candidates) == nw._MAX_REQUESTS


# ---------------------------------------------------------------------------
# Wiring into discovery.discover_candidates: domain-gated, additive, and
# never loses a tie against a same-scoring but irrelevant HTML-link
# candidate once truncated.
# ---------------------------------------------------------------------------


class _StubHTTPClient:
    """Minimal stand-in for HTTPClient -- returns a page with generic nav
    links (no real press-release content), exactly like NatWest's real
    hub page (confirmed live: zero press-release links in the static/
    browser-rendered HTML)."""

    def __init__(self, html):
        self._html = html
        self.robots = None

    def get(self, url, *, cache_policy=CachePolicy.CACHE):
        from research_assistant.retrieval.models import RetrievalMethod, RetrievedPage

        return RetrievedPage(
            url=url, final_url=url, status_code=200, retrieved_at=datetime.now(timezone.utc),
            method=RetrievalMethod.HTTP, content_type="text/html", html=self._html, success=True,
        )


_GENERIC_NAV_HTML = """
<html><body>
<a href="https://www.natwestgroup.com/sustainability/society/accessible-banking.html">Accessible banking</a>
<a href="https://www.natwestgroup.com/who-we-are/board-and-governance/compliance-statement.html">Compliance statement</a>
</body></html>
"""


def test_contenthub_discovery_only_fires_for_the_natwest_domain(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(nw, "discover_via_natwest_contenthub", lambda *a, **k: called.__setitem__("n", called["n"] + 1) or [])

    other_source = SourceDefinition(
        name="Some Other Source", domain="example.com", urls=["https://example.com/news"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.MEDIUM, category=SourceCategory.SECONDARY,
    )
    discover_candidates(other_source, http_client=_StubHTTPClient(_GENERIC_NAV_HTML), max_candidates=50)

    assert called["n"] == 0


def test_contenthub_candidates_are_never_truncated_out_by_a_tied_generic_link(monkeypatch):
    """Reproduces the real, confirmed scenario: a generic nav link
    ("Compliance statement...") scores an equal topic-keyword hit (1,
    via "compliance") to a real AI article, so without explicit ordering
    a low `max_candidates` could truncate away the real content. The
    content-hub result must win that tie."""
    from research_assistant.retrieval import discovery as discovery_module
    from research_assistant.retrieval.natwest_contenthub import _candidate_from_article

    real_candidate = _candidate_from_article(
        _SPEECH_ARTICLE, source=_natwest_source(), now=datetime.now(timezone.utc),
    )
    monkeypatch.setattr(discovery_module, "discover_via_natwest_contenthub", lambda *a, **k: [real_candidate])

    candidates = discover_candidates(
        _natwest_source(), http_client=_StubHTTPClient(_GENERIC_NAV_HTML), max_candidates=1,
        topic_keywords=["compliance", "ai"],
    )

    assert len(candidates) == 1
    assert candidates[0].discovery_method == DiscoveryMethod.CONTENT_HUB_API
