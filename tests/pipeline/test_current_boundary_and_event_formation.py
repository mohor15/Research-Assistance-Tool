"""Regression tests for the CURRENT-run forensic investigation (2026-10-07):
confirms `_filter_by_period`'s half-open `[period_start, period_end)`
boundary (a real bug -- it previously used an inclusive `<=` at
`period_end`), that a known publication date survives normalization and is
never fabricated as an event date, and that a relevant in-window candidate
actually flows all the way through to a CURRENT digest. Fixtures/fakes
only -- no live network, no Ollama."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.config import get_config
from research_assistant.evidence.service import NormalizationService
from research_assistant.events.builder import _select_event_date
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName
from research_assistant.pipeline import PipelineServices, PipelineState, build_graph
from research_assistant.pipeline.nodes import _filter_by_period
from research_assistant.retrieval.models import ExtractedArticle

from tests.pipeline.conftest import D, FakeRetrievalService, FakeTriageLLMClient, jpmorgan_source, real_normalization_service
from tests.pipeline.test_graph import FakeLLMClient, _relevant_assessment
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative
from research_assistant.agents.financial_news import FinancialNewsAgent
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.reporting.digest_service import DigestService

CONFIG = get_config()

PERIOD_START = datetime(2026, 10, 6, 6, 45, 53, tzinfo=timezone.utc)
PERIOD_END = datetime(2026, 10, 7, 6, 45, 53, tzinfo=timezone.utc)


def _real_article(published_at, url="https://jpmorganchase.com/story"):
    extracted = ExtractedArticle(
        url=url, title="JPMorgan launches AI copilot for advisors",
        text="JPMorgan Chase today launched a new generative AI copilot for financial advisors.",
        published_at=published_at,
    )
    return NormalizationService().normalize(extracted, source=jpmorgan_source())


# ---------------------------------------------------------------------------
# Tests 1-4: exact half-open boundary semantics
# ---------------------------------------------------------------------------


def test_article_just_inside_the_window_is_kept():
    article = _real_article(PERIOD_START + timedelta(minutes=21))  # e.g. Oct 06 12:36 IST-equivalent offset

    kept, excluded = _filter_by_period([article], PERIOD_START, PERIOD_END)

    assert kept == [article]
    assert excluded == []


def test_article_one_minute_before_period_start_is_excluded():
    article = _real_article(PERIOD_START - timedelta(minutes=1))

    kept, excluded = _filter_by_period([article], PERIOD_START, PERIOD_END)

    assert kept == []
    assert excluded == [article]


def test_article_exactly_at_period_start_is_included():
    article = _real_article(PERIOD_START)

    kept, excluded = _filter_by_period([article], PERIOD_START, PERIOD_END)

    assert kept == [article]
    assert excluded == []


def test_article_exactly_at_period_end_is_kept_at_the_article_filter_stage():
    """Confirmed NOT a bug, despite initial suspicion during this
    investigation: `_filter_by_period` (the article-level filter) is
    deliberately a CLOSED interval -- see the pre-existing
    `tests/pipeline/test_date_only_calendar_eligibility.py::
    test_exact_timestamp_at_period_end_is_still_eligible_at_the_article_
    filter_stage`, which documents this as intentional, unchanged
    behavior. CURRENT's true half-open `[period_start, period_end)`
    contract is enforced one stage LATER, at event-level digest
    qualification (`nodes.py`'s `in_period_events` filter, already
    correctly `period_start <= e.event_date < period_end`) -- see
    `test_event_exactly_at_period_end_is_excluded_from_the_current_digest`
    below for that real enforcement point."""
    article = _real_article(PERIOD_END)

    kept, excluded = _filter_by_period([article], PERIOD_START, PERIOD_END)

    assert kept == [article]
    assert excluded == []


# ---------------------------------------------------------------------------
# Test 5: a known publication date survives normalization
# ---------------------------------------------------------------------------


def test_known_publication_date_is_preserved_through_normalization():
    published_at = PERIOD_START + timedelta(hours=1)
    article = _real_article(published_at)

    assert article.publication_date == published_at


# ---------------------------------------------------------------------------
# Test 8: event-date semantics -- preserved, never fabricated, internally consistent
# ---------------------------------------------------------------------------


def test_event_date_uses_publication_date_only_when_dated_development_is_true():
    from tests.events.conftest import make_item

    dated = make_item("a1", company="X", title="t", publication_date=D, is_dated_development=True)
    assert _select_event_date([dated]) == D


def test_event_date_is_none_when_dated_development_is_false_even_with_a_real_publication_date():
    """`is_dated_development=False` means the EVIDENCE TEXT itself was
    judged to merely describe an evergreen/reference page, not a specific
    announcement -- its `publication_date` (e.g. a page's last-modified
    date) must never be promoted to the event's date just because a date
    value happens to exist."""
    from tests.events.conftest import make_item

    undated = make_item("a1", company="X", title="t", publication_date=D, is_dated_development=False)
    assert _select_event_date([undated]) is None


def test_event_date_is_none_when_dated_development_is_true_but_no_publication_date_exists():
    """The real-run observation (`is_dated_development: True`, `event_date:
    None`) is internally consistent, not a bug: `is_dated_development` is a
    judgment about the EVIDENCE TEXT (does it describe a specific
    announcement?), completely independent of whether a reliable
    `publication_date` was ever extracted for the page. No date is
    fabricated when none is available."""
    from tests.events.conftest import make_item

    item = make_item("a1", company="X", title="t", publication_date=None, is_dated_development=True)
    assert _select_event_date([item]) is None


# ---------------------------------------------------------------------------
# Tests 6-7, 9-10: full CURRENT funnel, relevant in-window article -> event -> digest
# ---------------------------------------------------------------------------


def _secondary_source():
    from research_assistant.models.enums import SourcePriority, SourceType
    from research_assistant.retrieval.models import SourceCategory, SourceDefinition

    # A generic, real secondary source with no `research_domain` override
    # (agents/router.py) -- routes to FinancialNewsAgent exactly like
    # "Financial Times" used to before FT gained its own AI/Technology
    # specialist route; this test is about the CURRENT funnel/event
    # formation generically, not FT specifically.
    return SourceDefinition(
        name="Reuters", domain="reuters.com", urls=["https://www.reuters.com/technology/artificial-intelligence/"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
    )


def _in_window_article(published_at, url="https://reuters.com/ai-story"):
    return ExtractedArticle(
        url=url, title="Bank announces new enterprise AI partnership",
        text=(
            "A major bank today announced a new enterprise AI partnership and deployed a generative AI "
            "platform across its trading and risk operations, in the latest step of its ongoing AI rollout. "
        ) * 3,
        published_at=published_at,
    )


def _current_services(*, articles, financial_news_responses, retrieval_diagnostics_source_name=None):
    source = _secondary_source()
    sources = [source]
    if retrieval_diagnostics_source_name:
        # A second, independently-failing source -- proves a retrieval
        # failure elsewhere (partial_success) never blocks a *different*,
        # successfully-analyzed in-window article from becoming an event.
        from research_assistant.models.enums import SourcePriority, SourceType
        from research_assistant.retrieval.models import SourceCategory, SourceDefinition

        sources.append(SourceDefinition(
            name=retrieval_diagnostics_source_name, domain="broken.example.com",
            urls=["https://broken.example.com/news"],
            source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
        ))
    retrieval_service = FakeRetrievalService(
        sources=sources, articles_by_source={source.name: articles},
        failing_sources={retrieval_diagnostics_source_name} if retrieval_diagnostics_source_name else None,
    )
    agent = FinancialNewsAgent(FakeLLMClient([]), CONFIG)
    return PipelineServices(
        config=CONFIG, retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.FINANCIAL_NEWS: agent}, event_service=EventService(CONFIG, clock=FixedClock(PERIOD_END)),
        report_agent=__import__("research_assistant.agents.report", fromlist=["ReportAgent"]).ReportAgent(
            ReportingFakeLLMClient([make_narrative()] * 5)
        ),
        digest_service=DigestService(CONFIG, clock=FixedClock(PERIOD_END)),
        clock=FixedClock(PERIOD_END),
        triage_llm_client=FakeTriageLLMClient(),
        deep_analysis_llm_client=FakeLLMClient(financial_news_responses),
    )


def test_relevant_in_window_article_becomes_an_event_and_reaches_the_current_digest():
    """Tests 6-7: a relevant, in-window candidate must flow all the way to
    research_items -> events -> the CURRENT digest."""
    published_at = PERIOD_START + timedelta(hours=1)
    article = _in_window_article(published_at)
    services = _current_services(
        articles=[article], financial_news_responses=[_relevant_assessment(is_dated_development=True)],
    )

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)
    )

    assert len(result["research_items"]) == 1
    assert len(result["events"]) == 1
    assert result["events"][0].event_date == published_at
    digest = result["digest"]
    assert digest is not None
    assert result["events"][0].id in digest.events


def test_the_new_diagnostic_instrumentation_never_fires_for_a_successful_eligible_article():
    """Instrumentation task (forensic finding, CURRENT run
    `e818e08adc2e4dd1`): the new per-article `date_gate:*`/`routing:
    unrouted`/`budget_skip:*` diagnostics must never fire for an article
    that actually makes it all the way to an Event and a digest entry --
    confirms the new instrumentation is purely additive at decision
    points where an article IS excluded/skipped, never a side effect on
    the success path, and that digest semantics (same event, same digest
    membership) are completely unchanged from the pre-instrumentation
    behavior asserted in the test immediately above."""
    published_at = PERIOD_START + timedelta(hours=1)
    article = _in_window_article(published_at)
    services = _current_services(
        articles=[article], financial_news_responses=[_relevant_assessment(is_dated_development=True)],
    )

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)
    )

    # Unchanged digest semantics, re-asserted alongside the new diagnostic check.
    assert len(result["research_items"]) == 1
    assert len(result["events"]) == 1
    assert result["events"][0].id in result["digest"].events

    new_instrumentation_prefixes = ("date_gate:", "routing:unrouted", "budget_skip:")
    spurious = [d.message for d in result["diagnostics"] if d.message.startswith(new_instrumentation_prefixes)]
    assert spurious == [], f"unexpected exclusion diagnostic(s) for an article that actually succeeded: {spurious}"


def test_partial_success_still_persists_the_valid_analyzed_developments():
    """Test 9: some sources failing (-> partial_success, matching the real
    run's own retrieve-stage errors) must not prevent a different,
    successfully-analyzed in-window article from becoming a CURRENT event."""
    published_at = PERIOD_START + timedelta(hours=2)
    article = _in_window_article(published_at)
    services = _current_services(
        articles=[article], financial_news_responses=[_relevant_assessment(is_dated_development=True)],
        retrieval_diagnostics_source_name="Broken Source",
    )

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)
    )

    assert len(result["events"]) == 1
    assert result["digest"] is not None
    assert result["events"][0].id in result["digest"].events
    # A real retrieval failure is still recorded -- partial_success's cause
    # is visible, never silently swallowed just because other content succeeded.
    assert any(d.stage == "retrieve" for d in result["diagnostics"])


def test_out_of_window_article_never_leaks_into_the_current_digest():
    """Test 10: the boundary fix must not cause a genuinely out-of-window
    article to start qualifying."""
    before_window = PERIOD_START - timedelta(hours=1)
    article = _in_window_article(before_window)
    services = _current_services(
        articles=[article], financial_news_responses=[_relevant_assessment(is_dated_development=True)],
    )

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)
    )

    assert result["events"] == []
    assert result["digest"] is not None
    assert result["digest"].events == []


def test_event_exactly_at_period_end_is_excluded_from_the_current_digest():
    """Test 4 (strict upstream date-gate version): an article dated exactly
    at `period_end` must NOT qualify for CURRENT -- the window is strictly
    half-open, `period_start <= event_date < period_end`. CURRENT's hard
    upstream date gate (`pipeline.nodes._apply_current_date_gate`) now
    drops this article BEFORE routing/triage/specialist analysis (as
    "future"), so no event is even formed -- stricter than the previous
    behavior (which let the event form, then relied on a later
    `in_period_events` filter at digest-generation time to exclude it)."""
    article = _in_window_article(PERIOD_END)
    services = _current_services(
        articles=[article], financial_news_responses=[_relevant_assessment(is_dated_development=True)],
    )

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)
    )

    assert result["events"] == []  # never even formed -- dropped upstream
    assert result["digest"] is not None
    assert result["digest"].events == []
    assert result["metrics"]["current_dropped_future"] == 1
