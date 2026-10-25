"""CURRENT's hard upstream date gate (`pipeline.nodes._apply_current_date_gate`):
an undated, stale, or future-dated article must never reach relevance
processing, routing, Qwen triage, specialist analysis, ResearchItem
creation, Event creation, or the CURRENT digest. Confirmed real-production
defect this closes (2026-10-07 live CURRENT run `2a55bf02ff1b7d1f`): 279 of
300 "in-window" articles were actually undated, yet all flowed through
evidence/routing/the bounded triage pool before being dropped only at
event formation. Fixtures/`FakeTriageLLMClient`/`FakeLLMClient` only -- no
live network, no Ollama."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.agents.financial_news import FinancialNewsAgent
from research_assistant.config import get_config
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.evidence.service import NormalizationService
from research_assistant.models.article import DatePrecision
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, SourcePriority, SourceType
from research_assistant.pipeline import PipelineServices, PipelineState, build_graph
from research_assistant.pipeline.nodes import _apply_current_date_gate, make_agent_analysis_node
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.extractor import extract_article
from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition

from tests.pipeline.conftest import D, FakeRetrievalService, FakeTriageLLMClient, jpmorgan_source, real_normalization_service
from tests.pipeline.test_graph import FakeLLMClient, _relevant_assessment
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative

CONFIG = get_config()

PERIOD_START = datetime(2026, 10, 6, 6, 45, 53, tzinfo=timezone.utc)
PERIOD_END = datetime(2026, 10, 7, 6, 45, 53, tzinfo=timezone.utc)


def _reuters_source() -> SourceDefinition:
    """A generic, real secondary source with no `research_domain` override
    (agents/router.py) -- routes to FinancialNewsAgent, unlike "Financial
    Times"/"PR Newswire" which now have their own specialist routes."""
    return SourceDefinition(
        name="Reuters", domain="reuters.com", urls=["https://www.reuters.com/technology/artificial-intelligence/"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
    )


def _article(published_at, *, precision=DatePrecision.EXACT, url="https://reuters.com/story", title="A real headline about a specific development"):
    extracted = ExtractedArticle(
        url=url, title=title,
        text="Reuters reported that a major bank today announced a new generative AI platform for financial advisors. " * 5,
        published_at=published_at,
    )
    article = NormalizationService().normalize(extracted, source=_reuters_source())
    if published_at is None:
        return article.model_copy(update={"publication_date": None, "publication_date_precision": DatePrecision.UNKNOWN})
    return article.model_copy(update={"publication_date_precision": precision})


# ---------------------------------------------------------------------------
# 1. Direct unit tests of the gate itself
# ---------------------------------------------------------------------------


def test_undated_article_is_dropped():
    article = _article(None)
    kept, counts = _apply_current_date_gate([article], PERIOD_START, PERIOD_END)
    assert kept == []
    assert counts == {"undated": 1, "stale": 0, "future": 0}


def test_month_precision_article_is_treated_as_untrustworthy_not_stale_or_future():
    """A MONTH-precision date cannot be verified against an exact 24h
    window -- it is reported as "undated" (untrustworthy), never
    mislabeled as stale/future."""
    article = _article(PERIOD_START, precision=DatePrecision.MONTH)
    kept, counts = _apply_current_date_gate([article], PERIOD_START, PERIOD_END)
    assert kept == []
    assert counts["undated"] == 1


def test_stale_article_before_period_start_is_dropped():
    article = _article(PERIOD_START - timedelta(hours=1))
    kept, counts = _apply_current_date_gate([article], PERIOD_START, PERIOD_END)
    assert kept == []
    assert counts == {"undated": 0, "stale": 1, "future": 0}


def test_future_dated_article_at_or_after_period_end_is_dropped():
    article = _article(PERIOD_END)
    kept, counts = _apply_current_date_gate([article], PERIOD_START, PERIOD_END)
    assert kept == []
    assert counts == {"undated": 0, "stale": 0, "future": 1}


def test_article_exactly_at_period_start_is_accepted():
    article = _article(PERIOD_START)
    kept, counts = _apply_current_date_gate([article], PERIOD_START, PERIOD_END)
    assert kept == [article]
    assert counts == {"undated": 0, "stale": 0, "future": 0}


def test_article_one_microsecond_before_period_start_is_dropped_as_stale():
    article = _article(PERIOD_START - timedelta(microseconds=1))
    kept, counts = _apply_current_date_gate([article], PERIOD_START, PERIOD_END)
    assert kept == []
    assert counts["stale"] == 1


def test_in_window_article_is_accepted():
    article = _article(PERIOD_START + timedelta(hours=1))
    kept, counts = _apply_current_date_gate([article], PERIOD_START, PERIOD_END)
    assert kept == [article]
    assert counts == {"undated": 0, "stale": 0, "future": 0}


def test_date_only_precision_in_window_is_trustworthy_and_accepted():
    article = _article(PERIOD_START + timedelta(hours=1), precision=DatePrecision.DATE_ONLY)
    kept, counts = _apply_current_date_gate([article], PERIOD_START, PERIOD_END)
    assert kept == [article]


# ---------------------------------------------------------------------------
# 2. End-to-end: downstream isolation -- an undated/stale/future article
# never reaches routing, triage, specialist analysis, ResearchItem, or Event
# ---------------------------------------------------------------------------


def _current_services(*, articles, financial_news_responses):
    source = _reuters_source()
    retrieval_service = FakeRetrievalService(sources=[source], articles_by_source={source.name: articles})
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


def test_undated_article_never_reaches_routing_triage_or_specialist_analysis():
    undated = ExtractedArticle(
        url="https://reuters.com/undated-story",
        title="A major bank announces new enterprise AI deal",
        text="Reuters reported that a major bank today announced a new enterprise AI deal with a major vendor. " * 5,
        published_at=None,
    )
    services = _current_services(articles=[undated], financial_news_responses=[_relevant_assessment()])

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)
    )

    metrics = result["metrics"]
    assert metrics["current_dropped_undated"] == 1
    assert metrics["routed_financial_news"] == 0  # never reached routing
    assert metrics["triage_candidates"] == 0  # never reached triage
    assert metrics["deep_analysis_attempted"] == 0  # never reached specialist analysis
    assert result["research_items"] == []
    assert result["events"] == []
    assert result["digest"].events == []


def _run_agent_analysis_node_directly(services, *, articles):
    """Calls the `agent_analysis` node function directly with a hand-built
    `state.articles`, bypassing `retrieve`/`normalize` -- those nodes
    already apply their OWN wide-lookback `_filter_by_period` (using
    `period_end` as the SAME upper bound CURRENT uses), which would
    independently exclude a stale/future-dated article before it ever
    reached `agent_analysis` at all in a full-graph run. Calling the node
    directly isolates and proves `_apply_current_date_gate` ITSELF (inside
    `agent_analysis`) also independently rejects it -- defense in depth,
    not reliance on the earlier node alone."""
    from research_assistant.evidence.conversion import article_to_evidence

    evidence = [article_to_evidence(a) for a in articles]
    node = make_agent_analysis_node(services)
    state = PipelineState(
        digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END,
        articles=articles, evidence=evidence,
    )
    return node(state)


def test_stale_article_never_reaches_routing_triage_or_specialist_analysis():
    """A far-stale date is actually caught TWICE in the real node: the
    pre-existing `_filter_by_period` (its own, already-correct
    `period_start <= pub_date <= period_end` check) excludes it before
    `_apply_current_date_gate` even runs, so `period_eligible_articles` is
    already empty here -- confirmed by `current_date_qualified == 0` with
    zero drops attributed to the NEW gate specifically. The gate's OWN
    stale-detection logic (for an article that reaches it) is proven
    directly in section 1's unit tests above
    (`test_stale_article_before_period_start_is_dropped`)."""
    stale = _article(PERIOD_START - timedelta(days=10), url="https://reuters.com/stale-story")
    services = _current_services(articles=[], financial_news_responses=[_relevant_assessment()])

    result = _run_agent_analysis_node_directly(services, articles=[stale])

    metrics = result["metrics"]
    assert metrics["current_date_qualified"] == 0
    assert metrics["routed_financial_news"] == 0
    assert metrics["triage_candidates"] == 0
    assert metrics["deep_analysis_attempted"] == 0
    assert result["research_items"] == []


def test_future_dated_article_never_reaches_routing_triage_or_specialist_analysis():
    """Same structural note as the stale case above: a far-future date is
    already excluded by the pre-existing `_filter_by_period` before
    `_apply_current_date_gate` runs. The gate's OWN future-detection logic
    (and the exact-at-`period_end` boundary case `_filter_by_period`'s
    inclusive check does NOT catch) is proven directly in section 1's unit
    tests (`test_future_dated_article_at_or_after_period_end_is_dropped`)
    and in `tests/pipeline/test_current_boundary_and_event_formation.py`'s
    full-graph boundary test."""
    future = _article(PERIOD_END + timedelta(days=1), url="https://reuters.com/future-story")
    services = _current_services(articles=[], financial_news_responses=[_relevant_assessment()])

    result = _run_agent_analysis_node_directly(services, articles=[future])

    metrics = result["metrics"]
    assert metrics["current_date_qualified"] == 0
    assert metrics["routed_financial_news"] == 0
    assert metrics["triage_candidates"] == 0
    assert result["research_items"] == []


# ---------------------------------------------------------------------------
# 3. End-to-end propagation: a genuinely dated, in-window, relevant article
# survives all the way into the CURRENT digest; an undated one never does
# ---------------------------------------------------------------------------


def test_dated_in_window_relevant_article_survives_into_the_current_digest():
    dated = ExtractedArticle(
        url="https://reuters.com/dated-story",
        title="A major bank announces new enterprise AI partnership",
        text="Reuters reported that a major bank today announced a new enterprise AI partnership with a major vendor. " * 5,
        published_at=PERIOD_START + timedelta(hours=2),
    )
    services = _current_services(
        articles=[dated], financial_news_responses=[_relevant_assessment(is_dated_development=True)],
    )

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)
    )

    assert result["metrics"]["current_dropped_undated"] == 0
    assert len(result["research_items"]) == 1
    assert len(result["events"]) == 1
    event = result["events"][0]
    # The required invariant, explicit: a real, non-null event_date inside
    # the exact CURRENT window -- never scan time, never "today", never
    # fabricated.
    assert event.event_date is not None
    assert PERIOD_START <= event.event_date < PERIOD_END
    assert event.id in result["digest"].events


def test_relevant_but_undated_article_never_reaches_the_digest():
    undated = ExtractedArticle(
        url="https://reuters.com/undated-story-2",
        title="A major bank announces new enterprise AI partnership",
        text="Reuters reported that a major bank today announced a new enterprise AI partnership with a major vendor. " * 5,
        published_at=None,
    )
    services = _current_services(
        articles=[undated], financial_news_responses=[_relevant_assessment(is_dated_development=True)],
    )

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)
    )

    assert result["research_items"] == []
    assert result["events"] == []
    assert result["digest"].events == []


# ---------------------------------------------------------------------------
# 4. Morning/Evening are completely unaffected by the new gate
# ---------------------------------------------------------------------------


def test_morning_digest_type_never_applies_the_current_date_gate():
    undated = ExtractedArticle(
        url="https://reuters.com/undated-morning-story",
        title="A major bank announces new enterprise AI partnership",
        text="Reuters reported that a major bank today announced a new enterprise AI partnership with a major vendor. " * 5,
        published_at=None,
    )
    source = _reuters_source()
    retrieval_service = FakeRetrievalService(sources=[source], articles_by_source={source.name: [undated]})
    # Morning/Evening never use the CURRENT-only triage/deep-analysis
    # client override -- `agent.analyze()` uses the agent's OWN `_llm`
    # client directly, unlike `_current_services`' CURRENT-shaped wiring.
    agent = FinancialNewsAgent(FakeLLMClient([_relevant_assessment(is_dated_development=True)]), CONFIG)
    services = PipelineServices(
        config=CONFIG, retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.FINANCIAL_NEWS: agent}, event_service=EventService(CONFIG, clock=FixedClock(PERIOD_END)),
        report_agent=__import__("research_assistant.agents.report", fromlist=["ReportAgent"]).ReportAgent(
            ReportingFakeLLMClient([make_narrative()] * 5)
        ),
        digest_service=DigestService(CONFIG, clock=FixedClock(PERIOD_END)),
        clock=FixedClock(PERIOD_END),
    )

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.MORNING, period_start=PERIOD_START, period_end=PERIOD_END)
    )

    # Morning's existing, unchanged lenient behavior: the undated article is
    # still retained and reaches the agent (the pre-existing EventBuilder
    # "no event without a date" guard, not the new CURRENT-only gate, is
    # what ultimately governs whether it becomes an event).
    assert "current_dropped_undated" not in result["metrics"] or result["metrics"].get("current_dropped_undated", 0) == 0
    assert len(result["research_items"]) == 1


# ---------------------------------------------------------------------------
# 5. Concrete regression fixtures: Deloitte/Anthropic-shaped real pages
# (field names/structure taken from the real 2026-10-07 cached pages --
# never matched by title/domain, only by the same generic field names
# `extract_article` already checks for any source)
# ---------------------------------------------------------------------------


def test_deloitte_shaped_hydration_payload_now_yields_a_trustworthy_date():
    """Real structure confirmed on the live Deloitte page from the
    2026-10-07 CURRENT run: a non-JSON-LD `dataLayer.page = {...}`
    tag-manager payload carries `datePublished` alongside `createDate`/
    `dateModified`. Matched generically by field name, never by this
    specific page's title/domain."""
    html = (
        "<html><head></head><body><article><p>"
        + "Substantive research report body text. " * 20
        + "</p></article>"
        '<script type="text/javascript">var dataLayer = {}; dataLayer.page = '
        '{"createDate": "14 Jan 2026", "dateModified": "23 Apr 2026", '
        '"datePublished": "20 Jan 2026"};</script>'
        "</body></html>"
    )
    result = extract_article(html, "https://example.test/content/report")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-01-20"


def test_anthropic_shaped_page_with_no_trustworthy_date_stays_undated():
    """Real structure confirmed on the live Anthropic page from the
    2026-10-07 CURRENT run: the only visible date signal is a bare
    "Month YYYY" string with no explicit publication label and no
    structured metadata anywhere -- this must stay undated rather than
    guessing from an unreliable, generated CSS class name."""
    html = (
        "<html><head></head><body>"
        '<p class="LaunchHero-module-scss-module__RCsuha__date">September 2026</p>'
        "<h1>Introducing a new model family</h1>"
        "<article><p>" + "Substantive announcement body text. " * 20 + "</p></article>"
        "</body></html>"
    )
    result = extract_article(html, "https://example.test/new-model-family")
    assert result.published_at is None


def test_deloitte_shaped_article_qualifies_for_current_when_the_recovered_date_is_in_window():
    """End-to-end: once the hydration-payload date is recovered, a Deloitte-
    shaped article with that date actually INSIDE the CURRENT window
    qualifies exactly like any other dated article -- no special-casing by
    title."""
    html = (
        "<html><head></head><body><article><p>"
        + "Substantive research report body text about enterprise AI adoption. " * 10
        + "</p></article>"
        '<script type="text/javascript">var dataLayer = {}; dataLayer.page = '
        '{"datePublished": "2026-10-06T12:00:00Z"};</script>'
        "</body></html>"
    )
    extracted = extract_article(html, "https://example.test/content/report-in-window")
    article = NormalizationService().normalize(extracted, source=jpmorgan_source())

    kept, counts = _apply_current_date_gate([article], PERIOD_START, PERIOD_END)
    assert kept == [article]
    assert counts == {"undated": 0, "stale": 0, "future": 0}


# ---------------------------------------------------------------------------
# 6. NVIDIA regression fixture: a real GA4 `PublishDate` event call
# (2026-10-07 CURRENT run `857d24ceac63b4ce`) recovers a real but STALE
# (September 28) date -- full end-to-end proof it is dropped BEFORE
# routing/triage/specialist analysis, never reaching Event/digest.
# ---------------------------------------------------------------------------


def _nvidia_shaped_html() -> str:
    return (
        "<html><head></head><body><article><p>"
        "NVIDIA today announced NVIDIA Open Agent Safety Platform, an open "
        "software platform and reference system design to strengthen AI "
        "security from agent testing to deployment. " * 5
        + "</p></article>"
        "<script>ga_four_event('PublishDate','View','September 28, 2026');</script>"
        "</body></html>"
    )


def test_nvidia_fixture_date_is_recovered_as_september_28():
    extracted = extract_article(_nvidia_shaped_html(), "https://nvidianews.nvidia.com/news/open-agent-safety-platform")
    assert extracted.published_at is not None
    assert extracted.published_at.date().isoformat() == "2026-09-28"


def test_nvidia_fixture_is_stale_for_an_october_6_to_7_current_window():
    """The exact CURRENT period from the real production run this fixture
    reproduces: 2026-10-06T16:43:27 -> 2026-10-07T16:43:27. A September 28
    publication date is well before `period_start` -- STALE, dropped."""
    october_period_start = datetime(2026, 10, 6, 16, 43, 27, tzinfo=timezone.utc)
    october_period_end = datetime(2026, 10, 7, 16, 43, 27, tzinfo=timezone.utc)

    extracted = extract_article(_nvidia_shaped_html(), "https://nvidianews.nvidia.com/news/open-agent-safety-platform")
    article = NormalizationService().normalize(extracted, source=_reuters_source())

    kept, counts = _apply_current_date_gate([article], october_period_start, october_period_end)
    assert kept == []
    assert counts == {"undated": 0, "stale": 1, "future": 0}


def test_nvidia_fixture_never_reaches_routing_triage_specialist_or_event_end_to_end():
    """Full end-to-end: the NVIDIA-shaped article, with its real (recovered)
    but stale September 28 date, must never reach routing, triage,
    specialist analysis, ResearchItem creation, Event creation, or the
    CURRENT digest -- confirmed real production defect (run
    `857d24ceac63b4ce`) this regression test guards against."""
    extracted_data = extract_article(_nvidia_shaped_html(), "https://nvidianews.nvidia.com/news/open-agent-safety-platform")
    nvidia_article = ExtractedArticle(
        url="https://nvidianews.nvidia.com/news/open-agent-safety-platform",
        title="NVIDIA Launches Open Agent Safety Platform to Secure Agents From Testing to Deployment",
        text=extracted_data.text,
        published_at=extracted_data.published_at,  # the real, recovered (but stale) date -- never fabricated
    )
    services = _current_services(articles=[], financial_news_responses=[_relevant_assessment()])

    article = NormalizationService().normalize(
        ExtractedArticle(
            url=nvidia_article.url, title=nvidia_article.title, text=nvidia_article.text,
            published_at=nvidia_article.published_at,
        ),
        source=_reuters_source(),
    )
    result = _run_agent_analysis_node_directly(services, articles=[article])

    # A September-28 date is far outside the Oct 6-7 window, so the
    # pre-existing `_filter_by_period` (unchanged) already excludes it one
    # step before `_apply_current_date_gate` runs -- see
    # `test_stale_article_never_reaches_routing_triage_or_specialist_analysis`
    # for the direct proof that the gate ITSELF also correctly classifies
    # this as "stale" when given the chance to see it
    # (`test_nvidia_fixture_is_stale_for_an_october_6_to_7_current_window`
    # above, via `_apply_current_date_gate` directly). What matters here is
    # the full end-to-end guarantee: zero candidates ever qualify.
    assert result["metrics"]["current_date_qualified"] == 0
    assert result["metrics"]["routed_financial_news"] == 0
    assert result["metrics"]["triage_candidates"] == 0
    assert result["metrics"]["deep_analysis_attempted"] == 0
    assert result["research_items"] == []


# ---------------------------------------------------------------------------
# Per-article date-gate diagnostics (instrumentation task, forensic finding
# CURRENT run `e818e08adc2e4dd1`: 194 of 201 period-eligible articles were
# dropped here with only an aggregate `current_dropped_undated` count, no
# persisted per-article record of which one or why). `diagnostics` is
# optional and purely additive -- every test above that omits it proves the
# gate's own kept/counts classification is completely unaffected.
# ---------------------------------------------------------------------------


def _date_gate_messages(diagnostics):
    return [d.message for d in diagnostics if d.message.startswith("date_gate:")]


def test_undated_article_emits_a_date_gate_diagnostic_when_diagnostics_list_supplied():
    article = _article(None)
    diagnostics = []
    kept, counts = _apply_current_date_gate([article], PERIOD_START, PERIOD_END, diagnostics=diagnostics)
    assert kept == []
    messages = _date_gate_messages(diagnostics)
    assert len(messages) == 1
    assert f"date_gate:undated candidate={article.id}" in messages[0]
    assert article.url in messages[0]


def test_stale_article_emits_a_date_gate_diagnostic_with_the_stale_reason():
    article = _article(PERIOD_START - timedelta(hours=1))
    diagnostics = []
    _apply_current_date_gate([article], PERIOD_START, PERIOD_END, diagnostics=diagnostics)
    messages = _date_gate_messages(diagnostics)
    assert len(messages) == 1
    assert f"date_gate:stale candidate={article.id}" in messages[0]


def test_future_article_emits_a_date_gate_diagnostic_with_the_future_reason():
    article = _article(PERIOD_END)
    diagnostics = []
    _apply_current_date_gate([article], PERIOD_START, PERIOD_END, diagnostics=diagnostics)
    messages = _date_gate_messages(diagnostics)
    assert len(messages) == 1
    assert f"date_gate:future candidate={article.id}" in messages[0]


def test_a_kept_in_window_article_never_emits_a_date_gate_diagnostic():
    article = _article(PERIOD_START + timedelta(hours=1))
    diagnostics = []
    kept, _counts = _apply_current_date_gate([article], PERIOD_START, PERIOD_END, diagnostics=diagnostics)
    assert kept == [article]
    assert _date_gate_messages(diagnostics) == []


def test_omitting_diagnostics_never_crashes_and_classification_is_unaffected():
    """The default -- every pre-existing call site in this file and in
    `pipeline.nodes` itself -- must behave exactly as before this
    instrumentation was added."""
    article = _article(None)
    kept, counts = _apply_current_date_gate([article], PERIOD_START, PERIOD_END)
    assert kept == []
    assert counts == {"undated": 1, "stale": 0, "future": 0}


def test_date_gate_diagnostics_are_never_duplicated_per_article():
    """One dropped article -> exactly one date_gate diagnostic, never one
    per internal branch/check."""
    article = _article(None)
    diagnostics = []
    _apply_current_date_gate([article], PERIOD_START, PERIOD_END, diagnostics=diagnostics)
    assert len(_date_gate_messages(diagnostics)) == 1


def test_mixed_batch_emits_one_diagnostic_per_dropped_article_and_none_for_kept_ones():
    undated = _article(None, url="https://reuters.com/undated")
    stale = _article(PERIOD_START - timedelta(hours=1), url="https://reuters.com/stale")
    kept_article = _article(PERIOD_START + timedelta(hours=1), url="https://reuters.com/kept")
    diagnostics = []
    kept, counts = _apply_current_date_gate(
        [undated, stale, kept_article], PERIOD_START, PERIOD_END, diagnostics=diagnostics,
    )
    assert kept == [kept_article]
    assert counts == {"undated": 1, "stale": 1, "future": 0}
    messages = _date_gate_messages(diagnostics)
    assert len(messages) == 2
    assert any(f"candidate={undated.id}" in m for m in messages)
    assert any(f"candidate={stale.id}" in m for m in messages)
    assert not any(f"candidate={kept_article.id}" in m for m in messages)


# ---------------------------------------------------------------------------
# Per-article routing-exclusion diagnostics (instrumentation task): an
# in-window, date-qualified CURRENT article that still matches no agent's
# routing rules must get a `routing:unrouted` diagnostic.
# ---------------------------------------------------------------------------


def _unrouted_source() -> SourceDefinition:
    """NOT a member of any configured peer/ai_provider/analyst/secondary
    registry -- mirrors `tests/agents/test_router.py`'s own "genuinely
    unrouted" fixture source."""
    return SourceDefinition(
        name="Random Blog", domain="random-blog.example.com", urls=["https://random-blog.example.com/"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.LOW, category=SourceCategory.SECONDARY,
    )


def test_an_unrouted_in_window_article_emits_a_routing_diagnostic():
    extracted = ExtractedArticle(
        url="https://random-blog.example.com/weather-report",
        title="Weather report", text="It rained heavily across the region today. " * 10,
        published_at=PERIOD_START + timedelta(hours=1),
    )
    article = NormalizationService().normalize(extracted, source=_unrouted_source())
    services = _current_services(articles=[], financial_news_responses=[_relevant_assessment()])

    result = _run_agent_analysis_node_directly(services, articles=[article])

    assert result["metrics"]["current_date_qualified"] == 1
    assert result["metrics"]["routed_unrouted"] == 1
    routing_messages = [d.message for d in result["diagnostics"] if d.message.startswith("routing:unrouted")]
    assert len(routing_messages) == 1
    assert f"candidate={article.id}" in routing_messages[0]
    assert article.url in routing_messages[0]
    # Genuinely unrouted -- never reaches triage/deep analysis either.
    assert result["metrics"]["triage_candidates"] == 0
    assert result["metrics"]["deep_analysis_attempted"] == 0


def test_a_routed_in_window_article_never_emits_a_routing_diagnostic():
    """Negative control: Reuters (a real, registered secondary source)
    always routes to FinancialNewsAgent -- it must never get a spurious
    `routing:unrouted` diagnostic."""
    article = _article(PERIOD_START + timedelta(hours=1))
    services = _current_services(articles=[], financial_news_responses=[_relevant_assessment()])

    result = _run_agent_analysis_node_directly(services, articles=[article])

    assert result["metrics"]["routed_unrouted"] == 0
    routing_messages = [d.message for d in result["diagnostics"] if d.message.startswith("routing:unrouted")]
    assert routing_messages == []
