"""Production-readiness end-to-end regression (final pass): proves a real,
relevant, dated article can travel the COMPLETE CURRENT pipeline --

  discovery -> date gate -> evidence -> routing -> triage (Qwen) ->
  deep analysis (Qwen) -> ResearchItem -> Event -> digest qualification ->
  persistence -> user-facing CURRENT UI read path

-- using the REAL `discover_candidates`/`extract_article`/`normalize_article`
implementations, the REAL LangGraph orchestration (`PipelineRunner`, the
exact entry point the admin "Run Live Current Scan" button calls), and a
REAL temp-file SQLite database, with ONLY the two genuinely external
dependencies faked: network (a listing-page/article-page HTML fixture
instead of a live fetch) and Ollama (deterministic `FakeTriageLLMClient`/
`FakeLLMClient`, never a real model call).

The regression fixture is the real PR Newswire JumpCloud release (the
forensic investigation's own subject). Its text is a trimmed, verbatim
copy of what the real page actually served (verified against `data/cache`
during that investigation) -- not invented, and not special-cased in any
production code path: `AIProductAgent`/`StartupAgent`/the router/the date
gate treat it exactly like any other candidate.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from research_assistant.agents.ai_product import AIProductAgent
from research_assistant.agents.report import ReportAgent
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.agents.startup import StartupAgent
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.evidence.service import NormalizationService
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, EntityType
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
    TypedEntity,
)
from research_assistant.pipeline import PipelineServices
from research_assistant.pipeline.runner import PipelineRunner
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.discovery import discover_candidates
from research_assistant.retrieval.extractor import extract_article
from research_assistant.retrieval.service import build_source_definitions
from research_assistant.storage import Database, PersistenceService
from research_assistant.ui import dashboard_data as dd

from tests.pipeline.conftest import FakeTriageLLMClient
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative
from tests.retrieval.fakes import FakeResponse, FakeSession

JUMPCLOUD_URL = (
    "https://www.prnewswire.com/news-releases/"
    "new-jumpcloud-research-finds-ai-access-is-outpacing-enterprise-governance-302901426.html"
)
JUMPCLOUD_TITLE = "New JumpCloud Research Finds AI Access Is Outpacing Enterprise Governance"
AI_TOPIC_URL = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"

# Real listing-page shape (category links alongside the real release) --
# proves discovery survives the exact bug class the PR Newswire fix
# addressed, inside this same end-to-end run.
_AI_TOPIC_PAGE_HTML = f"""
<html><body><main>
  <a href="{JUMPCLOUD_URL}">{JUMPCLOUD_TITLE}</a>
  <a href="/news-releases/latest-news-topics/artificial-intelligence/">All AI News and Press Releases</a>
  <a href="/news-releases/financial-services-latest-news/venture-capital-list/">All Venture Capital News</a>
</main></body></html>
"""

# Trimmed, verbatim copy of the real JSON-LD + dateline this URL actually
# served (confirmed against `data/cache` during the forensic
# investigation) -- includes the real "today released" phrasing whose
# missing deterministic pattern was this task's actual production fix.
_JUMPCLOUD_ARTICLE_HTML = f"""<html><head>
<title>{JUMPCLOUD_TITLE}</title>
<script type="application/ld+json">
        {{
            "@context": "https://schema.org",
            "@type": "NewsArticle",
            "headline": "{JUMPCLOUD_TITLE}",
            "datePublished": "2026-10-07T13:04:00-04:00",
            "dateModified": "2026-10-07T13:05:23-04:00",
            "publisher": {{"@type": "Organization", "name": "Cision PR Newswire"}}
        }}
    </script>
</head>
<body><article>
<h1>{JUMPCLOUD_TITLE}</h1>
<p>LOUISVILLE, Colo., Oct. 7, 2026 /PRNewswire/ -- JumpCloud Inc. today released a new study, Enterprise AI Access Is Outpacing Control. The report shows that organizations are giving AI agents access to systems faster than they can secure them. More than half of organizations now use AI agents that can change systems, permissions, or files.</p>
<p>The study of 250 IT leaders across the U.S. and U.K. shows how organizations are managing AI as it gains greater access across the enterprise.</p>
</article></body></html>"""

# CURRENT window: a real 24h period containing the real publication instant
# (2026-10-07T17:04:00Z).
PERIOD_END = datetime(2026, 10, 7, 20, 0, tzinfo=timezone.utc)
PERIOD_START = PERIOD_END - timedelta(hours=24)


def _discover_jumpcloud_candidate():
    """Real `discover_candidates()` against a realistic listing page --
    proves the PR Newswire discovery fix and this end-to-end regression
    are the same, unregressed code path."""
    session = FakeSession([FakeResponse(status_code=200, url=AI_TOPIC_URL, text=_AI_TOPIC_PAGE_HTML)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    config = __import__("research_assistant.config", fromlist=["get_config"]).get_config()
    pr_newswire = next(s for s in build_source_definitions(config) if s.name == "PR Newswire")
    candidates = discover_candidates(pr_newswire, http_client=client, max_candidates=50)
    jumpcloud = next(c for c in candidates if c.url == JUMPCLOUD_URL)
    return pr_newswire, jumpcloud


def _extract_jumpcloud_article():
    """Real `extract_article()` against the real (trimmed) page content."""
    return extract_article(_JUMPCLOUD_ARTICLE_HTML, JUMPCLOUD_URL)


class _FakeRetrievalServiceForOneSource:
    """Minimal real-shaped retrieval service: `list_sources()` returns the
    REAL configured source registry (so routing/`research_domain` are
    exercised exactly as production would), `retrieve_source()` returns the
    real-extracted JumpCloud article for PR Newswire and nothing for every
    other configured source (no network, no other fixtures needed)."""

    def __init__(self, sources, pr_newswire_name: str, extracted_article):
        self._sources = sources
        self._pr_newswire_name = pr_newswire_name
        self._extracted_article = extracted_article

    def list_sources(self):
        return list(self._sources)

    def retrieve_source(self, source, *, cache_policy=None) -> List:
        if source.name == self._pr_newswire_name:
            return [self._extracted_article]
        return []


class _DeepAnalysisFakeLLMClient:
    """Deterministic stand-in for CURRENT's dedicated deep-analysis Ollama
    client -- never a live model call. Returns a realistic `AgentAssessment`
    for the JumpCloud candidate specifically (matched by checking the
    prompt contains the real evidence text), and raises for anything else
    so an unexpected candidate reaching deep analysis fails the test
    loudly instead of silently returning a canned response.

    `relevant=False` scripts the EXACT scenario this file's override
    regression proves: Qwen itself judges the real JumpCloud content
    IRRELEVANT (a plausible, real small-model mistake) -- the deterministic
    recall safeguard in `agents.base._resolve_relevance_override` must
    still let it through, since it is genuinely dated and genuinely
    touches >=2 scope categories."""

    def __init__(self, *, relevant: bool = True):
        self.calls = []
        self._relevant = relevant

    def generate_structured(self, *, system, prompt, schema):
        self.calls.append(prompt)
        assert "JumpCloud" in prompt, "only the JumpCloud candidate should reach deep analysis in this fixture"
        label = RelevanceLabel.RELEVANT if self._relevant else RelevanceLabel.IRRELEVANT
        reason = (
            "JumpCloud research quantifies a real governance gap in enterprise AI agent access."
            if self._relevant else
            "Not significant enough to warrant a dedicated research item."
        )
        return AgentAssessment(
            relevance=RelevanceDecision(label=label, reason=reason),
            category=TopicCategory.ENTERPRISE_AI,
            summary="JumpCloud released research finding AI agent access is outpacing governance controls.",
            is_dated_development=self._relevant,
            entities=[],
            people=[],
            products=["JumpCloud"],
            technologies=["AI agents", "IAM"],
            geography=["United States", "United Kingdom"],
            financial_services_relevance=RelevanceAssessment(
                level=RelevanceLevel.MEDIUM, rationale="Enterprise AI governance is relevant to financial-services risk/compliance.",
            ),
            strategic_significance=RelevanceAssessment(
                level=RelevanceLevel.MEDIUM, rationale="Highlights a widening identity/governance gap as AI agents gain access.",
            ),
        )


def _build_end_to_end_services(config, database: Database, *, deep_relevant: bool = True):
    pr_newswire, _jumpcloud_candidate = _discover_jumpcloud_candidate()
    extracted = _extract_jumpcloud_article()
    all_sources = build_source_definitions(config)

    deep_client = _DeepAnalysisFakeLLMClient(relevant=deep_relevant)
    startup_agent = StartupAgent(deep_client, config)  # agent's own ._llm is never used for CURRENT

    return PipelineServices(
        config=config,
        retrieval_service=_FakeRetrievalServiceForOneSource(all_sources, pr_newswire.name, extracted),
        normalization_service=NormalizationService(),
        agents={AgentName.STARTUP: startup_agent},
        event_service=EventService(config, clock=FixedClock(PERIOD_END)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
        digest_service=DigestService(config, clock=FixedClock(PERIOD_END)),
        clock=FixedClock(PERIOD_END),
        persistence_service=PersistenceService(database),
        triage_llm_client=FakeTriageLLMClient(default_relevant=True),
        deep_analysis_llm_client=deep_client,
    ), deep_client


def test_jumpcloud_survives_the_complete_current_pipeline_end_to_end(tmp_path: Path):
    from research_assistant.config import get_config

    config = get_config()
    database = Database(tmp_path / "e2e.db")
    database.initialize()
    services, deep_client = _build_end_to_end_services(config, database)

    runner = PipelineRunner(config, services)
    run_result = runner.run(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)

    counts = run_result.counts

    # --- A/B. Discovery + date gate ---------------------------------------
    assert counts["candidates_discovered"] == 1
    assert counts["current_date_qualified"] == 1
    assert counts["current_dropped_undated"] == 0

    # --- I. Routing (research_domain: startup_funding -> StartupAgent) ---
    assert counts["routed_startup_funding"] == 1
    assert counts["routed_financial_news"] == 0

    # --- K/L. Triage actually ran and marked it relevant ------------------
    assert counts["triage_batches_attempted"] >= 1
    assert counts["triage_relevant"] == 1
    assert counts["triage_irrelevant"] == 0

    # --- M/N. Deep analysis reached and succeeded (the real fix) ---------
    assert counts["deep_analysis_attempted"] == 1
    assert counts["deep_analysis_success"] == 1
    assert counts["deep_analysis_timeouts"] == 0
    assert counts["deep_analysis_failures"] == 0
    assert len(deep_client.calls) == 1

    # --- O. ResearchItem ---------------------------------------------------
    assert counts["research_items_total"] == 1
    assert counts["research_items_relevant"] == 1

    # --- Q. Event, with a real, non-fabricated event_date -----------------
    assert counts["events_formed"] == 1

    # --- S. Digest qualification: the event's date is inside the period --
    assert counts["digest_event_count"] == 1
    assert counts["events_excluded_from_digest_period"] == 0

    # --- R. Persistence: a real run row + real DB rows --------------------
    assert run_result.status.value in ("success", "partial_success")
    assert run_result.digest_persistence_status.value == "persisted"
    assert run_result.digest_id is not None

    conn = database.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM research_items").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM digests").fetchone()[0] == 1
        event_row = conn.execute(
            "SELECT event_date, primary_company FROM events"
        ).fetchone()
        assert event_row[0] is not None  # a real event_date, not NULL
        assert event_row[1] == "PR Newswire"
    finally:
        conn.close()

    # --- T. The user-facing CURRENT UI read path actually surfaces it -----
    persistence_service = services.persistence_service
    view = dd.get_latest_current_intelligence(persistence_service, config)
    assert view.status == "ok"
    assert view.event_count == 1
    assert len(view.events) == 1
    card = view.events[0]
    assert "JumpCloud" in card.canonical_title
    assert card.event_date is not None
    assert PERIOD_START <= card.event_date < PERIOD_END


def test_jumpcloud_event_date_is_correctly_derived_from_its_dated_announcement(tmp_path: Path):
    """Narrower, more targeted assertion: the exact production fix this
    task made (the missing "today released" announcement pattern) is what
    makes the Event's `event_date` non-null -- not any other mechanism."""
    from research_assistant.agents.base import _contains_dated_announcement_language

    extracted = _extract_jumpcloud_article()
    assert extracted.text is not None
    assert _contains_dated_announcement_language(extracted.text) is True


# ---------------------------------------------------------------------------
# Failure-path assertions: an article must not be able to silently
# disappear at any of these stages without a diagnostic/metric reflecting it.
# ---------------------------------------------------------------------------


def test_irrelevant_triage_result_is_visible_in_diagnostics_not_silent(tmp_path: Path):
    from research_assistant.config import get_config

    config = get_config()
    database = Database(tmp_path / "e2e_irrelevant.db")
    database.initialize()
    services, _deep_client = _build_end_to_end_services(config, database)
    services.triage_llm_client = FakeTriageLLMClient(default_relevant=False)

    runner = PipelineRunner(config, services)
    run_result = runner.run(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)

    counts = run_result.counts
    assert counts["triage_relevant"] == 0
    assert counts["triage_irrelevant"] == 1
    assert counts["deep_analysis_attempted"] == 0  # correctly never reached -- not silently lost
    assert counts["events_formed"] == 0
    assert run_result.digest_persistence_status.value == "no_qualifying_events"


def test_undated_article_is_dropped_by_the_hard_gate_with_a_visible_count(tmp_path: Path):
    """The same JumpCloud content, but with no extractable date at all,
    must be dropped by the hard gate -- and that drop must be visible in
    `current_dropped_undated`, never silently absorbed."""
    from research_assistant.config import get_config
    from research_assistant.retrieval.models import ExtractedArticle

    config = get_config()
    database = Database(tmp_path / "e2e_undated.db")
    database.initialize()
    pr_newswire = next(s for s in build_source_definitions(config) if s.name == "PR Newswire")
    all_sources = build_source_definitions(config)
    undated_article = ExtractedArticle(
        url=JUMPCLOUD_URL, title=JUMPCLOUD_TITLE, text="JumpCloud Inc. today released a new study.",
        published_at=None,
    )
    deep_client = _DeepAnalysisFakeLLMClient()
    services = PipelineServices(
        config=config,
        retrieval_service=_FakeRetrievalServiceForOneSource(all_sources, pr_newswire.name, undated_article),
        normalization_service=NormalizationService(),
        agents={AgentName.STARTUP: StartupAgent(deep_client, config)},
        event_service=EventService(config, clock=FixedClock(PERIOD_END)),
        report_agent=ReportAgent(ReportingFakeLLMClient([])),
        digest_service=DigestService(config, clock=FixedClock(PERIOD_END)),
        clock=FixedClock(PERIOD_END),
        persistence_service=PersistenceService(database),
        triage_llm_client=FakeTriageLLMClient(default_relevant=True),
        deep_analysis_llm_client=deep_client,
    )

    runner = PipelineRunner(config, services)
    run_result = runner.run(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)

    counts = run_result.counts
    assert counts["current_date_qualified"] == 0
    assert counts["current_dropped_undated"] == 1
    assert counts["deep_analysis_attempted"] == 0
    assert len(deep_client.calls) == 0


# ---------------------------------------------------------------------------
# Deterministic recall safeguard, end to end: Qwen is allowed to make a
# relevance mistake, but a clearly in-scope, genuinely dated article must
# not disappear solely because of it. A separate, negative regression
# proves an actually-irrelevant article is still correctly rejected.
# ---------------------------------------------------------------------------


def test_jumpcloud_survives_the_pipeline_even_when_qwen_deep_analysis_says_irrelevant(tmp_path: Path):
    """The exact mandated scenario: discovered -> date-qualified -> routed
    to StartupAgent -> triage relevant -> deep analysis executes -> the
    (fake, but real-code-path) Qwen deep assessment says IRRELEVANT -- and
    the system does NOT silently discard the article, because deterministic
    scope evidence (>=2 scope categories + genuine dated-announcement
    language, both independently re-derived from the real evidence text,
    never from whatever Qwen itself claimed) establishes it as clearly
    in-scope. Every subsequent stage must still be the REAL code path: no
    ResearchItem/Event is inserted directly, no date is fabricated, no
    agent's own relevance classification was removed."""
    from research_assistant.config import get_config

    config = get_config()
    database = Database(tmp_path / "e2e_override.db")
    database.initialize()
    services, deep_client = _build_end_to_end_services(config, database, deep_relevant=False)

    runner = PipelineRunner(config, services)
    run_result = runner.run(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)
    counts = run_result.counts

    # Qwen really was called and really did say IRRELEVANT -- this is not
    # a bypass of deep analysis.
    assert len(deep_client.calls) == 1
    assert counts["deep_analysis_attempted"] == 1
    assert counts["deep_analysis_success"] == 1  # the override makes this a success, not a failure/timeout
    assert counts["deep_analysis_timeouts"] == 0
    assert counts["deep_analysis_failures"] == 0

    # ResearchItem exists, carries an honest record of the disagreement,
    # and was never forced into existence by the test itself.
    assert counts["research_items_total"] == 1
    assert counts["research_items_relevant"] == 1

    conn = database.connect()
    try:
        row = conn.execute(
            "SELECT relevance_label, relevance_override FROM research_items"
        ).fetchone()
        assert row[0] == "relevant"
        assert row[1] is not None
        assert "qwen_relevance=irrelevant" in row[1]
        assert "deterministic_scope_signal=in_scope" in row[1]
        assert "final_relevance=relevant" in row[1]

        event_row = conn.execute("SELECT event_date FROM events").fetchone()
        assert event_row is not None
        assert event_row[0] is not None  # a real, non-fabricated event_date

        assert conn.execute("SELECT COUNT(*) FROM digests WHERE id = ?", (run_result.digest_id,)).fetchone()[0] == 1
    finally:
        conn.close()

    # Digest-qualified and visible to the user-facing UI read path.
    assert counts["digest_event_count"] == 1
    assert run_result.digest_persistence_status.value == "persisted"

    view = dd.get_latest_current_intelligence(services.persistence_service, config)
    assert view.status == "ok"
    assert view.event_count == 1
    assert "JumpCloud" in view.events[0].canonical_title

    # The disagreement is surfaced in the run's own diagnostics too, not
    # only buried in the DB column.
    override_diagnostics = [d for d in run_result.diagnostics if "relevance_override" in d.message]
    assert len(override_diagnostics) == 1
    assert "qwen_relevance=irrelevant" in override_diagnostics[0].message


def test_an_actually_irrelevant_article_is_still_correctly_rejected(tmp_path: Path):
    """Negative control: the safeguard must not become a blanket override.
    A thin, single-category article that Qwen marks irrelevant must still
    be rejected -- no ResearchItem, no Event, no digest entry."""
    from research_assistant.config import get_config
    from research_assistant.retrieval.models import ExtractedArticle

    config = get_config()
    database = Database(tmp_path / "e2e_genuinely_irrelevant.db")
    database.initialize()
    pr_newswire = next(s for s in build_source_definitions(config) if s.name == "PR Newswire")
    all_sources = build_source_definitions(config)
    # Passes both the keyword prefilter (one bare "AI" mention) and the
    # structural gate's substantive-body-length check, and is even
    # genuinely dated ("today released"), but hits only ONE scope category
    # (ai_core) -- below the >=2-category override threshold, so it must
    # still be rejected despite being dated and substantive-length.
    thin_article = ExtractedArticle(
        url=JUMPCLOUD_URL, title="JumpCloud announces minor artificial intelligence UI update",
        text=(
            "JumpCloud Inc. today released a minor artificial intelligence UI update to its admin console. "
            "The update changes the color scheme of several buttons and reorganizes the left-hand "
            "navigation menu for easier access to commonly used screens. Customers will also notice "
            "a refreshed loading animation and a handful of small wording tweaks across the console. "
            "No new capabilities were added in this release; it is a purely cosmetic refresh intended "
            "to modernize the look and feel of the existing admin console ahead of next quarter's "
            "broader visual redesign, which the company plans to detail in a later announcement."
        ),
        published_at=PERIOD_END - timedelta(hours=2),
    )

    class _ThinIrrelevantDeepClient:
        def __init__(self):
            self.calls = []

        def generate_structured(self, *, system, prompt, schema):
            self.calls.append(prompt)
            return AgentAssessment(
                relevance=RelevanceDecision(label=RelevanceLabel.IRRELEVANT, reason="Routine product UI update, no substantive development."),
                category=TopicCategory.OTHER,
                is_dated_development=False,
                financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
                strategic_significance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
            )

    deep_client = _ThinIrrelevantDeepClient()
    services = PipelineServices(
        config=config,
        retrieval_service=_FakeRetrievalServiceForOneSource(all_sources, pr_newswire.name, thin_article),
        normalization_service=NormalizationService(),
        agents={AgentName.STARTUP: StartupAgent(deep_client, config)},
        event_service=EventService(config, clock=FixedClock(PERIOD_END)),
        report_agent=ReportAgent(ReportingFakeLLMClient([])),
        digest_service=DigestService(config, clock=FixedClock(PERIOD_END)),
        clock=FixedClock(PERIOD_END),
        persistence_service=PersistenceService(database),
        triage_llm_client=FakeTriageLLMClient(default_relevant=True),
        deep_analysis_llm_client=deep_client,
    )

    runner = PipelineRunner(config, services)
    run_result = runner.run(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)
    counts = run_result.counts

    assert len(deep_client.calls) == 1  # Qwen really was called
    assert counts["deep_analysis_attempted"] == 1
    assert counts["deep_analysis_success"] == 0  # correctly NOT overridden
    assert counts["research_items_total"] == 0
    assert counts["events_formed"] == 0
    assert run_result.digest_persistence_status.value == "no_qualifying_events"


# ---------------------------------------------------------------------------
# Real production incident: CURRENT run 7aab105954d38497. Bits In Glass's
# real PR Newswire release ("has achieved Select partner status") reached
# deep analysis, Qwen correctly judged it RELEVANT and is_dated_development
# =True, but the deterministic `_resolve_dated_development` guardrail
# narrowed it back to False because "has achieved" was missing from
# `_DATED_ANNOUNCEMENT_PATTERNS` -- losing the event's date and dropping it
# from the digest despite being genuinely relevant and genuinely dated.
# This proves the fix end to end: discovery -> ... -> digest -> UI.
# ---------------------------------------------------------------------------

BITS_IN_GLASS_URL = (
    "https://www.prnewswire.com/news-releases/"
    "bits-in-glass-achieves-select-partner-status-in-the-claude-partner-network-services-track-302903136.html"
)
BITS_IN_GLASS_TITLE = "Bits In Glass achieves Select Partner Status in the Claude Partner Network Services Track"

# Trimmed, verbatim copy of the real page's own `<meta name='date'>` tag
# and body text (confirmed via live inspection of the real URL during the
# forensic investigation).
_BITS_IN_GLASS_ARTICLE_HTML = f"""<html><head>
<meta name='date' content="2026-10-08T20:40:00-04:00"/>
<title>{BITS_IN_GLASS_TITLE}</title>
</head>
<body><article>
<h1>{BITS_IN_GLASS_TITLE}</h1>
<p>DENVER, Oct. 8, 2026 /PRNewswire/ -- Bits In Glass (BIG) has achieved Select partner status in the Claude Partner Network Services Track. The Services Track recognizes partners for building a certified delivery team, delivering Claude use cases for clients, and proving the work through public customer references.</p>
<p>BIG brings over two decades of business process and data automation to applied AI. Today, its certified team helps enterprises move past pilots by putting agentic AI to work in the middle and back office, across risk, compliance, underwriting, reconciliations, and case management.</p>
</article></body></html>"""

# The real run's own period: 2026-10-08 10:04 IST .. 2026-10-09 10:04 IST.
BIG_PERIOD_END = datetime(2026, 10, 9, 4, 34, tzinfo=timezone.utc)   # 10:04 IST
BIG_PERIOD_START = BIG_PERIOD_END - timedelta(hours=24)


def _extract_bits_in_glass_article():
    return extract_article(_BITS_IN_GLASS_ARTICLE_HTML, BITS_IN_GLASS_URL)


class _BitsInGlassDeepAnalysisFakeLLMClient:
    """Qwen's REAL observed judgment on this candidate (confirmed from the
    actual persisted ResearchItem in the real run): RELEVANT,
    is_dated_development=True, reason citing the announcement language it
    itself recognized. This is not a scripted "make it pass" response --
    it is what the real model actually said; the bug was entirely in the
    deterministic guardrail that ran AFTER this assessment."""

    def __init__(self):
        self.calls = []

    def generate_structured(self, *, system, prompt, schema):
        self.calls.append(prompt)
        assert "Bits In Glass" in prompt
        return AgentAssessment(
            relevance=RelevanceDecision(
                label=RelevanceLabel.RELEVANT,
                reason="Describes a specific action (achieved Select partner status) with announcement language.",
            ),
            category=TopicCategory.ENTERPRISE_AI,
            summary="Bits In Glass achieved Select partner status in the Claude Partner Network.",
            is_dated_development=True,
            entities=[],
            people=[],
            products=[],
            technologies=["Agentic AI"],
            geography=["Denver"],
            financial_services_relevance=RelevanceAssessment(
                level=RelevanceLevel.HIGH, rationale="Includes a reference to a financial services use case.",
            ),
            strategic_significance=RelevanceAssessment(
                level=RelevanceLevel.HIGH, rationale="Describes AI transforming risk, compliance, and underwriting work.",
            ),
        )


def test_bits_in_glass_dated_announcement_survives_the_complete_current_pipeline(tmp_path: Path):
    """Discovery -> date gate -> evidence -> routing -> triage -> deep
    analysis -> ResearchItem (is_dated_development=True, preserved, not
    narrowed) -> Event (event_date derived from the publication timestamp)
    -> CURRENT digest qualification -> persistence -> UI read path."""
    from research_assistant.config import get_config

    config = get_config()
    database = Database(tmp_path / "e2e_bits_in_glass.db")
    database.initialize()
    pr_newswire = next(s for s in build_source_definitions(config) if s.name == "PR Newswire")
    all_sources = build_source_definitions(config)
    extracted = _extract_bits_in_glass_article()
    assert extracted.published_at is not None  # the date-extraction fix, proven inline

    deep_client = _BitsInGlassDeepAnalysisFakeLLMClient()
    services = PipelineServices(
        config=config,
        retrieval_service=_FakeRetrievalServiceForOneSource(all_sources, pr_newswire.name, extracted),
        normalization_service=NormalizationService(),
        # Real routing finding: this text content routes to StartupAgent
        # (AgentName.STARTUP), not AI_PRODUCT -- confirmed by running the
        # real router against it; the real production run's persisted
        # `agent=ai_product` reflects a slightly different real article
        # body than this trimmed fixture, not a test assumption.
        agents={AgentName.STARTUP: StartupAgent(deep_client, config)},
        event_service=EventService(config, clock=FixedClock(BIG_PERIOD_END)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
        digest_service=DigestService(config, clock=FixedClock(BIG_PERIOD_END)),
        clock=FixedClock(BIG_PERIOD_END),
        persistence_service=PersistenceService(database),
        triage_llm_client=FakeTriageLLMClient(default_relevant=True),
        deep_analysis_llm_client=deep_client,
    )

    runner = PipelineRunner(config, services)
    run_result = runner.run(digest_type=DigestType.CURRENT, period_start=BIG_PERIOD_START, period_end=BIG_PERIOD_END)
    counts = run_result.counts

    assert len(deep_client.calls) == 1
    assert counts["deep_analysis_attempted"] == 1
    assert counts["deep_analysis_success"] == 1
    assert counts["research_items_total"] == 1

    conn = database.connect()
    try:
        row = conn.execute(
            "SELECT is_dated_development, relevance_label FROM research_items"
        ).fetchone()
        assert bool(row[0]) is True  # preserved, not narrowed back to False
        assert row[1] == "relevant"

        event_row = conn.execute("SELECT event_date FROM events").fetchone()
        assert event_row is not None
        assert event_row[0] is not None
        event_date = datetime.fromisoformat(event_row[0].replace("Z", "+00:00")) if isinstance(event_row[0], str) else event_row[0]
    finally:
        conn.close()

    assert counts["events_formed"] == 1
    assert counts["digest_event_count"] == 1
    assert counts["events_excluded_from_digest_period"] == 0
    assert run_result.digest_persistence_status.value == "persisted"

    view = dd.get_latest_current_intelligence(services.persistence_service, config)
    assert view.status == "ok"
    assert view.event_count == 1
    assert "Bits In Glass" in view.events[0].canonical_title
    assert view.events[0].event_date is not None
    assert BIG_PERIOD_START <= view.events[0].event_date < BIG_PERIOD_END


# ---------------------------------------------------------------------------
# CaixaBank forensic regression (2026-10-12): the exact real article named
# in the "relevant, dated articles are not reaching Events/the digest" bug
# report. Direct persisted-database inspection during the investigation
# showed this specific article ALREADY completes the full real pipeline
# (routing -> triage -> deep analysis -> ResearchItem -> Event -> digest)
# -- the one confirmed, reproducible defect was that its persisted Event's
# `primary_company` was "PR Newswire" (the press-distribution WIRE SERVICE
# that distributed the release), never "CaixaBank" (the actual subject of
# the release) -- see `events.builder._select_primary_company`. This test
# proves, through the REAL pipeline (not a unit test of the builder in
# isolation), that a press-distribution article naming its real subject
# company in a COMPANY-typed entity now gets the correct `primary_company`,
# while every other stage (routing, triage, deep analysis, ResearchItem,
# Event, digest qualification) is completely unaffected.
# ---------------------------------------------------------------------------

CAIXABANK_URL = (
    "https://www.prnewswire.com/news-releases/"
    "caixabank-expands-strategic-collaboration-with-google-cloud-to-accelerate-ai-and-data-capabilities-302902164.html"
)
CAIXABANK_TITLE = "CaixaBank Expands Strategic Collaboration with Google Cloud to Accelerate AI and Data Capabilities"
CAIXABANK_TOPIC_URL = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"

_CAIXABANK_TOPIC_PAGE_HTML = f"""
<html><body><main>
  <a href="{CAIXABANK_URL}">{CAIXABANK_TITLE}</a>
</main></body></html>
"""

# Trimmed, verbatim copy of the real page's own dateline/lede (confirmed
# against the real persisted `articles.raw_text` row during the
# investigation).
_CAIXABANK_ARTICLE_HTML = f"""<html><head>
<title>{CAIXABANK_TITLE}</title>
<script type="application/ld+json">
        {{
            "@context": "https://schema.org",
            "@type": "NewsArticle",
            "headline": "{CAIXABANK_TITLE}",
            "datePublished": "2026-10-09T00:00:00Z",
            "publisher": {{"@type": "Organization", "name": "Cision PR Newswire"}}
        }}
    </script>
</head>
<body><article>
<h1>{CAIXABANK_TITLE}</h1>
<p>SUNNYVALE, Calif. and MADRID, Oct. 9, 2026 /PRNewswire/ -- CaixaBank today announced an expanded
strategic agreement with Google Cloud, strengthening its technological capabilities in key areas such as
artificial intelligence (AI), data management, data analytics, and cloud infrastructure. As part of the
new agreement, CaixaBank will deploy Gemini Enterprise, Google Cloud's AI platform for building, governing,
and deploying agents.</p>
</article></body></html>"""

CAIXABANK_PERIOD_END = datetime(2026, 10, 9, 20, 0, tzinfo=timezone.utc)
CAIXABANK_PERIOD_START = CAIXABANK_PERIOD_END - timedelta(hours=24)


def _discover_caixabank_candidate():
    session = FakeSession([FakeResponse(status_code=200, url=CAIXABANK_TOPIC_URL, text=_CAIXABANK_TOPIC_PAGE_HTML)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    from research_assistant.config import get_config

    config = get_config()
    pr_newswire = next(s for s in build_source_definitions(config) if s.name == "PR Newswire")
    candidates = discover_candidates(pr_newswire, http_client=client, max_candidates=50)
    caixabank = next(c for c in candidates if c.url == CAIXABANK_URL)
    return pr_newswire, caixabank


def _extract_caixabank_article():
    return extract_article(_CAIXABANK_ARTICLE_HTML, CAIXABANK_URL)


class _CaixaBankDeepAnalysisFakeLLMClient:
    """Deterministic deep-analysis stand-in naming CaixaBank and Google
    Cloud as COMPANY-typed entities -- exactly what a real Qwen deep
    analysis of this article's content would extract -- never a live
    model call."""

    def __init__(self):
        self.calls = []

    def generate_structured(self, *, system, prompt, schema):
        self.calls.append(prompt)
        assert "CaixaBank" in prompt, "only the CaixaBank candidate should reach deep analysis in this fixture"
        return AgentAssessment(
            relevance=RelevanceDecision(
                label=RelevanceLabel.RELEVANT,
                reason="CaixaBank expanded a strategic AI/cloud agreement with Google Cloud.",
            ),
            category=TopicCategory.ENTERPRISE_AI,
            summary="CaixaBank expanded its strategic agreement with Google Cloud to deploy Gemini Enterprise.",
            is_dated_development=True,
            entities=[
                TypedEntity(name="CaixaBank", type=EntityType.COMPANY),
                TypedEntity(name="Google Cloud", type=EntityType.COMPANY),
            ],
            people=[],
            products=["Gemini Enterprise"],
            technologies=["generative AI"],
            geography=["Spain"],
            financial_services_relevance=RelevanceAssessment(
                level=RelevanceLevel.HIGH, rationale="A bank expanding its core AI/cloud infrastructure agreement.",
            ),
            strategic_significance=RelevanceAssessment(
                level=RelevanceLevel.HIGH, rationale="Deepens CaixaBank's dependence on Google Cloud's AI platform.",
            ),
        )


def test_caixabank_survives_the_complete_current_pipeline_with_the_correct_primary_company(tmp_path: Path):
    """Real pipeline, real routing (content-signal match on "Google Cloud",
    a configured ai_provider name, since PR Newswire is not itself an
    ai_provider/peer/analyst registry member) -> AIProductAgent -> a real
    Event whose `primary_company` is the subject ("CaixaBank"), never the
    distributing wire service ("PR Newswire")."""
    from research_assistant.config import get_config

    config = get_config()
    database = Database(tmp_path / "caixabank_e2e.db")
    database.initialize()

    pr_newswire, _candidate = _discover_caixabank_candidate()
    extracted = _extract_caixabank_article()
    all_sources = build_source_definitions(config)
    deep_client = _CaixaBankDeepAnalysisFakeLLMClient()
    ai_product_agent = AIProductAgent(deep_client, config)

    services = PipelineServices(
        config=config,
        retrieval_service=_FakeRetrievalServiceForOneSource(all_sources, pr_newswire.name, extracted),
        normalization_service=NormalizationService(),
        agents={AgentName.AI_PRODUCT: ai_product_agent},
        event_service=EventService(config, clock=FixedClock(CAIXABANK_PERIOD_END)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
        digest_service=DigestService(config, clock=FixedClock(CAIXABANK_PERIOD_END)),
        clock=FixedClock(CAIXABANK_PERIOD_END),
        persistence_service=PersistenceService(database),
        triage_llm_client=FakeTriageLLMClient(default_relevant=True),
        deep_analysis_llm_client=deep_client,
    )

    runner = PipelineRunner(config, services)
    run_result = runner.run(
        digest_type=DigestType.CURRENT, period_start=CAIXABANK_PERIOD_START, period_end=CAIXABANK_PERIOD_END,
    )
    counts = run_result.counts

    # Routing: Google Cloud is a configured ai_provider -> content-signal
    # route to AI_PRODUCT (PR Newswire itself carries no fixed primary-agent
    # membership here, so this is case-2 routing, not case-3/case-1).
    assert counts["routed_ai_technology"] == 1

    # Triage/deep analysis actually ran (never bypassed/fabricated).
    assert counts["triage_relevant"] == 1
    assert counts["deep_analysis_attempted"] == 1
    assert counts["deep_analysis_success"] == 1
    assert len(deep_client.calls) == 1

    # ResearchItem -> Event -> digest qualification, same as any other
    # genuinely dated, in-period development.
    assert counts["research_items_total"] == 1
    assert counts["events_formed"] == 1
    assert counts["digest_event_count"] == 1
    assert counts["events_excluded_from_digest_period"] == 0
    assert run_result.digest_persistence_status.value == "persisted"

    conn = database.connect()
    try:
        event_row = conn.execute(
            "SELECT primary_company, related_companies_json, event_date FROM events"
        ).fetchone()
    finally:
        conn.close()

    # The actual fix: the subject company, never the distributing wire service.
    assert event_row[0] == "CaixaBank"
    assert "Google Cloud" in event_row[1]
    assert event_row[2] is not None

    view = dd.get_latest_current_intelligence(services.persistence_service, config)
    assert view.status == "ok"
    assert view.event_count == 1
    assert view.events[0].primary_company == "CaixaBank"
