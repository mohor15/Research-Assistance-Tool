"""Phase 6 (offline end-to-end audit): one CURRENT run, eight distinct
synthetic candidates, each engineered to terminate at a DIFFERENT real
pipeline stage -- discovery dedup, the hard date gate, the wide history
lookback, extraction failure, triage, deep-analysis rejection, deep-
analysis exception, and full success into the digest.

Uses the REAL production entry points end to end -- `discover_candidates`,
`extract_article`, `normalize_article` (via `NormalizationService`), the
REAL LangGraph `PipelineRunner`, and a REAL temp-file SQLite database.
Only the two genuinely external dependencies are faked: network (HTML
fixtures instead of live fetches) and Ollama (deterministic fake clients).
No company/article name here is special-cased anywhere in production
code -- the two sources ("Morgan Stanley", "JPMorgan Chase") and agent
(PeerAgent) are real, already-configured registry entries this project's
own other end-to-end tests already reuse, chosen only so routing works
without a new registry-coupled mismatch. Two distinct companies are used
for the two successful candidates deliberately (not one) -- clustering's
own entity-overlap signal (`events.similarity.is_disjoint_entities`) can
never veto two same-company items sharing no other entities, so same-
company items legitimately risk merging into one Event; that is existing,
correct clustering behavior (tested elsewhere), not what THIS test is
about, so two unrelated companies keep the two Events it checks genuinely
independent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.report import ReportAgent
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.config import get_config
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.evidence.service import NormalizationService
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
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

_SOURCE_NAME = "Morgan Stanley"  # real, already-configured peer registry entry
_NEWSROOM_URL = "https://www.morganstanley.com/about-us-newsroom"
_SECOND_SOURCE_NAME = "JPMorgan Chase"  # a second, real, already-configured peer -- see module docstring
_SECOND_NEWSROOM_URL = "https://www.jpmorganchase.com/newsroom"

PERIOD_END = datetime(2026, 10, 7, 20, 0, tzinfo=timezone.utc)
PERIOD_START = PERIOD_END - timedelta(hours=24)

# --- 1/2. Two genuinely relevant, dated announcements -----------------------

ALPHA_URL = "https://www.morganstanley.com/press-releases/alpha-corp-ai-copilot-launch"
ALPHA_DUP_URL = ALPHA_URL + "?utm_source=twitter&utm_campaign=launch"  # tracking-param duplicate
ALPHA_TITLE = "Morgan Stanley Today Announced the Launch of an AI Copilot for Advisors"
ALPHA_HTML = f"""<html><head>
<title>{ALPHA_TITLE}</title>
<script type="application/ld+json">
{{"@context": "https://schema.org", "@type": "NewsArticle", "headline": "{ALPHA_TITLE}",
  "datePublished": "2026-10-07T12:00:00-04:00", "publisher": {{"@type": "Organization", "name": "Morgan Stanley"}}}}
</script></head><body><article>
<h1>{ALPHA_TITLE}</h1>
<p>NEW YORK, Oct. 7, 2026 -- Morgan Stanley today announced the launch of a new generative AI copilot
designed to help financial advisors research client portfolios faster and surface relevant insights during
live client conversations. The tool draws on the firm's internal research library and public market data,
and was built in partnership with the firm's own technology division over the past eighteen months.</p>
<p>The rollout begins with a pilot group of 500 advisors across the firm's wealth management division,
with a firm-wide rollout planned for early next year pending feedback from the pilot cohort.</p>
</article></body></html>"""

BETA_URL = "https://www.jpmorganchase.com/newsroom/beta-labs-fraud-detection-launch"
BETA_TITLE = "JPMorgan Chase Has Launched a New Real-Time Fraud Detection Model for Capital Markets"
BETA_HTML = f"""<html><head>
<title>{BETA_TITLE}</title>
<script type="application/ld+json">
{{"@context": "https://schema.org", "@type": "NewsArticle", "headline": "{BETA_TITLE}",
  "datePublished": "2026-10-07T15:30:00-04:00", "publisher": {{"@type": "Organization", "name": "JPMorgan Chase"}}}}
</script></head><body><article>
<h1>{BETA_TITLE}</h1>
<p>NEW YORK, Oct. 7, 2026 -- JPMorgan Chase has launched a new real-time fraud detection model covering
capital markets trading activity, built using machine learning techniques trained on several years of
historical trading pattern data. The model flags anomalous order patterns for compliance review within
seconds rather than the hours a manual review previously required.</p>
<p>The firm says the new system has already reduced false-positive compliance alerts by a meaningful margin
during an internal pilot phase conducted earlier this year across two trading desks.</p>
</article></body></html>"""

# --- 3. Dated, but evergreen/no-new-development (negative control) ---------

EVERGREEN_URL = "https://www.morganstanley.com/press-releases/evergreen-ai-commentary"
EVERGREEN_TITLE = "Perspectives on How AI Is Reshaping Financial Services"
EVERGREEN_HTML = f"""<html><head>
<title>{EVERGREEN_TITLE}</title>
<script type="application/ld+json">
{{"@context": "https://schema.org", "@type": "NewsArticle", "headline": "{EVERGREEN_TITLE}",
  "datePublished": "2026-10-07T09:00:00-04:00", "publisher": {{"@type": "Organization", "name": "Morgan Stanley"}}}}
</script></head><body><article>
<h1>{EVERGREEN_TITLE}</h1>
<p>Artificial intelligence continues to reshape how financial institutions think about research, advice,
and operations. Industry observers note that firms across the sector have been investing steadily in
machine learning capabilities for several years, and that trend shows no sign of slowing as capabilities
mature and adoption broadens across front, middle, and back-office functions alike.</p>
<p>Analysts generally agree that the long-term trajectory favors firms that invest early, though the pace
of visible, customer-facing change has so far been gradual rather than sudden across most of the industry.</p>
</article></body></html>"""

# --- 4. Relevant-looking and dated, but the deep-analysis call itself fails -

GAMMA_URL = "https://www.morganstanley.com/press-releases/gamma-fintech-trading-platform"
GAMMA_TITLE = "Morgan Stanley Today Announced a New Electronic Trading Platform for Institutional Clients"
GAMMA_HTML = f"""<html><head>
<title>{GAMMA_TITLE}</title>
<script type="application/ld+json">
{{"@context": "https://schema.org", "@type": "NewsArticle", "headline": "{GAMMA_TITLE}",
  "datePublished": "2026-10-07T11:00:00-04:00", "publisher": {{"@type": "Organization", "name": "Morgan Stanley"}}}}
</script></head><body><article>
<h1>{GAMMA_TITLE}</h1>
<p>NEW YORK, Oct. 7, 2026 -- Morgan Stanley today announced a new electronic trading platform aimed at
institutional clients trading across multiple asset classes from a single, unified interface. The platform
integrates existing execution algorithms with a new enterprise AI order-routing layer intended to reduce
market impact on large block trades placed during volatile trading sessions.</p>
<p>The firm plans a phased rollout to select institutional clients beginning next quarter, with broader
availability expected to follow depending on client feedback gathered during the initial phase.</p>
</article></body></html>"""

# --- 5. No extractable publication date at all ------------------------------

NODATE_URL = "https://www.morganstanley.com/press-releases/nodate-press-release"
NODATE_TITLE = "An Undated Press Release With No Metadata Date Anywhere"
NODATE_HTML = f"""<html><head><title>{NODATE_TITLE}</title></head><body><article>
<h1>{NODATE_TITLE}</h1>
<p>This article deliberately carries no date metadata of any kind -- no JSON-LD, no meta tag, no visible
labelled publication date -- so it must fail the hard CURRENT date gate and never reach routing, triage,
or deep analysis, regardless of how relevant its content might otherwise read to a human.</p>
<p>Padding this body past the minimum substantive-content threshold so that, if the date gate were ever
bypassed by a defect, this candidate would not also be rejected for an unrelated, confusing reason.</p>
</article></body></html>"""

# --- 6. Genuinely old -- outside even the wide history-retention window ----

STALE_URL = "https://www.morganstanley.com/press-releases/old-stale-news"
STALE_TITLE = "Morgan Stanley Announced This Weeks Before the Current Scan Window"
STALE_HTML = f"""<html><head>
<title>{STALE_TITLE}</title>
<script type="application/ld+json">
{{"@context": "https://schema.org", "@type": "NewsArticle", "headline": "{STALE_TITLE}",
  "datePublished": "2026-09-20T10:00:00-04:00", "publisher": {{"@type": "Organization", "name": "Morgan Stanley"}}}}
</script></head><body><article>
<h1>{STALE_TITLE}</h1>
<p>This article has a perfectly real, trustworthy publication date -- it is simply well over two weeks
before this run's own [period_start, period_end) window and the wider history-retention lookback, so it
must be excluded before ever reaching routing or triage, not merely excluded from the digest at the very
end.</p>
</article></body></html>"""

# --- 7. Extraction failure: no usable article content at all ----------------

GARBAGE_URL = "https://www.morganstanley.com/press-releases/garbage-extraction-failure"
GARBAGE_HTML = "<html><body><nav>Menu</nav></body></html>"  # far below MIN_SUCCESS_CHARS -- not an article


def _real_source(name: str):
    config = get_config()
    return next(s for s in build_source_definitions(config) if s.name == name)


def _discover_against_listing_with_duplicate():
    """Real `discover_candidates()` against TWO sources' own listing pages
    (discovery never follows a cross-domain link -- Beta's JPMorgan Chase
    URL could never be discovered from Morgan Stanley's own hub page, so
    each source's hub is scanned separately, exactly as production does
    one source at a time). The Morgan Stanley listing contains BOTH
    Alpha's canonical URL and a tracking-parameter duplicate variant --
    proves discovery-level handling of a duplicate link before anything
    downstream ever sees two (see
    `test_discovery_keeps_both_url_variants_for_normalize_to_canonicalize`
    for exactly what discovery itself does and does not collapse)."""
    morgan_stanley_html = f"""
    <html><body><main>
      <a href="{ALPHA_URL}">{ALPHA_TITLE}</a>
      <a href="{ALPHA_DUP_URL}">{ALPHA_TITLE} (shared on social)</a>
      <a href="{EVERGREEN_URL}">{EVERGREEN_TITLE}</a>
      <a href="{GAMMA_URL}">{GAMMA_TITLE}</a>
      <a href="{NODATE_URL}">{NODATE_TITLE}</a>
      <a href="{STALE_URL}">{STALE_TITLE}</a>
      <a href="{GARBAGE_URL}">Garbage page</a>
    </main></body></html>
    """
    jpmorgan_html = f"""<html><body><main><a href="{BETA_URL}">{BETA_TITLE}</a></main></body></html>"""

    ms_session = FakeSession([FakeResponse(status_code=200, url=_NEWSROOM_URL, text=morgan_stanley_html)])
    ms_client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=ms_session)
    ms_candidates = discover_candidates(_real_source(_SOURCE_NAME), http_client=ms_client, max_candidates=50)

    jpm_session = FakeSession([FakeResponse(status_code=200, url=_SECOND_NEWSROOM_URL, text=jpmorgan_html)])
    jpm_client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=jpm_session)
    jpm_candidates = discover_candidates(_real_source(_SECOND_SOURCE_NAME), http_client=jpm_client, max_candidates=50)

    return ms_candidates + jpm_candidates


_ARTICLE_HTML_BY_URL = {
    ALPHA_URL: ALPHA_HTML,
    ALPHA_DUP_URL: ALPHA_HTML,
    BETA_URL: BETA_HTML,
    EVERGREEN_URL: EVERGREEN_HTML,
    GAMMA_URL: GAMMA_HTML,
    NODATE_URL: NODATE_HTML,
    STALE_URL: STALE_HTML,
    GARBAGE_URL: GARBAGE_HTML,
}


class _FakeRetrievalServiceForNamedSources:
    def __init__(self, sources, extracted_by_source_name: dict):
        self._sources = sources
        self._extracted_by_source_name = extracted_by_source_name

    def list_sources(self):
        return list(self._sources)

    def retrieve_source(self, source, *, cache_policy=None):
        return list(self._extracted_by_source_name.get(source.name, []))


class _DeepAnalysisFakeLLMClient:
    """Dispatches on which fixture's distinctive title text appears in the
    prompt -- Alpha/Beta succeed as genuine dated developments, Evergreen
    is judged IRRELEVANT (same real shape as the IQuest-Q1 production
    case), Gamma raises to prove a single candidate's analysis failure
    never aborts the run or silently drops the other candidates."""

    def __init__(self):
        self.calls: List[str] = []

    def generate_structured(self, *, system, prompt, schema):
        self.calls.append(prompt)
        if "Alpha Corp" in prompt or "AI Copilot for Advisors" in prompt:
            return self._relevant_assessment(
                "Morgan Stanley today announced the launch of a generative AI copilot for advisors.",
            )
        if "Fraud Detection Model" in prompt:
            return self._relevant_assessment(
                "Morgan Stanley has launched a new real-time fraud detection model for capital markets.",
            )
        if "Perspectives on How AI Is Reshaping" in prompt:
            return AgentAssessment(
                relevance=RelevanceDecision(
                    label=RelevanceLabel.IRRELEVANT,
                    reason="Evergreen industry commentary -- no specific new action or announcement.",
                ),
                category=TopicCategory.ENTERPRISE_AI,
                summary=None,
                is_dated_development=False,
                entities=[], people=[], products=[], technologies=[], geography=[],
                financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
                strategic_significance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
            )
        if "Electronic Trading Platform" in prompt:
            raise RuntimeError("simulated deep-analysis model error for Gamma Fintech")
        raise AssertionError(f"unexpected candidate reached deep analysis: {prompt[:200]}")

    @staticmethod
    def _relevant_assessment(reason: str) -> AgentAssessment:
        return AgentAssessment(
            relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason=reason),
            category=TopicCategory.ENTERPRISE_AI,
            summary=reason,
            is_dated_development=True,
            entities=[], people=[], products=[], technologies=["AI"], geography=["United States"],
            financial_services_relevance=RelevanceAssessment(
                level=RelevanceLevel.HIGH, rationale="Directly describes a financial-services AI deployment.",
            ),
            strategic_significance=RelevanceAssessment(
                level=RelevanceLevel.MEDIUM, rationale="A real, dated operational development.",
            ),
        )


def _build_services(config, database: Database, deep_client):
    all_sources = build_source_definitions(config)
    peer_agent = PeerAgent(deep_client, config)

    candidates = _discover_against_listing_with_duplicate()
    extracted_by_source_name: dict = {_SOURCE_NAME: [], _SECOND_SOURCE_NAME: []}
    for candidate in candidates:
        extracted_by_source_name[candidate.source_name].append(
            extract_article(_ARTICLE_HTML_BY_URL[candidate.url], candidate.url)
        )

    return PipelineServices(
        config=config,
        retrieval_service=_FakeRetrievalServiceForNamedSources(all_sources, extracted_by_source_name),
        normalization_service=NormalizationService(),
        agents={AgentName.PEER: peer_agent},
        event_service=EventService(config, clock=FixedClock(PERIOD_END)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative() for _ in range(2)])),
        digest_service=DigestService(config, clock=FixedClock(PERIOD_END)),
        clock=FixedClock(PERIOD_END),
        persistence_service=PersistenceService(database),
        triage_llm_client=FakeTriageLLMClient(default_relevant=True),
        deep_analysis_llm_client=deep_client,
    ), candidates


def test_discovery_keeps_both_url_variants_for_normalize_to_canonicalize():
    """Stage A, verified rather than assumed: `discover_candidates`'s own
    dedup (`discovery._canonical`) only strips a fragment/trailing slash --
    it deliberately does NOT strip query parameters, so a tracking-param
    duplicate still produces two distinct raw candidates here. The REAL
    canonicalization dedup (stripping `utm_*` tracking parameters) lives
    one stage later, in `evidence.urls.canonicalize_url` via
    `NormalizationService.deduplicate` -- proven in the full run below,
    where the two variants collapse into a single persisted Article."""
    candidates = _discover_against_listing_with_duplicate()
    urls = [c.url for c in candidates]
    alpha_hits = [u for u in urls if "alpha-corp-ai-copilot-launch" in u]

    assert len(alpha_hits) == 2  # both variants are still distinct candidates at THIS stage
    assert len(urls) == 8  # all 8 listed links survive raw discovery


def test_candidate_accounting_reconciles_across_every_terminal_stage(tmp_path: Path):
    config = get_config()
    database = Database(tmp_path / "e2e_accounting.db")
    database.initialize()
    deep_client = _DeepAnalysisFakeLLMClient()
    services, candidates = _build_services(config, database, deep_client)

    runner = PipelineRunner(config, services)
    run_result = runner.run(digest_type=DigestType.CURRENT, period_start=PERIOD_START, period_end=PERIOD_END)
    counts = run_result.counts

    # --- Discovery: 8 raw candidates -- discovery itself only dedups a
    # fragment/trailing-slash variant, never a tracking-parameter one (see
    # `test_discovery_keeps_both_url_variants_for_normalize_to_canonicalize`) --
    assert counts["candidates_discovered"] == 8
    assert len(candidates) == 8

    # --- Normalize/dedup/history lookback -------------------------------------
    # All 8 normalize successfully (even the "garbage" one -- extraction
    # failure is a content problem, not a normalization crash); the article-
    # level canonical-URL dedup then collapses Alpha's tracking-param
    # duplicate, leaving 7; the stale 2026-09-20 article is the only one
    # excluded by the wide history-lookback window.
    assert counts["articles_normalized"] == 8
    assert counts["articles_after_dedup"] == 7  # Alpha's tracking-param duplicate collapsed here
    assert counts["articles_excluded_out_of_lookback"] == 1  # Stale

    # --- Evidence creation: the garbage page has no usable content ------------
    assert counts["evidence_created"] == 5  # Alpha, Beta, Evergreen, Gamma, NoDate (NOT Stale/Garbage)
    assert counts["evidence_rejected_missing_both"] == 1  # Garbage

    # --- Hard CURRENT date gate -------------------------------------------------
    # The gate checks the ARTICLE's own publication_date, independent of
    # whether Evidence exists for it -- a retained-but-unretrievable article
    # (Garbage, no usable content) is "undated" by this gate for the same
    # underlying reason it was already evidence-rejected above (extraction
    # never recovered a date OR a body); NoDate is undated for a genuinely
    # different, confirmed reason (real content, zero date metadata of any
    # kind). Two independent undated candidates, not a double-count of one.
    assert counts["current_date_qualified"] == 4  # Alpha, Beta, Evergreen, Gamma
    assert counts["current_dropped_undated"] == 2  # NoDate + Garbage
    assert counts["current_dropped_stale"] == 0  # Stale was already excluded upstream, not double-counted here
    assert counts["current_dropped_future"] == 0

    # --- Routing: all 4 route to PEER (Morgan Stanley is a configured peer) ---
    assert counts["routed_peer"] == 4

    # --- Triage: all 4 structurally eligible candidates sent, all marked relevant
    # by the triage fake (triage is coarse -- it cannot yet see that Evergreen
    # has no new development or that Gamma's deep analysis will fail) ----------
    assert counts["candidates_before_llm"] == 4
    assert counts["candidates_after_deterministic_gate"] == 4
    assert counts["triage_candidates"] == 4
    assert counts["triage_relevant"] == 4
    assert counts["triage_irrelevant"] == 0

    # --- Deep analysis: 4 attempted, 2 succeed, 1 correctly rejected as
    # evergreen, 1 fails with a reported, isolated exception -------------------
    assert counts["deep_analysis_attempted"] == 4
    assert counts["deep_analysis_success"] == 2
    assert counts["deep_analysis_failures"] == 1  # Gamma's simulated model error
    assert counts["agent_failures"] == 1

    # --- Terminal ResearchItem/Event/digest population -------------------------
    assert counts["research_items_total"] == 2  # Alpha + Beta only
    assert counts["events_formed"] == 2
    assert counts["digest_event_count"] == 2
    assert counts["events_excluded_from_digest_period"] == 0

    # No candidate is both a success and a failure, and nothing silently
    # vanished: every one of the 7 discovered candidates is accounted for by
    # exactly one of the counts above (1 stale, 1 garbage/no-evidence,
    # 1 undated, 1 evergreen-rejected, 1 analysis-failed, 2 successful).

    conn = database.connect()
    try:
        titles = {row["title"] for row in conn.execute("SELECT title FROM research_items").fetchall()}
    finally:
        conn.close()
    assert titles == {ALPHA_TITLE, BETA_TITLE}

    # --- Dashboard reads exactly this run's own digest -------------------------
    view = dd.get_latest_current_intelligence(services.persistence_service, config)
    assert view.status == "ok"
    assert view.event_count == 2
    canonical_titles = {e.canonical_title for e in view.events}
    assert canonical_titles == {ALPHA_TITLE, BETA_TITLE}
    assert run_result.digest_persistence_status.value == "persisted"

    # Deep-analysis call isolation: Gamma's failure is visible in diagnostics,
    # by name, not swallowed -- and the run still completed with a persisted
    # digest despite it.
    assert any("Gamma" in d.message or "gamma-fintech" in d.message for d in run_result.diagnostics)
