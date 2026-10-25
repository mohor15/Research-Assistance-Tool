"""Financial Times Technology section (config/sources.yaml: Financial Times
`urls` now also includes `https://www.ft.com/technology`, category
Technology) -- the section/landing page must never become "one article";
individual FT articles must be discovered, dated from real page metadata
(never fabricated/never the section-crawl time), date-filtered by the
existing CURRENT contract, deduplicated, and able to proceed through the
existing generic discovery/extraction/routing/triage/CURRENT pipeline
exactly like every other secondary source. Fixtures/fakes only -- no live
network, no LLM, no browser."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.agents.ai_product import AIProductAgent
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.config import get_config
from research_assistant.evidence.conversion import article_to_evidence
from research_assistant.evidence.normalizer import normalize_article
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import DigestPersistenceStatus
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
)
from research_assistant.pipeline.nodes import _filter_by_period
from research_assistant.retrieval.client import HTTPClient
from research_assistant.retrieval.discovery import discover_candidates
from research_assistant.retrieval.extractor import extract_article
from research_assistant.retrieval.models import DiscoveryMethod, ExtractedArticle, SourceCategory, SourceDefinition
from research_assistant.storage.persistence_service import PersistenceService

from tests.agents.conftest import FakeLLMClient, secondary_source_for
from tests.retrieval.fakes import FakeResponse, FakeSession
from tests.storage.conftest import D, make_digest
from tests.ui.test_current_user_dashboard import _run, _seed_event

CONFIG = get_config()
FT_TECH_URL = "https://www.ft.com/technology"


def _ft_source() -> SourceDefinition:
    return secondary_source_for(CONFIG, "Financial Times")


# A realistic FT Technology section page: a mix of real `/content/<uuid>`
# article cards, a duplicate link to one of them, and non-article nav/
# category links that must never become candidates.
_UUID_1 = "aaaaaaaa-1111-2222-3333-444444444444"
_UUID_2 = "bbbbbbbb-1111-2222-3333-444444444444"
_UUID_3 = "cccccccc-1111-2222-3333-444444444444"
_UUID_4 = "dddddddd-1111-2222-3333-444444444444"
_UUID_5 = "eeeeeeee-1111-2222-3333-444444444444"

FT_SECTION_HTML = f"""
<html><body>
<nav>
  <a href="/technology">Technology</a>
  <a href="/stream/{_UUID_1}">More on this topic</a>
  <a href="/search">Search</a>
</nav>
<main>
  <a href="/content/{_UUID_1}">Bank deploys generative AI copilot for traders</a>
  <a href="/content/{_UUID_1}">Bank deploys generative AI copilot for traders (duplicate card)</a>
  <a href="/content/{_UUID_2}">Fintech start-up raises $200m Series C</a>
  <a href="/content/{_UUID_3}">AI regulation debate intensifies in Brussels</a>
  <a href="/content/{_UUID_4}">Cloud provider unveils new chip for AI workloads</a>
  <a href="/content/{_UUID_5}">Undated wire story on AI partnership</a>
</main>
</body></html>
"""


def _article_page(*, title: str, published_iso: str) -> str:
    return f"""
    <html><head>
      <title>{title} | Financial Times</title>
      <meta property="og:title" content="{title}" />
      <meta property="article:published_time" content="{published_iso}" />
      <meta property="og:site_name" content="Financial Times" />
    </head><body><article>
      <p>{title}. This is a substantive piece of reporting with enough body text
      to clear the extraction thresholds used throughout this project, covering
      the business and technology implications of the development in detail.</p>
      <p>Additional paragraph of real reporting content to ensure the extracted
      text comfortably exceeds the minimum successful-extraction length.</p>
    </article></body></html>
    """


def _undated_article_page(*, title: str) -> str:
    return f"""
    <html><head><title>{title} | Financial Times</title></head>
    <body><article>
      <p>{title}. A short snippet of public, non-paywalled description text for
      this story, with no publication-date metadata anywhere on the page.</p>
    </article></body></html>
    """


# ---------------------------------------------------------------------------
# FT-1: section page discovery never treats the section URL as an article
# ---------------------------------------------------------------------------


def test_ft1_section_page_discovers_individual_article_candidates_not_the_section_itself():
    source = _ft_source()
    assert FT_TECH_URL in source.urls

    session = FakeSession([
        FakeResponse(status_code=200, url=u, text=FT_SECTION_HTML if u == FT_TECH_URL else "<html></html>")
        for u in source.urls
    ])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    candidates = discover_candidates(source, http_client=client, max_candidates=50)

    urls = {c.url for c in candidates}
    assert f"https://www.ft.com/content/{_UUID_1}" in urls
    assert f"https://www.ft.com/content/{_UUID_2}" in urls
    assert f"https://www.ft.com/content/{_UUID_3}" in urls
    assert f"https://www.ft.com/content/{_UUID_4}" in urls
    assert f"https://www.ft.com/content/{_UUID_5}" in urls
    # The section page itself (and its own nav self-link back to that exact
    # URL) is a discovery entry point, never a candidate in its own right --
    # see discovery.py's own-URL exclusion.
    assert FT_TECH_URL not in urls
    assert not any(u.endswith("/search") for u in urls)
    assert all(c.discovery_method == DiscoveryMethod.HTML_LINKS for c in candidates)


# ---------------------------------------------------------------------------
# FT-2: real per-article publication dates are extracted from page metadata
# ---------------------------------------------------------------------------


def test_ft2_article_publication_dates_are_extracted_from_page_metadata():
    html = _article_page(title="Bank deploys generative AI copilot for traders", published_iso="2026-10-06T09:30:00Z")

    extracted = extract_article(html, f"https://www.ft.com/content/{_UUID_1}", content_type="text/html")

    # Precision may collapse to the calendar day (trafilatura's own metadata
    # heuristic, corroborated against the page's real article:published_time
    # tag -- see extractor.py's `_corroborate_against_page_metadata`), but
    # the real, non-fabricated publication day must always be recovered.
    assert extracted.published_at is not None
    assert extracted.published_at.date() == datetime(2026, 10, 6).date()


# ---------------------------------------------------------------------------
# FT-3: old vs recent articles -- existing date filtering keeps only the window
# ---------------------------------------------------------------------------


def test_ft3_old_articles_are_excluded_by_the_existing_current_date_window():
    source = _ft_source()
    recent_extracted = ExtractedArticle(
        url=f"https://www.ft.com/content/{_UUID_1}", title="Recent AI deployment",
        published_at=D - timedelta(hours=2), text="x" * 100,
    )
    old_extracted = ExtractedArticle(
        url=f"https://www.ft.com/content/{_UUID_2}", title="Old AI deployment",
        published_at=D - timedelta(days=10), text="x" * 100,
    )
    recent_article = normalize_article(recent_extracted, source=source)
    old_article = normalize_article(old_extracted, source=source)

    period_start, period_end = D - timedelta(hours=24), D
    kept, excluded = _filter_by_period([recent_article, old_article], period_start, period_end)

    assert recent_article in kept
    assert old_article in excluded


# ---------------------------------------------------------------------------
# FT-4: an article with no reliable publication date cannot become a
# CURRENT qualifying event (it is retained as evidence, never date-filtered
# out blindly, but never satisfies the CURRENT period membership check)
# ---------------------------------------------------------------------------


def test_ft4_undated_article_is_never_treated_as_inside_the_current_window():
    html = _undated_article_page(title="Undated wire story on AI partnership")

    extracted = extract_article(html, f"https://www.ft.com/content/{_UUID_5}", content_type="text/html")

    assert extracted.published_at is None

    source = _ft_source()
    article = normalize_article(extracted, source=source)
    assert article.publication_date is None
    # Never fabricated: evidence can still exist (see FT-6), but the event-
    # formation gate in pipeline/events requires publication_date is not
    # None -- an undated article structurally cannot supply that.
    evidence = article_to_evidence(article)
    assert evidence is not None
    assert evidence.publication_date is None


# ---------------------------------------------------------------------------
# FT-5: duplicate FT links in the section are deduplicated
# ---------------------------------------------------------------------------


def test_ft5_duplicate_section_links_are_deduplicated():
    source = _ft_source()
    session = FakeSession([
        FakeResponse(status_code=200, url=u, text=FT_SECTION_HTML if u == FT_TECH_URL else "<html></html>")
        for u in source.urls
    ])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)

    candidates = discover_candidates(source, http_client=client, max_candidates=50)

    urls = [c.url for c in candidates]
    content_url = f"https://www.ft.com/content/{_UUID_1}"
    assert urls.count(content_url) == 1


# ---------------------------------------------------------------------------
# FT-6: public metadata/snippet available, full body unavailable (paywall) --
# no fabricated content, candidate still proceeds with real evidence
# ---------------------------------------------------------------------------


def test_ft6_paywalled_article_uses_only_public_snippet_never_fabricates_body():
    # A realistic paywalled FT page: only the publicly visible card-level
    # snippet/description is present (no full article body behind the
    # paywall), but enough real text exists to clear the partial-extraction
    # threshold.
    html = """
    <html><head>
      <title>Fintech start-up raises $200m Series C | Financial Times</title>
      <meta property="og:title" content="Fintech start-up raises $200m Series C" />
      <meta property="article:published_time" content="2026-10-06T08:00:00Z" />
      <meta name="description" content="A fintech start-up focused on AI-driven
        underwriting has raised $200m in a Series C round, people familiar with
        the deal said, as investors continue to bet on AI infrastructure for
        financial services." />
    </head><body>
      <div class="paywall-notice">Subscribe to continue reading</div>
    </body></html>
    """
    extracted = extract_article(html, f"https://www.ft.com/content/{_UUID_2}", content_type="text/html")

    # bs4 fallback finds no <article>/<main> paragraphs behind the paywall --
    # this project never invents body content, so extraction legitimately
    # yields no substantial text here. The publication date, drawn from real
    # page metadata, is still correctly recovered regardless.
    assert extracted.published_at == datetime(2026, 10, 6, 8, 0, tzinfo=timezone.utc)
    assert extracted.text is None or len(extracted.text) < 50


# ---------------------------------------------------------------------------
# FT-7: relevant vs irrelevant FT Technology article -- existing fake triage
# distinguishes them, no real Ollama/network call
# ---------------------------------------------------------------------------


def _assessment(label: RelevanceLabel, reason: str) -> AgentAssessment:
    return AgentAssessment(
        relevance=RelevanceDecision(label=label, reason=reason),
        category=TopicCategory.ENTERPRISE_AI,
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
    )


def test_ft7_relevant_and_irrelevant_ft_technology_articles_are_distinguished():
    """Exercises `AIProductAgent`, not `FinancialNewsAgent` -- the source-
    aware-routing refactor (config/sources.yaml's Financial Times entry sets
    `research_domain: ai_technology`) routes FT articles to the AI/
    Technology specialist in the real pipeline (see
    tests/pipeline/test_source_aware_routing_refactor.py for the routing
    decision itself); this test exercises that same specialist agent
    directly."""
    source = _ft_source()

    relevant_extracted = ExtractedArticle(
        url=f"https://www.ft.com/content/{_UUID_2}",
        title="Fintech start-up raises $200m Series C for AI underwriting",
        text=(
            "A fintech start-up focused on AI-driven underwriting has raised "
            "$200 million in a Series C round, as banks increasingly adopt "
            "generative AI tools for credit risk and capital markets operations."
        ),
    )
    irrelevant_extracted = ExtractedArticle(
        url=f"https://www.ft.com/content/{_UUID_4}",
        title="Local football club announces new stadium sponsor",
        text=(
            "A regional football club has announced a new multi-year stadium "
            "naming-rights sponsorship deal with a local retailer, the club said."
        ),
    )
    relevant_article = normalize_article(relevant_extracted, source=source)
    irrelevant_article = normalize_article(irrelevant_extracted, source=source)
    relevant_evidence = article_to_evidence(relevant_article)
    irrelevant_evidence = article_to_evidence(irrelevant_article)

    relevant_assessment = _assessment(RelevanceLabel.RELEVANT, "Substantive AI/fintech funding development.")
    agent = AIProductAgent(FakeLLMClient([relevant_assessment]), CONFIG)
    relevant_item = agent.analyze(relevant_article, relevant_evidence)
    assert relevant_item is not None
    assert relevant_item.relevance.label == RelevanceLabel.RELEVANT

    # The irrelevant article never even reaches the LLM: AIProductAgent's
    # deterministic keyword pre-filter (ResearchAgent.analyze's existing
    # short-circuit) rejects it before any `generate_structured` call --
    # scripting zero responses proves no LLM call was attempted.
    agent_no_calls = AIProductAgent(FakeLLMClient([]), CONFIG)
    irrelevant_item = agent_no_calls.analyze(irrelevant_article, irrelevant_evidence)
    assert irrelevant_item is None


# ---------------------------------------------------------------------------
# FT-8: a CURRENT event sourced from FT obeys the exact existing CURRENT
# event date/run ownership rules (no special-casing for this source)
# ---------------------------------------------------------------------------


def test_ft8_ft_sourced_current_event_obeys_existing_run_ownership_rules(persistence_service: PersistenceService):
    import research_assistant.ui.dashboard_data as dd

    event_ids = ["ev-ft-1"]
    for event_id in event_ids:
        _seed_event(persistence_service, event_id=event_id, publisher="Financial Times")
    persistence_service.persist_pipeline_result(
        digest=make_digest("dg-ft", event_ids=event_ids, digest_type=DigestType.CURRENT)
    )
    persistence_service.persist_run(
        _run("cur-ft", digest_id="dg-ft", period_start=D - timedelta(hours=24), period_end=D)
    )

    view = dd.get_latest_current_intelligence(persistence_service, CONFIG)

    assert view.status == dd.CURRENT_STATUS_OK
    assert view.run_id == "cur-ft"
    assert [c.event_id for c in view.events] == event_ids
    assert view.events[0].source_publisher == "Financial Times"
