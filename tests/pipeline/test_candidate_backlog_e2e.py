"""End-to-end proof (real `PipelineRunner`, real temp-file SQLite, only
network+Ollama faked) that a candidate deferred by one CURRENT run's
exhausted deep-analysis budget is durably retried and completed by a LATER
run, without being re-triaged and without duplicating its ResearchItem/
Event. Two real, already-configured PR Newswire startup-funding
announcements compete for a deliberately tiny `max_deep_analysis=1`
budget."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from research_assistant.agents.report import ReportAgent
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.agents.startup import StartupAgent
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

from tests.pipeline.conftest import FakeTriageLLMClient
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative
from tests.retrieval.fakes import FakeResponse, FakeSession

ALPHA_URL = "https://www.prnewswire.com/news-releases/alpha-robotics-raises-20m-series-a-302900001.html"
ALPHA_TITLE = "Alpha Robotics Raises $20M Series A to Scale Warehouse Automation"
BETA_URL = "https://www.prnewswire.com/news-releases/beta-analytics-raises-15m-series-a-302900002.html"
BETA_TITLE = "Beta Analytics Raises $15M Series A for Fraud Detection"
TOPIC_URL = "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/"

_TOPIC_PAGE_HTML = f"""
<html><body><main>
  <a href="{ALPHA_URL}">{ALPHA_TITLE}</a>
  <a href="{BETA_URL}">{BETA_TITLE}</a>
</main></body></html>
"""


def _article_html(title: str, body: str, date_published: str) -> str:
    return f"""<html><head>
<title>{title}</title>
<script type="application/ld+json">
{{"@context": "https://schema.org", "@type": "NewsArticle", "headline": "{title}",
  "datePublished": "{date_published}", "publisher": {{"@type": "Organization", "name": "Cision PR Newswire"}}}}
</script></head>
<body><article><h1>{title}</h1><p>{body}</p></article></body></html>"""


PERIOD1_END = datetime(2026, 10, 9, 20, 0, tzinfo=timezone.utc)
PERIOD1_START = PERIOD1_END - timedelta(hours=24)
# A second, later CURRENT run -- a different run_id (deterministic from
# digest_type+period), proving resume works ACROSS genuinely distinct runs,
# not just a retried identical one. Both real articles' publication dates
# stay comfortably inside both periods' heavy overlap.
PERIOD2_END = PERIOD1_END + timedelta(hours=1)
PERIOD2_START = PERIOD2_END - timedelta(hours=24)
_PUBLISHED_AT = "2026-10-09T10:00:00-04:00"  # 14:00 UTC -- well inside both periods' overlap

_ALPHA_BODY = (
    "BOSTON, Oct. 9, 2026 /PRNewswire/ -- Alpha Robotics today announced it has raised $20 million in "
    "Series A funding to scale its warehouse automation platform. " + ("Additional funding detail. " * 15)
)
_BETA_BODY = (
    "AUSTIN, Oct. 9, 2026 /PRNewswire/ -- Beta Analytics today announced it has raised $15 million in "
    "Series A funding for its fraud-detection platform. " + ("Additional funding detail. " * 15)
)

ALPHA_HTML = _article_html(ALPHA_TITLE, _ALPHA_BODY, _PUBLISHED_AT)
BETA_HTML = _article_html(BETA_TITLE, _BETA_BODY, _PUBLISHED_AT)


def _discover_two_candidates():
    session = FakeSession([FakeResponse(status_code=200, url=TOPIC_URL, text=_TOPIC_PAGE_HTML)])
    client = HTTPClient(timeout_seconds=5, max_retries=0, retry_delay_seconds=0, session=session)
    config = get_config()
    pr_newswire = next(s for s in build_source_definitions(config) if s.name == "PR Newswire")
    candidates = discover_candidates(pr_newswire, http_client=client, max_candidates=50)
    return pr_newswire, candidates


class _TwoArticleRetrievalService:
    """Round 1: PR Newswire yields both real extracted articles. Round 2
    (`extracted=[]`): nothing newly discovered -- the ONLY way a
    ResearchItem can result this run is by resuming the durable backlog,
    isolating the resume path from any discovery-layer dedup behavior."""

    def __init__(self, sources, pr_newswire_name: str, extracted: List):
        self._sources = sources
        self._pr_newswire_name = pr_newswire_name
        self._extracted = extracted

    def list_sources(self):
        return list(self._sources)

    def retrieve_source(self, source, *, cache_policy=None) -> List:
        if source.name == self._pr_newswire_name:
            return list(self._extracted)
        return []


class _StartupDeepAnalysisClient:
    """Deterministic deep-analysis stand-in for both candidates -- never a
    live model call. Tracks exactly which candidate(s) it was asked to
    analyze, so the test can assert the resumed run re-uses the persisted
    triage result and makes exactly one deep-analysis call for exactly the
    resumed candidate."""

    def __init__(self):
        self.calls: List[str] = []

    def generate_structured(self, *, system, prompt, schema):
        self.calls.append(prompt)
        if "Alpha Robotics" in prompt:
            company, product = "Alpha Robotics", "warehouse automation platform"
        elif "Beta Analytics" in prompt:
            company, product = "Beta Analytics", "fraud-detection platform"
        else:
            raise AssertionError(f"unexpected candidate reached deep analysis: {prompt[:200]!r}")
        return AgentAssessment(
            relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason=f"{company} raised Series A funding."),
            category=TopicCategory.ENTERPRISE_AI,
            summary=f"{company} raised Series A funding for its {product}.",
            is_dated_development=True,
            entities=[], people=[], products=[product], technologies=["AI"], geography=["United States"],
            financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
            strategic_significance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="x"),
        )


def _build_services(config, database: Database, *, extracted, max_deep_analysis: int, triage_client, deep_client):
    pr_newswire, _candidates = _discover_two_candidates()
    all_sources = build_source_definitions(config)
    startup_agent = StartupAgent(deep_client, config)

    run_config = config.model_copy(deep=True)
    run_config.settings.current_llm.max_deep_analysis = max_deep_analysis

    return PipelineServices(
        config=run_config,
        retrieval_service=_TwoArticleRetrievalService(all_sources, pr_newswire.name, extracted),
        normalization_service=NormalizationService(),
        agents={AgentName.STARTUP: startup_agent},
        event_service=EventService(run_config, clock=FixedClock(PERIOD1_END)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative(), make_narrative()])),
        digest_service=DigestService(run_config, clock=FixedClock(PERIOD1_END)),
        clock=FixedClock(PERIOD1_END),
        persistence_service=PersistenceService(database),
        triage_llm_client=triage_client,
        deep_analysis_llm_client=deep_client,
    )


def _backlog_rows(database: Database) -> List[sqlite3.Row]:
    conn = database.connect()
    try:
        return conn.execute("SELECT * FROM candidate_backlog").fetchall()
    finally:
        conn.close()


def test_candidate_deferred_by_budget_is_resumed_and_completed_by_a_later_run(tmp_path: Path):
    config = get_config()
    database = Database(tmp_path / "backlog_e2e.db")
    database.initialize()

    # ---- Run 1: two real, dated, triage-relevant candidates compete for a
    # deliberately tiny deep-analysis budget of 1 -- exactly one is
    # analyzed, the other must be durably deferred, never lost. ----
    pr_newswire, candidates = _discover_two_candidates()
    extracted_alpha = extract_article(ALPHA_HTML, ALPHA_URL)
    extracted_beta = extract_article(BETA_HTML, BETA_URL)
    triage_client_1 = FakeTriageLLMClient(default_relevant=True)
    deep_client_1 = _StartupDeepAnalysisClient()
    services_1 = _build_services(
        config, database, extracted=[extracted_alpha, extracted_beta], max_deep_analysis=1,
        triage_client=triage_client_1, deep_client=deep_client_1,
    )
    runner_1 = PipelineRunner(config, services_1)
    result_1 = runner_1.run(digest_type=DigestType.CURRENT, period_start=PERIOD1_START, period_end=PERIOD1_END)

    assert result_1.counts["research_items_total"] == 1  # only one budget slot
    assert result_1.counts["events_formed"] == 1
    assert result_1.counts["candidates_pending_carried_forward"] == 1  # the other, durably deferred
    assert len(deep_client_1.calls) == 1

    rows_after_run_1 = _backlog_rows(database)
    assert len(rows_after_run_1) == 1
    deferred_row = rows_after_run_1[0]
    assert deferred_row["status"] == "pending_deep_analysis_budget"
    assert deferred_row["triage_relevant"] == 1  # triage already succeeded -- never re-triaged
    assert deferred_row["attempt_count"] == 0

    conn = database.connect()
    events_after_run_1 = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
    conn.close()
    assert events_after_run_1 == 1

    # ---- Run 2: nothing newly discovered (retrieval returns []) -- the
    # ONLY possible source of a new ResearchItem is the durable backlog.
    # Triage must NOT run again for the resumed candidate (its persisted
    # triage_relevant=True is reused); deep analysis must run exactly once,
    # for exactly the deferred candidate. ----
    triage_client_2 = FakeTriageLLMClient(default_relevant=True)
    deep_client_2 = _StartupDeepAnalysisClient()
    services_2 = _build_services(
        config, database, extracted=[], max_deep_analysis=1,
        triage_client=triage_client_2, deep_client=deep_client_2,
    )
    runner_2 = PipelineRunner(config, services_2)
    result_2 = runner_2.run(digest_type=DigestType.CURRENT, period_start=PERIOD2_START, period_end=PERIOD2_END)

    assert triage_client_2.calls == []  # never re-triaged
    assert len(deep_client_2.calls) == 1  # exactly the resumed candidate
    assert result_2.counts["research_items_total"] == 1
    assert result_2.counts["events_formed"] == 1
    assert result_2.counts["candidates_completed_from_backlog"] == 1

    rows_after_run_2 = _backlog_rows(database)
    assert len(rows_after_run_2) == 1  # still one row -- never duplicated
    completed_row = rows_after_run_2[0]
    assert completed_row["status"] == "completed"
    assert completed_row["research_item_id"] is not None
    assert completed_row["article_id"] == deferred_row["article_id"]  # the same candidate, resolved

    conn = database.connect()
    total_events = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
    total_research_items = conn.execute("SELECT COUNT(*) AS n FROM research_items").fetchone()["n"]
    total_digests = conn.execute("SELECT COUNT(*) AS n FROM digests").fetchone()["n"]
    conn.close()
    assert total_events == 2  # one per run, never a duplicate for the resumed candidate
    assert total_research_items == 2
    assert total_digests == 2  # each run persisted its own digest
    assert result_2.digest_persistence_status.value == "persisted"
