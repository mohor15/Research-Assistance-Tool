"""Read-only CURRENT Scan Explorer (dashboard_data layer) -- fixtures/mocks
only, no live network, no LLM, no pipeline execution. Confirms the
explorer surfaces retrieved articles and formed events even when the
digest itself is empty, and never confuses "not analyzed" with
"irrelevant"."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.config import get_config
from research_assistant.evidence import article_to_evidence
from research_assistant.models.digest import Digest, DigestTheme, DigestType
from research_assistant.models.enums import AgentName, EventType
from research_assistant.models.research_item import RelevanceDecision, RelevanceLabel
from research_assistant.pipeline.runner import PipelineRunResult, RunStatus
from research_assistant.pipeline.state import PipelineDiagnostic
from research_assistant.storage.persistence_service import PersistenceService
from research_assistant.ui import dashboard_data as dd
from research_assistant.ui.app import _render_current_pipeline_funnel
from research_assistant.ui.dashboard_data import CurrentScanExplorerData

from tests.storage.conftest import D, make_article, make_event, make_report, make_research_item

CONFIG = get_config()


def _current_run(
    run_id="cur1", *, started_at=D - timedelta(minutes=30), finished_at=D + timedelta(minutes=30),
    period_start=None, period_end=None, counts=None, digest_id=None, diagnostics=None,
):
    return PipelineRunResult(
        run_id=run_id, digest_type=DigestType.CURRENT, started_at=started_at, finished_at=finished_at,
        status=RunStatus.PARTIAL_SUCCESS, diagnostics=diagnostics or [], counts=counts or {}, period_start=period_start,
        period_end=period_end, digest_id=digest_id,
    )


def _evening_run(run_id="eve1"):
    return PipelineRunResult(
        run_id=run_id, digest_type=DigestType.EVENING, started_at=D, finished_at=D, status=RunStatus.SUCCESS,
        diagnostics=[], counts={},
    )


# ---------------------------------------------------------------------------
# 1-2: run selector
# ---------------------------------------------------------------------------


def test_latest_current_run_is_first_in_the_selector(persistence_service: PersistenceService):
    older = _current_run("cur-old", started_at=D - timedelta(days=1), finished_at=D - timedelta(days=1))
    newer = _current_run("cur-new", started_at=D, finished_at=D)
    persistence_service.persist_run(older)
    persistence_service.persist_run(newer)

    options = dd.get_current_run_options(persistence_service)

    assert options[0].run_id == "cur-new"


def test_morning_and_evening_runs_never_appear_in_the_current_selector(persistence_service: PersistenceService):
    persistence_service.persist_run(_current_run("cur1"))
    persistence_service.persist_run(_evening_run("eve1"))

    options = dd.get_current_run_options(persistence_service)

    assert {o.run_id for o in options} == {"cur1"}


def test_previous_current_runs_can_be_selected_individually(persistence_service: PersistenceService):
    persistence_service.persist_run(_current_run("cur-a", started_at=D - timedelta(days=2), finished_at=D - timedelta(days=2)))
    persistence_service.persist_run(_current_run("cur-b", started_at=D, finished_at=D))

    options = dd.get_current_run_options(persistence_service)
    run_ids = {o.run_id for o in options}

    assert run_ids == {"cur-a", "cur-b"}
    data_a = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur-a")
    data_b = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur-b")
    assert data_a.run_id == "cur-a"
    assert data_b.run_id == "cur-b"


# ---------------------------------------------------------------------------
# 3: summary metrics come from persisted counts, never recomputed/hardcoded
# ---------------------------------------------------------------------------


def test_summary_metrics_come_verbatim_from_persisted_run_counts(persistence_service: PersistenceService):
    counts = {
        "candidates_discovered": 546, "evidence_created": 247, "agent_candidates_analyzed": 25,
        "research_items_relevant": 11, "events_formed": 13,
    }
    persistence_service.persist_run(_current_run("cur1", counts=counts))

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    assert data.metrics["candidates_discovered"] == 546
    assert data.metrics["evidence_created"] == 247
    assert data.metrics["agent_candidates_analyzed"] == 25
    assert data.metrics["research_items_relevant"] == 11
    assert data.metrics["events_formed"] == 13


def test_digest_persistence_status_is_exposed_verbatim_from_the_run(persistence_service: PersistenceService):
    """Pipeline Funnel fix (dashboard observability task): the run's own
    authoritative `digest_persistence_status` (e.g. "no_qualifying_events")
    must be surfaced, never recomputed from the events list."""
    from research_assistant.models.enums import DigestPersistenceStatus

    run = _current_run("cur-dps")
    run = run.model_copy(update={"digest_persistence_status": DigestPersistenceStatus.NO_QUALIFYING_EVENTS})
    persistence_service.persist_run(run)

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur-dps")

    assert data.digest_persistence_status == "no_qualifying_events"


def test_digest_persistence_status_defaults_to_an_honest_unavailable_value():
    """Stale-process regression (AttributeError fix): a caller that
    constructs `CurrentScanExplorerData` without this field -- e.g. an
    older test fixture, or any future construction path -- must get the
    explicit literal "unavailable", never a crash and never a fabricated
    status inferred from zero Events."""
    data = CurrentScanExplorerData(
        run_id="r", status="partial_success", started_at=D, finished_at=D,
        period_start=None, period_end=None, metrics={}, articles=[], events=[],
        coverage_evidence=0, coverage_analyzed=0, coverage_rejected_by_agent=0,
        coverage_not_analyzed=0, coverage_content_unavailable=0,
        coverage_relevant=0, coverage_irrelevant=0, coverage_ambiguous=0,
        reconstruction_note="note",
    )
    assert data.digest_persistence_status == "unavailable"


def test_pipeline_funnel_renders_with_an_unavailable_digest_persistence_status():
    """The Pipeline Funnel must never crash when fed the defensive
    "unavailable" literal -- it is just displayed as a plain metric
    value, same as any other string status."""
    _render_current_pipeline_funnel({}, "unavailable")


# ---------------------------------------------------------------------------
# 4-5, 7: Retrieved Articles, analysis status, "Not analyzed" != "Irrelevant"
# ---------------------------------------------------------------------------


def test_retrieved_articles_uses_the_selected_run_and_its_retrieved_at_window(persistence_service: PersistenceService):
    article = make_article("art1", company="JPMorgan Chase")  # retrieved_at = D
    evidence = article_to_evidence(article)
    persistence_service.persist_pipeline_result(articles=[article], evidence=[evidence])
    persistence_service.persist_run(_current_run("cur1", started_at=D - timedelta(minutes=5), finished_at=D + timedelta(minutes=5)))

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    assert len(data.articles) == 1
    assert data.articles[0].article_id == "art1"


def test_an_article_outside_the_runs_retrieved_at_window_is_excluded(persistence_service: PersistenceService):
    article = make_article("art1", company="JPMorgan Chase")  # retrieved_at = D
    evidence = article_to_evidence(article)
    persistence_service.persist_pipeline_result(articles=[article], evidence=[evidence])
    # a run whose window does not include D
    persistence_service.persist_run(
        _current_run("cur1", started_at=D + timedelta(days=5), finished_at=D + timedelta(days=5, minutes=30))
    )

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    assert data.articles == []


def test_analyzed_article_shows_its_actual_agent_and_relevance(persistence_service: PersistenceService):
    article = make_article("art1", company="JPMorgan Chase")
    evidence = article_to_evidence(article)
    item = make_research_item("ri1", article_id="art1", agent=AgentName.PEER, evidence_ids=[evidence.evidence_id], run_id="cur1")
    persistence_service.persist_pipeline_result(articles=[article], evidence=[evidence], research_items=[item])
    persistence_service.persist_run(_current_run("cur1"))

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    row = data.articles[0]
    assert row.analysis_status == "Analyzed"
    assert row.agent == "peer"
    assert row.relevance == "Relevant"


def test_unanalyzed_article_shows_not_analyzed_never_irrelevant(persistence_service: PersistenceService):
    article = make_article("art1", company="JPMorgan Chase")
    evidence = article_to_evidence(article)
    persistence_service.persist_pipeline_result(articles=[article], evidence=[evidence])  # no research_item at all
    persistence_service.persist_run(_current_run("cur1"))

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    row = data.articles[0]
    assert row.analysis_status == "Not analyzed"
    assert row.relevance is None  # UI renders this as "Not analyzed", never "Irrelevant" or 0
    assert row.agent is None


# ---------------------------------------------------------------------------
# 5: article URL comes from the persisted canonical/original URL
# ---------------------------------------------------------------------------


def test_article_url_comes_from_the_persisted_url_never_invented(persistence_service: PersistenceService):
    article = make_article("art1", company="JPMorgan Chase", url="https://jpmorganchase.com/real-article")
    evidence = article_to_evidence(article)
    persistence_service.persist_pipeline_result(articles=[article], evidence=[evidence])
    persistence_service.persist_run(_current_run("cur1"))

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    assert data.articles[0].url == "https://jpmorganchase.com/real-article"


# ---------------------------------------------------------------------------
# 8-9: Event mapping and digest-qualified status
# ---------------------------------------------------------------------------


def test_event_mapping_is_correct_and_includes_events_outside_the_empty_digest(persistence_service: PersistenceService):
    article = make_article("art1", company="JPMorgan Chase")
    evidence = article_to_evidence(article)
    item = make_research_item("ri1", article_id="art1", evidence_ids=[evidence.evidence_id], run_id="cur1")
    event = make_event("ev1", research_item_ids=["ri1"], article_ids=["art1"], event_date=D, run_id="cur1")
    report = make_report("ev1")
    persistence_service.persist_pipeline_result(
        articles=[article], evidence=[evidence], research_items=[item], events=[event], reports=[report],
    )
    persistence_service.persist_run(_current_run("cur1", counts={"events_formed": 1}))  # no digest_id -> empty digest

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    assert len(data.events) == 1
    assert data.events[0].event_id == "ev1"
    assert data.events[0].report_title == "An event title"
    assert data.events[0].digest_qualified is False
    # the article row is linked to the event too
    assert data.articles[0].event_id == "ev1"
    assert data.articles[0].has_event is True


def test_digest_qualified_event_is_distinguished_from_formed_but_unqualified(persistence_service: PersistenceService):
    article_a = make_article("art-a", company="JPMorgan Chase", content_hash="hash-a")
    article_b = make_article("art-b", company="Anthropic", content_hash="hash-b")
    evidence_a = article_to_evidence(article_a)
    evidence_b = article_to_evidence(article_b)
    item_a = make_research_item("ri-a", article_id="art-a", evidence_ids=[evidence_a.evidence_id], run_id="cur1")
    item_b = make_research_item("ri-b", article_id="art-b", company="Anthropic", evidence_ids=[evidence_b.evidence_id], run_id="cur1")
    event_qualified = make_event("ev-qualified", research_item_ids=["ri-a"], article_ids=["art-a"], event_date=D, run_id="cur1")
    event_unqualified = make_event("ev-unqualified", research_item_ids=["ri-b"], article_ids=["art-b"], event_date=None, company="Anthropic", run_id="cur1")
    digest = Digest(
        id="dg1", digest_type=DigestType.CURRENT, generated_at=D, period_start=D, period_end=D,
        events=["ev-qualified"], themes=[],
    )
    persistence_service.persist_pipeline_result(
        articles=[article_a, article_b], evidence=[evidence_a, evidence_b], research_items=[item_a, item_b],
        events=[event_qualified, event_unqualified], digest=digest,
    )
    persistence_service.persist_run(_current_run("cur1", digest_id="dg1"))

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")
    by_id = {e.event_id: e for e in data.events}

    assert by_id["ev-qualified"].digest_qualified is True
    assert by_id["ev-qualified"].qualification_reason == "Included"
    assert by_id["ev-unqualified"].digest_qualified is False
    assert by_id["ev-unqualified"].qualification_reason == "Missing date"


# ---------------------------------------------------------------------------
# 10: filtering is pure (never mutates persisted data)
# ---------------------------------------------------------------------------


def test_filtering_in_the_ui_layer_never_mutates_persisted_data(persistence_service: PersistenceService):
    article = make_article("art1", company="JPMorgan Chase")
    evidence = article_to_evidence(article)
    persistence_service.persist_pipeline_result(articles=[article], evidence=[evidence])
    persistence_service.persist_run(_current_run("cur1"))

    data_before = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")
    filtered = [a for a in data_before.articles if a.source_name == "Nonexistent Source"]
    assert filtered == []  # the filter itself is just a Python list comprehension

    data_after = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")
    assert len(data_after.articles) == 1  # unchanged -- nothing was written


# ---------------------------------------------------------------------------
# 12: FT visibility
# ---------------------------------------------------------------------------


def test_ft_articles_are_visible_and_categorized_as_financial_news(persistence_service: PersistenceService):
    article = make_article("art-ft", company="Financial Times", content_hash="hash-ft")
    evidence = article_to_evidence(article)
    persistence_service.persist_pipeline_result(articles=[article], evidence=[evidence])
    persistence_service.persist_run(_current_run("cur1"))

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    ft_rows = [a for a in data.articles if a.source_name == "Financial Times"]
    assert len(ft_rows) == 1
    assert ft_rows[0].category == "Financial News"


# ---------------------------------------------------------------------------
# 14: empty digest still shows Retrieved Articles and Events
# ---------------------------------------------------------------------------


def test_empty_digest_still_shows_retrieved_articles_and_events(persistence_service: PersistenceService):
    article = make_article("art1", company="JPMorgan Chase")
    evidence = article_to_evidence(article)
    item = make_research_item("ri1", article_id="art1", evidence_ids=[evidence.evidence_id], run_id="cur1")
    event = make_event("ev1", research_item_ids=["ri1"], article_ids=["art1"], event_date=None, run_id="cur1")
    persistence_service.persist_pipeline_result(
        articles=[article], evidence=[evidence], research_items=[item], events=[event],
    )
    persistence_service.persist_run(_current_run("cur1", counts={"digest_event_count": 0}))  # no digest_id -> empty

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    assert len(data.articles) == 1  # NOT hidden merely because the digest is empty
    assert len(data.events) == 1
    assert data.events[0].digest_qualified is False


# ---------------------------------------------------------------------------
# 6/11: analysis coverage is internally consistent with the listed articles
# ---------------------------------------------------------------------------


def test_coverage_counts_match_the_actual_listed_articles(persistence_service: PersistenceService):
    analyzed_article = make_article("art-analyzed", company="JPMorgan Chase", content_hash="hash-analyzed")
    unanalyzed_article = make_article("art-unanalyzed", company="Anthropic", content_hash="hash-unanalyzed")
    evidence_a = article_to_evidence(analyzed_article)
    evidence_b = article_to_evidence(unanalyzed_article)
    item = make_research_item("ri1", article_id="art-analyzed", evidence_ids=[evidence_a.evidence_id], run_id="cur1")
    persistence_service.persist_pipeline_result(
        articles=[analyzed_article, unanalyzed_article], evidence=[evidence_a, evidence_b], research_items=[item],
    )
    persistence_service.persist_run(_current_run("cur1"))

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    assert data.coverage_evidence == 2
    assert data.coverage_analyzed == 1
    assert data.coverage_not_analyzed == 1
    assert data.coverage_relevant == 1  # make_research_item defaults to RELEVANT


# ---------------------------------------------------------------------------
# Forensic finding (run 7aab105954d38497, IQuest-Q1): a candidate that an
# agent actually ran triage/deep analysis on and rejected must never be
# reported as "Not analyzed" -- that label must mean "never reached an
# agent at all". No silent omission: every date-qualified candidate gets a
# clear, distinguishable terminal outcome.
# ---------------------------------------------------------------------------


def test_deep_analysis_rejected_candidate_is_not_labelled_not_analyzed(persistence_service: PersistenceService):
    article = make_article("iquest-q1", company="PR Newswire", content_hash="hash-iquest")
    evidence = article_to_evidence(article)
    persistence_service.persist_pipeline_result(articles=[article], evidence=[evidence])  # no ResearchItem -- Qwen rejected it
    diagnostics = [
        PipelineDiagnostic(
            stage="agent_analysis",
            message="triage:relevant candidate=iquest-q1 agent=startup source='PR Newswire' title='IQuest-Q1' url=https://example.com/iquest-q1 confidence=0.95",
            severity="warning",
        ),
        PipelineDiagnostic(
            stage="agent_analysis",
            message="deep_analysis:irrelevant candidate=iquest-q1 agent=startup source='PR Newswire' title='IQuest-Q1' url=https://example.com/iquest-q1",
            severity="warning",
        ),
    ]
    persistence_service.persist_run(_current_run("cur1", diagnostics=diagnostics))

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    row = data.articles[0]
    assert row.analysis_status == "Rejected by deep analysis"
    assert row.analysis_status != "Not analyzed"
    assert row.relevance is None  # still no ResearchItem -- but the status is no longer misleading


def test_triage_rejected_candidate_is_distinguished_from_deep_analysis_rejection(persistence_service: PersistenceService):
    article = make_article("ft-hub", company="Financial Times", content_hash="hash-ft-hub")
    evidence = article_to_evidence(article)
    persistence_service.persist_pipeline_result(articles=[article], evidence=[evidence])
    diagnostics = [
        PipelineDiagnostic(
            stage="agent_analysis",
            message="triage:irrelevant candidate=ft-hub agent=ai_product source='Financial Times' title='hub' url=https://ft.com/hub confidence=0.0",
            severity="warning",
        ),
    ]
    persistence_service.persist_run(_current_run("cur1", diagnostics=diagnostics))

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    row = data.articles[0]
    assert row.analysis_status == "Rejected by triage"


def test_a_candidate_with_no_diagnostic_trace_at_all_is_genuinely_not_analyzed(persistence_service: PersistenceService):
    """A candidate dropped purely by the deep-analysis budget (never sent
    to triage/deep analysis at all) still correctly reports "Not analyzed"
    -- distinguishing a legitimate budget skip from an agent's rejection is
    exactly the point of this fix, not collapsing everything to one label."""
    article = make_article("budget-skipped", company="JPMorgan Chase", content_hash="hash-budget")
    evidence = article_to_evidence(article)
    persistence_service.persist_pipeline_result(articles=[article], evidence=[evidence])
    persistence_service.persist_run(_current_run("cur1", diagnostics=[]))

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    row = data.articles[0]
    assert row.analysis_status == "Not analyzed"


def test_coverage_distinguishes_rejected_by_agent_from_not_analyzed(persistence_service: PersistenceService):
    rejected_article = make_article("rejected", company="PR Newswire", content_hash="hash-rejected")
    skipped_article = make_article("skipped", company="JPMorgan Chase", content_hash="hash-skipped")
    evidence_r = article_to_evidence(rejected_article)
    evidence_s = article_to_evidence(skipped_article)
    persistence_service.persist_pipeline_result(
        articles=[rejected_article, skipped_article], evidence=[evidence_r, evidence_s],
    )
    diagnostics = [
        PipelineDiagnostic(
            stage="agent_analysis",
            message="deep_analysis:irrelevant candidate=rejected agent=startup source='PR Newswire' title='x' url=https://example.com/x",
            severity="warning",
        ),
    ]
    persistence_service.persist_run(_current_run("cur1", diagnostics=diagnostics))

    data = dd.get_current_scan_explorer(persistence_service, CONFIG, "cur1")

    assert data.coverage_evidence == 2
    assert data.coverage_rejected_by_agent == 1
    assert data.coverage_not_analyzed == 1  # the genuinely-skipped one only
