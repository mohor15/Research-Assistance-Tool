"""End-to-end (through the real LangGraph) proof that CURRENT's dedicated
candidate-selection policy (Part B1-B4) actually reaches FinancialNewsAgent,
and that Morning/Evening are completely unaffected -- complements the pure
unit tests in test_current_candidate_selection.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.agents.financial_news import FinancialNewsAgent
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, SourcePriority, SourceType
from research_assistant.pipeline import PipelineServices, PipelineState, build_graph
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition

from tests.pipeline.conftest import D, FakeRetrievalService, FakeTriageLLMClient, real_normalization_service
from tests.pipeline.test_graph import FakeLLMClient, _relevant_assessment
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative

# Real, already-configured secondary-source names (config/sources.yaml) --
# routing to FinancialNewsAgent is an exact source_name membership check
# against that registry (agents/router.py), so these must be real entries.
_SECONDARY_SOURCES = ["Reuters", "Wall Street Journal", "CNBC", "Barron's", "TechCrunch"]


def _secondary_source(name: str) -> SourceDefinition:
    return SourceDefinition(
        name=name, domain=f"{name.lower().replace(chr(39), '').replace(' ', '')}.example.com",
        urls=[f"https://{name.lower()}.example.com/ai"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
    )


def _article_for(name: str, i: int) -> ExtractedArticle:
    return ExtractedArticle(
        url=f"https://{name.lower().replace(' ', '')}.example.com/story-{i}",
        title=f"{name} story {i}: Bank announces major AI partnership",
        text=(
            f"{name} reported today (story {i}) that a major bank announced a new enterprise AI "
            f"partnership and deployed a generative AI platform across its trading and risk "
            f"operations, in development number {i} of its ongoing AI rollout. " * 3
        ),
        published_at=D - timedelta(hours=1),
    )


def _build_current_services(config, *, financial_news_responses):
    sources = [_secondary_source(name) for name in _SECONDARY_SOURCES]
    articles_by_source = {name: [_article_for(name, 0), _article_for(name, 1)] for name in _SECONDARY_SOURCES}
    retrieval_service = FakeRetrievalService(sources=sources, articles_by_source=articles_by_source)
    # The agent's OWN `._llm` is never used for CURRENT's bounded funnel
    # (deep analysis always passes the dedicated `deep_analysis_llm_client`
    # override -- see agents.base.ResearchAgent.analyze's `llm_client`
    # param) -- an empty FakeLLMClient here would raise loudly if that
    # assumption is ever wrong.
    agent = FinancialNewsAgent(FakeLLMClient([]), config)
    return PipelineServices(
        config=config,
        retrieval_service=retrieval_service,
        normalization_service=real_normalization_service(),
        agents={AgentName.FINANCIAL_NEWS: agent},
        event_service=EventService(config, clock=FixedClock(D)),
        report_agent=__import__(
            "research_assistant.agents.report", fromlist=["ReportAgent"]
        ).ReportAgent(ReportingFakeLLMClient([make_narrative()] * 10)),
        digest_service=DigestService(config, clock=FixedClock(D)),
        clock=FixedClock(D),
        triage_llm_client=FakeTriageLLMClient(),
        deep_analysis_llm_client=FakeLLMClient(financial_news_responses),
    )


def test_current_mode_applies_the_global_bounded_llm_funnel(config):
    """5 secondary sources x 2 articles = 10 eligible candidates. CURRENT's
    bounded funnel (config/settings.yaml current_llm) ranks and gates all
    10 (well under the 24-candidate triage cap), triages them in a single
    batch (<= 3 allowed), and deep-analyzes only up to the GLOBAL
    max_deep_analysis=4 budget -- never all 10, and never per-source/
    per-agent top_k (that mechanism no longer applies to CURRENT)."""
    config = config.model_copy(deep=True)
    services = _build_current_services(config, financial_news_responses=[_relevant_assessment()] * 10)

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=D - timedelta(hours=24), period_end=D)
    )

    metrics = result["metrics"]
    assert metrics["current_ranked_candidates"] == 10  # all 10 pass the deterministic gate
    assert metrics["candidates_before_llm"] == 10
    assert metrics["triage_batches_attempted"] == 4  # 10 candidates / triage_batch_size=3 -> ceil(10/3)=4 batches
    assert metrics["triage_relevant"] == 10  # fake triage marks everything relevant by default
    assert metrics["deep_analysis_budget"] == 4
    assert metrics["current_candidates_analyzed"] == 4  # GLOBAL cap, not per-source/per-agent
    assert metrics["llm_budget_exhausted"]  # PipelineState.metrics is Dict[str, int] -- True coerces to 1
    assert metrics["candidates_left_unprocessed"] == 6
    assert len(result["research_items"]) == 4


def test_morning_mode_is_unaffected_by_current_top_k(config):
    """The same 10-candidate pool, run as a plain Morning digest: top_k=5
    (not current_top_k) applies, and plain rank-order truncation (no
    diversity selection) is used -- CURRENT's policy must never leak into
    Morning/Evening."""
    config = config.model_copy(deep=True)
    services = _build_current_services(config, financial_news_responses=[_relevant_assessment()] * 5)

    result = build_graph(services).invoke(PipelineState())  # digest_type defaults to MORNING

    assert "current_candidates_analyzed" not in result["metrics"]
    assert result["metrics"]["agent_candidates_analyzed"] == 5
