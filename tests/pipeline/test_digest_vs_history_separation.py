"""Phase 9, end-to-end: article HISTORY (wide lookback window, persisted)
must stay separate from DIGEST CONTENT (narrow period_start/period_end
window, analyzed by agents and reported). Proves, through the real
LangGraph (not a single-stage unit test), that:

  * a historically-retained article never triggers an extra LLM call
  * a historically-retained article never leaks into the digest it
    doesn't belong to
  * an in-period article is completely unaffected
"""

from __future__ import annotations

from datetime import timedelta

from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.report import ReportAgent
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName
from research_assistant.pipeline import PipelineServices, PipelineState, build_graph
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import ExtractedArticle

from tests.pipeline.conftest import D, FakeRetrievalService, jpmorgan_source, real_normalization_service
from tests.pipeline.test_graph import FakeLLMClient, _relevant_assessment
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative

PERIOD_END = D
PERIOD_START = D - timedelta(hours=12)  # a 12h digest window, e.g. a morning digest


def _in_period_article() -> ExtractedArticle:
    return ExtractedArticle(
        url="https://jpmorganchase.com/in-period", title="JPMorgan launches AI copilot (in period)",
        text="JPMorgan Chase today launched a new generative AI copilot for financial advisors.",
        published_at=PERIOD_END - timedelta(hours=1),
    )


def _history_only_article() -> ExtractedArticle:
    """Published before this digest's own period_start, but still within
    the configured 24h lookback window (research.lookback_hours)."""
    return ExtractedArticle(
        url="https://jpmorganchase.com/history-only", title="JPMorgan announces AI partnership (history only)",
        text="JPMorgan Chase separately announced an AI partnership with a major cloud provider.",
        published_at=PERIOD_START - timedelta(hours=2),
    )


def _build_services(config, *, peer_llm_responses):
    peer_llm = FakeLLMClient(peer_llm_responses)

    return PipelineServices(
        config=config,
        retrieval_service=FakeRetrievalService(
            sources=[jpmorgan_source()],
            articles_by_source={"JPMorgan Chase": [_in_period_article(), _history_only_article()]},
        ),
        normalization_service=real_normalization_service(),
        agents={AgentName.PEER: PeerAgent(peer_llm, config)},
        event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
        digest_service=DigestService(config, clock=FixedClock(D)),
        clock=FixedClock(D),
    ), peer_llm


def test_history_only_article_is_persisted_but_never_analyzed_or_digested(config):
    services, peer_llm = _build_services(config, peer_llm_responses=[_relevant_assessment()])

    state = PipelineState(period_start=PERIOD_START, period_end=PERIOD_END, digest_type=DigestType.MORNING)
    result = build_graph(services).invoke(state)

    # 1. Both articles are retained as article history.
    article_urls = {a.url for a in result["articles"]}
    assert article_urls == {
        "https://jpmorganchase.com/in-period", "https://jpmorganchase.com/history-only",
    }

    # 2. Exactly ONE LLM call was made -- for the in-period article only.
    #    The history-only article never reached the agent at all.
    assert len(peer_llm.calls) == 1

    # 3. Exactly one research item / event, both attributable to the in-period article.
    assert len(result["research_items"]) == 1
    assert result["research_items"][0].url == "https://jpmorganchase.com/in-period"
    assert len(result["events"]) == 1

    # 4. The digest contains only that one event -- the history-only
    #    article's absence from analysis means it can never appear here.
    assert result["digest"].events == [result["events"][0].id]


def test_an_in_period_only_run_is_completely_unaffected(config):
    """Sanity check: with no history-only article at all, behavior is
    identical to before this change -- one article, one LLM call, one event."""
    peer_llm = FakeLLMClient([_relevant_assessment()])
    services = PipelineServices(
        config=config,
        retrieval_service=FakeRetrievalService(
            sources=[jpmorgan_source()], articles_by_source={"JPMorgan Chase": [_in_period_article()]}
        ),
        normalization_service=real_normalization_service(),
        agents={AgentName.PEER: PeerAgent(peer_llm, config)},
        event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
        digest_service=DigestService(config, clock=FixedClock(D)),
        clock=FixedClock(D),
    )

    state = PipelineState(period_start=PERIOD_START, period_end=PERIOD_END, digest_type=DigestType.MORNING)
    result = build_graph(services).invoke(state)

    assert len(result["articles"]) == 1
    assert len(peer_llm.calls) == 1
    assert len(result["events"]) == 1


def test_rerunning_the_same_retrieval_does_not_duplicate_article_history(config, tmp_path):
    """Re-running against the same source content (e.g. a retry, or a later
    run whose lookback window overlaps the previous one) must upsert, not
    duplicate -- the existing content-hash-derived deterministic Article id
    already guarantees this; this test proves it still holds for the wider
    history window introduced in this phase."""
    from research_assistant.storage import Database, PersistenceService

    database = Database(tmp_path / "rerun.db")
    database.initialize()
    persistence_service = PersistenceService(database)

    def _services():
        peer_llm = FakeLLMClient([_relevant_assessment()])
        return PipelineServices(
            config=config,
            retrieval_service=FakeRetrievalService(
                sources=[jpmorgan_source()],
                articles_by_source={"JPMorgan Chase": [_in_period_article(), _history_only_article()]},
            ),
            normalization_service=real_normalization_service(),
            agents={AgentName.PEER: PeerAgent(peer_llm, config)},
            event_service=EventService(config, clock=FixedClock(D)),
            report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()])),
            digest_service=DigestService(config, clock=FixedClock(D)),
            clock=FixedClock(D),
            persistence_service=persistence_service,
        )

    state = PipelineState(period_start=PERIOD_START, period_end=PERIOD_END, digest_type=DigestType.MORNING)
    build_graph(_services()).invoke(state)
    build_graph(_services()).invoke(state)  # identical re-run, fresh services each time (like a real retry)

    conn = database.connect()
    try:
        count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    finally:
        conn.close()

    assert count == 2  # not 4 -- upserted, never duplicated


def test_evening_digest_type_is_equally_unaffected_by_history_articles(config):
    """Evening digest semantics are untouched by this change -- same
    separation applies regardless of digest_type."""
    services, peer_llm = _build_services(config, peer_llm_responses=[_relevant_assessment()])

    state = PipelineState(period_start=PERIOD_START, period_end=PERIOD_END, digest_type=DigestType.EVENING)
    result = build_graph(services).invoke(state)

    assert len(peer_llm.calls) == 1
    assert len(result["events"]) == 1
    assert result["digest"].digest_type == DigestType.EVENING
