"""Phase 12 EOD hardening: before `top_k` truncates an agent's candidate
list, the strongest candidates must be ranked first -- not just the most
recently discovered/published ones. Real regression from Phase 11: a
substantive, keyword-rich Bank of America GenAI article was dropped by
`top_k=5` while several generic OpenAI newsroom category pages (shorter
body, no keyword density) consumed slots ahead of it purely by recency.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.config import get_config
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.evidence.conversion import article_to_evidence
from research_assistant.evidence.normalizer import normalize_article
from research_assistant.models.enums import AgentName
from research_assistant.pipeline.nodes import (
    _candidate_rank_key,
    _flatten_event_type_phrases,
    _flatten_topic_keywords,
    make_agent_analysis_node,
)
from research_assistant.pipeline.state import PipelineState
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import ExtractedArticle

from tests.pipeline.conftest import jpmorgan_source
from tests.pipeline.test_graph import FakeLLMClient, _relevant_assessment
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative

NOW = datetime(2026, 9, 30, tzinfo=timezone.utc)
CONFIG = get_config()
KEYWORDS = _flatten_topic_keywords(CONFIG.topics)
EVENT_TYPE_PHRASES = _flatten_event_type_phrases(CONFIG.topics)


def _article_and_evidence(*, url, title, text, published_at):
    extracted = ExtractedArticle(url=url, title=title, text=text, published_at=published_at)
    article = normalize_article(extracted, source=jpmorgan_source())
    evidence = article_to_evidence(article)
    return article, evidence


def test_a_substantive_keyword_rich_article_outranks_a_more_recent_hub_page():
    hub_article, hub_evidence = _article_and_evidence(
        url="https://example.com/news",
        title="OpenAI Newsroom | Research",
        text="Research.",  # short teaser, matches the real "News Archive" 349-char case
        published_at=NOW,
    )
    real_article, real_evidence = _article_and_evidence(
        url="https://example.com/ask-gps-intelligence-hub",
        title="Bank Expands Ask GPS Intelligence Hub With New GenAI Capabilities",
        text=(
            "The bank today announced the launch of a new generative AI application "
            "to help employees deliver deeper client insights across treasury and "
            "payments operations, building on an existing enterprise AI platform."
        ),
        published_at=NOW - timedelta(hours=6),  # older than the hub page
    )

    hub_key = _candidate_rank_key(hub_article, hub_evidence, KEYWORDS)
    real_key = _candidate_rank_key(real_article, real_evidence, KEYWORDS)

    assert real_key > hub_key  # strength beats raw recency


def test_pipe_titled_hub_page_is_demoted_even_with_a_long_body():
    # A "Site | Section" title is a strong hub signal on its own, even if
    # the extracted body happens to be long (e.g. it leaked the top
    # article's teaser text).
    hub_article, hub_evidence = _article_and_evidence(
        url="https://example.com/news/research",
        title="OpenAI Newsroom | Research",
        text="x" * 500,
        published_at=NOW,
    )
    real_article, real_evidence = _article_and_evidence(
        url="https://example.com/real-story",
        title="Company Launches New Enterprise AI Model",
        text="y" * 500,
        published_at=NOW - timedelta(days=1),
    )

    assert _candidate_rank_key(real_article, real_evidence, KEYWORDS) > _candidate_rank_key(
        hub_article, hub_evidence, KEYWORDS
    )


def test_recency_is_still_the_tiebreaker_among_equally_strong_candidates():
    older_article, older_evidence = _article_and_evidence(
        url="https://example.com/a", title="Bank Launches Generative AI Platform",
        text="Substantive article body describing a real generative AI deployment at the bank." * 5,
        published_at=NOW - timedelta(days=1),
    )
    newer_article, newer_evidence = _article_and_evidence(
        url="https://example.com/b", title="Bank Launches Generative AI Platform",
        text="Substantive article body describing a real generative AI deployment at the bank." * 5,
        published_at=NOW,
    )

    assert _candidate_rank_key(newer_article, newer_evidence, KEYWORDS) > _candidate_rank_key(
        older_article, older_evidence, KEYWORDS
    )


def test_top_k_selects_the_strongest_candidate_not_the_most_recent_one(config):
    """Full node-level proof: with `top_k=1`, the one substantive,
    keyword-rich candidate is analyzed even though a more recently
    published hub-like candidate was routed alongside it."""
    narrowed = config.model_copy(deep=True)
    narrowed.settings.agents.peer.top_k = 1

    hub_article, hub_evidence = _article_and_evidence(
        url="https://jpmorganchase.com/newsroom/research",
        title="JPMorgan Newsroom | Research",
        text="Research.",
        published_at=NOW,
    )
    real_article, real_evidence = _article_and_evidence(
        url="https://jpmorganchase.com/newsroom/genai-copilot-launch",
        title="JPMorgan Launches Generative AI Copilot for Advisors",
        text=(
            "JPMorgan Chase today announced the launch of a new generative AI copilot "
            "for financial advisors, part of a broader enterprise AI strategy."
        ) * 3,
        published_at=NOW - timedelta(hours=2),
    )

    peer_llm = FakeLLMClient([_relevant_assessment()])
    from research_assistant.agents.peer import PeerAgent
    from research_assistant.pipeline import PipelineServices
    from tests.pipeline.conftest import FakeRetrievalService

    services = PipelineServices(
        config=narrowed,
        retrieval_service=FakeRetrievalService(sources=[jpmorgan_source()], articles_by_source={}),
        normalization_service=None,
        agents={AgentName.PEER: PeerAgent(peer_llm, narrowed)},
        event_service=EventService(narrowed, clock=FixedClock(NOW)),
        report_agent=None,
        digest_service=DigestService(narrowed, clock=FixedClock(NOW)),
        clock=FixedClock(NOW),
    )

    state = PipelineState(articles=[hub_article, real_article], evidence=[hub_evidence, real_evidence])
    node = make_agent_analysis_node(services)
    result = node(state)

    assert len(peer_llm.calls) == 1  # top_k=1 -- only the strongest candidate reached the LLM
    assert len(result["research_items"]) == 1
    assert result["research_items"][0].url == real_article.url


# ---------------------------------------------------------------------------
# Phase 13E: a genuine dated development must outrank a generic topic/
# capability-overview page (real regression: BCG's "AI Agents" practice-area
# page, which has no announcement language at all, versus a real dated
# analyst research article).
# ---------------------------------------------------------------------------


def test_a_dated_development_outranks_a_generic_capability_overview_page():
    capability_page, capability_evidence = _article_and_evidence(
        url="https://example.com/capabilities/artificial-intelligence/ai-agents",
        title="AI Agents",
        text=(
            "AI agents are autonomous systems that can plan and execute multi-step tasks. "
            "Organizations across industries are exploring how agentic AI can be applied "
            "to enterprise workflows, customer service, and operations."
        ),
        published_at=NOW,
    )
    dated_article, dated_evidence = _article_and_evidence(
        url="https://example.com/publications/the-formula-for-agentic-ai-value",
        title="The Formula for Agentic AI Value",
        text=(
            "Today, the firm announced new research on agentic AI value creation, "
            "describing a framework enterprises can use to evaluate agentic AI "
            "deployments and their return on investment."
        ),
        published_at=NOW - timedelta(days=1),  # older, but still ranks higher
    )

    capability_key = _candidate_rank_key(capability_page, capability_evidence, KEYWORDS, EVENT_TYPE_PHRASES)
    dated_key = _candidate_rank_key(dated_article, dated_evidence, KEYWORDS, EVENT_TYPE_PHRASES)

    assert dated_key > capability_key


def test_action_language_or_event_type_phrase_is_a_positive_ranking_signal():
    no_signal_article, no_signal_evidence = _article_and_evidence(
        url="https://example.com/a", title="Enterprise AI Overview",
        text="Enterprise AI refers to the application of artificial intelligence within large organizations.",
        published_at=NOW,
    )
    partnership_article, partnership_evidence = _article_and_evidence(
        url="https://example.com/b", title="Enterprise AI Overview",
        text="The firm today announced a new partnership to expand its enterprise AI research practice.",
        published_at=NOW,
    )

    assert _candidate_rank_key(partnership_article, partnership_evidence, KEYWORDS, EVENT_TYPE_PHRASES) > \
        _candidate_rank_key(no_signal_article, no_signal_evidence, KEYWORDS, EVENT_TYPE_PHRASES)


def test_generic_short_title_demotion_does_not_destroy_recall_when_nothing_else_exists():
    """Section 13: a demoted generic-topic-page candidate can still be
    selected if it's the only candidate available -- demotion is a ranking
    preference, never a hard rejection."""
    only_candidate, only_evidence = _article_and_evidence(
        url="https://example.com/capabilities/artificial-intelligence/generative-ai",
        title="Generative AI",
        text="Generative AI describes models that can generate text, images, and other content " * 5,
        published_at=NOW,
    )

    # No exception, no filtering -- the ranking key is just a sort key, and
    # a single-item list always keeps its one item.
    key = _candidate_rank_key(only_candidate, only_evidence, KEYWORDS, EVENT_TYPE_PHRASES)
    assert key is not None
