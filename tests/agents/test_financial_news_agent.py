"""FinancialNewsAgent (final production gate "Live Current" extension):
relevant/irrelevant articles from configured secondary/financial-news
sources (Reuters, FT, ...), no primary-source-confirmation requirement,
evidence linkage, LLM error propagation, keyword pre-filter short-circuit."""

from __future__ import annotations

import pytest

from research_assistant.agents.financial_news import FinancialNewsAgent
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.llm.errors import LLMUnavailableError
from research_assistant.models.enums import AgentName
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
)

from tests.agents.conftest import FakeLLMClient, make_article_and_evidence, secondary_source_for

RELEVANT_AI_BANKING_TEXT = (
    "Reuters reported Monday that a major global bank has deployed a new "
    "generative AI system across its trading desks, citing people familiar "
    "with the rollout, in one of the largest AI infrastructure deployments "
    "in capital markets to date."
)

RELEVANT_FUNDING_TEXT = (
    "The Financial Times reported that an AI infrastructure startup has "
    "raised a $400 million Series D round led by a major growth-equity "
    "firm, valuing the company at $6 billion, to expand its inference "
    "compute capacity for enterprise customers."
)

GENERIC_AI_EXPLAINER_TEXT = (
    "Artificial intelligence, or AI, refers to computer systems that can "
    "perform tasks normally requiring human intelligence. This explainer "
    "covers the basic history and definitions of AI for general readers."
)

NO_KEYWORDS_TEXT = (
    "The city council approved a new public park renovation budget this "
    "week, with construction expected to begin next spring."
)


def _assessment(label: RelevanceLabel, reason: str, **overrides) -> AgentAssessment:
    kwargs = dict(
        relevance=RelevanceDecision(label=label, reason=reason),
        category=TopicCategory.ENTERPRISE_AI,
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
    )
    kwargs.update(overrides)
    return AgentAssessment(**kwargs)


def test_agent_name_is_financial_news(config):
    agent = FinancialNewsAgent(FakeLLMClient([]), config)
    assert agent.agent_name == AgentName.FINANCIAL_NEWS


def test_relevant_reuters_ai_banking_article_produces_research_item(config):
    source = secondary_source_for(config, "Reuters")
    article, evidence = make_article_and_evidence(
        source, url="https://www.reuters.com/technology/bank-ai-deployment",
        title="Major bank deploys generative AI across trading desks", text=RELEVANT_AI_BANKING_TEXT,
    )
    assessment = _assessment(
        RelevanceLabel.RELEVANT, "Substantive AI infrastructure deployment in capital markets, independently reported.",
    )
    agent = FinancialNewsAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.relevance.label == RelevanceLabel.RELEVANT
    assert item.company == "Reuters"


def test_relevant_ft_funding_article_produces_research_item(config):
    source = secondary_source_for(config, "Financial Times")
    article, evidence = make_article_and_evidence(
        source, url="https://www.ft.com/content/ai-infra-funding",
        title="AI infrastructure startup raises $400 million", text=RELEVANT_FUNDING_TEXT,
    )
    assessment = _assessment(RelevanceLabel.RELEVANT, "Substantive, dated funding event for AI infrastructure.")
    agent = FinancialNewsAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.relevance.label == RelevanceLabel.RELEVANT
    assert item.company == "Financial Times"


def test_no_primary_source_confirmation_required(config):
    """The defining property of this agent (vs. Peer/AIProduct/Analyst):
    a Reuters article about a bank's AI deployment is valid intelligence
    on its own -- nothing here checks whether the bank's own newsroom also
    published something. Confirmed by construction: `make_article_and_evidence`
    never touches any other source/registry at all."""
    source = secondary_source_for(config, "Reuters")
    article, evidence = make_article_and_evidence(
        source, url="https://www.reuters.com/technology/bank-ai-deployment-2",
        title="Bank deploys AI agents in operations", text=RELEVANT_AI_BANKING_TEXT,
    )
    assessment = _assessment(RelevanceLabel.RELEVANT, "Independently reported, substantive AI deployment.")
    agent = FinancialNewsAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)
    assert item is not None


def test_generic_ai_explainer_is_judged_irrelevant_by_the_llm(config):
    source = secondary_source_for(config, "Reuters")
    article, evidence = make_article_and_evidence(
        source, url="https://www.reuters.com/technology/what-is-ai",
        title="What is artificial intelligence?", text=GENERIC_AI_EXPLAINER_TEXT,
    )
    assessment = _assessment(RelevanceLabel.IRRELEVANT, "Generic AI explainer with no substantive new development.")
    llm = FakeLLMClient([assessment])
    agent = FinancialNewsAgent(llm, config)

    item = agent.analyze(article, evidence)

    assert item is None
    assert len(llm.calls) == 1  # went through the LLM, not just the pre-filter


def test_article_with_no_ai_or_financial_keywords_is_rejected_by_prefilter(config):
    source = secondary_source_for(config, "Reuters")
    article, evidence = make_article_and_evidence(
        source, url="https://www.reuters.com/world/park-renovation",
        title="City approves park renovation budget", text=NO_KEYWORDS_TEXT,
    )
    llm = FakeLLMClient([])  # no responses scripted -- an LLM call would raise AssertionError
    agent = FinancialNewsAgent(llm, config)

    item = agent.analyze(article, evidence)

    assert item is None
    assert len(llm.calls) == 0


def test_llm_unavailable_error_propagates_cleanly(config):
    source = secondary_source_for(config, "Reuters")
    article, evidence = make_article_and_evidence(
        source, url="https://www.reuters.com/technology/bank-ai-deployment-3",
        title="Bank deploys AI", text=RELEVANT_AI_BANKING_TEXT,
    )
    agent = FinancialNewsAgent(FakeLLMClient([LLMUnavailableError("Ollama not running")]), config)

    with pytest.raises(LLMUnavailableError):
        agent.analyze(article, evidence)


def test_supports_restricts_to_configured_secondary_sources(config):
    agent = FinancialNewsAgent(FakeLLMClient([]), config)
    reuters_article, _ = make_article_and_evidence(
        secondary_source_for(config, "Reuters"), url="https://www.reuters.com/x", title="t", text="AI news.",
    )
    assert agent.supports(reuters_article) is True

    from research_assistant.models.enums import SourcePriority, SourceType
    from research_assistant.retrieval.models import SourceCategory, SourceDefinition

    peer_bank_source = SourceDefinition(
        name="JPMorgan Chase", domain="jpmorganchase.com", urls=[], source_type=SourceType.PRIMARY,
        priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )
    peer_article, _ = make_article_and_evidence(peer_bank_source, url="https://jpmorganchase.com/x", title="t", text="AI news.")
    assert agent.supports(peer_article) is False
