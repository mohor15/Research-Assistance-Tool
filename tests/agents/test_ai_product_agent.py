"""AIProductAgent: relevant/irrelevant AI provider articles, scope restriction,
and that the agent doesn't give special treatment to its own underlying model."""

from __future__ import annotations

from pathlib import Path

import research_assistant.agents.ai_product as ai_product_module
from research_assistant.agents.ai_product import AIProductAgent
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
)
from research_assistant.retrieval.models import SourceCategory, SourceDefinition

from tests.agents.conftest import FakeLLMClient, make_article_and_evidence

RELEVANT_TEXT = (
    "OpenAI today announced a new AI infrastructure partnership and released "
    "an updated reasoning models lineup with expanded context length and "
    "improved tool-use capabilities, alongside a new enterprise API tier."
)

VAGUE_TEXT = (
    "OpenAI published a blog post reflecting on the broader history of "
    "artificial intelligence research over the past decade, without "
    "announcing any new product, model, or partnership."
)


def _assessment(label: RelevanceLabel, reason: str) -> AgentAssessment:
    return AgentAssessment(
        relevance=RelevanceDecision(label=label, reason=reason),
        category=TopicCategory.MODELS,
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="x"),
    )


def test_relevant_ai_provider_article_produces_research_item(config, ai_provider_source):
    article, evidence = make_article_and_evidence(
        ai_provider_source, url="https://openai.com/news/new-model",
        title="OpenAI releases new reasoning model", text=RELEVANT_TEXT,
    )
    assessment = _assessment(RelevanceLabel.RELEVANT, "Concrete model release with new enterprise API tier.")
    agent = AIProductAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.company == "OpenAI"
    assert item.category == TopicCategory.MODELS


def test_irrelevant_ai_provider_article_is_rejected(config, ai_provider_source):
    article, evidence = make_article_and_evidence(
        ai_provider_source, url="https://openai.com/news/history-retrospective",
        title="A history of artificial intelligence research", text=VAGUE_TEXT,
    )
    assessment = _assessment(RelevanceLabel.IRRELEVANT, "Historical retrospective, no new product or announcement.")
    llm = FakeLLMClient([assessment])
    agent = AIProductAgent(llm, config)

    item = agent.analyze(article, evidence)

    assert item is None
    assert len(llm.calls) == 1


def test_supports_restricts_to_configured_ai_providers(config, ai_provider_source):
    agent = AIProductAgent(FakeLLMClient([]), config)
    article, _ = make_article_and_evidence(ai_provider_source, url="https://openai.com/x", title="t", text=RELEVANT_TEXT)
    assert agent.supports(article) is True

    peer_bank = SourceDefinition(
        name="JPMorgan Chase", domain="jpmorganchase.com", urls=[], source_type=SourceType.PRIMARY,
        priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )
    other_article, _ = make_article_and_evidence(peer_bank, url="https://jpmorganchase.com/x", title="t", text=RELEVANT_TEXT)
    assert agent.supports(other_article) is False


def test_ai_product_agent_never_imports_retrieval_or_makes_http_calls():
    source = Path(ai_product_module.__file__).read_text(encoding="utf-8")
    assert "research_assistant.retrieval" not in source
    assert "import requests" not in source


def test_system_prompt_instructs_no_special_treatment_of_the_agents_own_model(config):
    """The system prompt must explicitly guard against the agent going easy on
    (or being fixated on) Qwen just because it's the local model doing the analysis."""
    agent = AIProductAgent(FakeLLMClient([]), config)
    prompt_text = " ".join(agent._system_prompt.lower().split())
    assert "not specifically analyzing any particular model" in prompt_text
    assert "do not give special treatment" in prompt_text


def test_system_prompt_contains_anti_hallucination_rules(config):
    agent = AIProductAgent(FakeLLMClient([]), config)
    prompt_text = " ".join(agent._system_prompt.lower().split())  # collapse newlines/indentation
    assert "never invent facts" in prompt_text
    assert "use only the evidence" in prompt_text
    assert "leave `deal_value` null" in prompt_text
    assert "never estimate or fabricate a number" in prompt_text
