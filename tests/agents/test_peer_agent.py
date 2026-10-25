"""PeerAgent: relevant/irrelevant/ambiguous articles, evidence linkage, no scraping,
LLM error propagation, empty/minimal evidence, multiple articles."""

from __future__ import annotations

from pathlib import Path

import pytest

import research_assistant.agents.base as base_module
import research_assistant.agents.peer as peer_module
from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.llm.errors import LLMUnavailableError
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
)

from tests.agents.conftest import FakeLLMClient, make_article_and_evidence

RELEVANT_TEXT = (
    "JPMorgan Chase today announced the launch of a new generative AI copilot "
    "for its financial advisors, built on internal infrastructure, to help "
    "draft client research summaries and speed up portfolio reviews."
)

NO_KEYWORDS_TEXT = (
    "JPMorgan Chase announced today that its quarterly charitable giving "
    "program donated $2 million to local community food banks across the "
    "United States this holiday season."
)

# Contains an AI keyword ("generative AI") so it passes the cheap pre-filter,
# but is the kind of vague trend-commentary the LLM itself should still
# judge irrelevant — distinct from NO_KEYWORDS_TEXT, which never reaches the LLM.
VAGUE_AI_MENTION_TEXT = (
    "JPMorgan Chase published a blog post today discussing the broader "
    "industry trend toward generative AI adoption across the banking "
    "sector, without disclosing any of the bank's own specific initiatives, "
    "products, or strategic decisions."
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


def test_relevant_peer_article_produces_research_item(config, peer_source):
    article, evidence = make_article_and_evidence(
        peer_source, url="https://www.jpmorganchase.com/newsroom/ai-copilot",
        title="JPMorgan launches AI copilot", text=RELEVANT_TEXT,
    )
    assessment = _assessment(
        RelevanceLabel.RELEVANT, "Concrete GenAI copilot launch for advisors.",
        summary="JPMorgan launched a GenAI copilot for financial advisors.",
    )
    agent = PeerAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.relevance.label == RelevanceLabel.RELEVANT
    assert item.company == "JPMorgan Chase"
    assert item.category == TopicCategory.ENTERPRISE_AI


def test_irrelevant_peer_article_is_rejected(config, peer_source):
    """LLM-driven rejection: the text mentions AI (passes the pre-filter) but
    the LLM itself judges it a non-event — distinct from the pre-filter test below."""
    article, evidence = make_article_and_evidence(
        peer_source, url="https://www.jpmorganchase.com/newsroom/trend-piece",
        title="JPMorgan discusses generative AI industry trends", text=VAGUE_AI_MENTION_TEXT,
    )
    assessment = _assessment(RelevanceLabel.IRRELEVANT, "Vague trend commentary, no concrete initiative disclosed.")
    llm = FakeLLMClient([assessment])
    agent = PeerAgent(llm, config)

    item = agent.analyze(article, evidence)

    assert item is None
    assert len(llm.calls) == 1  # confirms this went through the LLM, not just the pre-filter


def test_ambiguous_peer_article_is_surfaced_not_dropped(config, peer_source):
    article, evidence = make_article_and_evidence(
        peer_source, url="https://www.jpmorganchase.com/newsroom/digital",
        title="JPMorgan discusses generative AI adoption",
        text=(
            "JPMorgan Chase executives discussed the bank's broader adoption of "
            "generative AI and AI agents at an industry conference, without "
            "disclosing specific product names, partners, or initiatives."
        ),
    )
    assessment = _assessment(
        RelevanceLabel.AMBIGUOUS, "Mentions generative AI and AI agents but no concrete initiative is described."
    )
    llm = FakeLLMClient([assessment])
    agent = PeerAgent(llm, config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.relevance.label == RelevanceLabel.AMBIGUOUS
    assert len(llm.calls) == 1  # confirms this went through the LLM, not the pre-filter


def test_evidence_linkage_is_preserved_on_the_research_item(config, peer_source):
    article, evidence = make_article_and_evidence(
        peer_source, url="https://www.jpmorganchase.com/newsroom/ai-copilot",
        title="JPMorgan launches AI copilot", text=RELEVANT_TEXT,
    )
    assessment = _assessment(RelevanceLabel.RELEVANT, "Concrete GenAI launch.")
    agent = PeerAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item.article_id == article.id
    assert item.evidence == [evidence.evidence_id]
    assert item.source_quality.source_type == article.source_type
    assert item.source_quality.source_priority == article.source_priority


def test_agent_never_imports_retrieval_or_makes_http_calls():
    """Agents reason over supplied Article/Evidence only — they must never
    scrape a webpage themselves. Checks actual import statements (not just
    any substring) so a docstring mentioning the retrieval package by name
    doesn't trip a false positive."""
    for module in (base_module, peer_module):
        lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
        import_lines = [line.strip() for line in lines if line.strip().startswith(("import ", "from "))]
        assert not any("research_assistant.retrieval" in line for line in import_lines)
        assert not any(line == "import requests" or line.startswith("import requests ") for line in import_lines)


def test_llm_unavailable_error_propagates_cleanly(config, peer_source):
    article, evidence = make_article_and_evidence(
        peer_source, url="https://www.jpmorganchase.com/newsroom/ai-copilot",
        title="JPMorgan launches AI copilot", text=RELEVANT_TEXT,
    )
    agent = PeerAgent(FakeLLMClient([LLMUnavailableError("Ollama not running")]), config)

    with pytest.raises(LLMUnavailableError):
        agent.analyze(article, evidence)


def test_article_with_no_ai_keywords_is_rejected_by_prefilter_without_calling_llm(config, peer_source):
    article, evidence = make_article_and_evidence(
        peer_source, url="https://www.jpmorganchase.com/newsroom/charity",
        title="JPMorgan donates to food banks", text=NO_KEYWORDS_TEXT,
    )
    llm = FakeLLMClient([])  # no responses scripted — an LLM call would raise AssertionError
    agent = PeerAgent(llm, config)

    item = agent.analyze(article, evidence)

    assert item is None
    assert len(llm.calls) == 0  # pre-filter short-circuited before any LLM call


def test_minimal_evidence_content_does_not_crash(config, peer_source):
    article, evidence = make_article_and_evidence(
        peer_source, url="https://www.jpmorganchase.com/newsroom/x",
        title="AI", text="AI.",
    )
    assessment = _assessment(RelevanceLabel.IRRELEVANT, "Not enough substance to assess.")
    agent = PeerAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item is None


def test_multiple_articles_are_processed_independently_via_analyze_many(config, peer_source):
    relevant_article, relevant_evidence = make_article_and_evidence(
        peer_source, url="https://www.jpmorganchase.com/newsroom/ai-copilot",
        title="JPMorgan launches AI copilot", text=RELEVANT_TEXT,
    )
    irrelevant_article, irrelevant_evidence = make_article_and_evidence(
        peer_source, url="https://www.jpmorganchase.com/newsroom/trend-piece",
        title="JPMorgan discusses generative AI industry trends", text=VAGUE_AI_MENTION_TEXT,
    )
    agent = PeerAgent(
        FakeLLMClient([
            _assessment(RelevanceLabel.RELEVANT, "Concrete GenAI launch."),
            _assessment(RelevanceLabel.IRRELEVANT, "Vague trend commentary, no concrete initiative."),
        ]),
        config,
    )

    results = agent.analyze_many([(relevant_article, relevant_evidence), (irrelevant_article, irrelevant_evidence)])

    assert len(results) == 1
    assert results[0].article_id == relevant_article.id


def test_supports_restricts_to_configured_peer_banks(config, peer_source):
    agent = PeerAgent(FakeLLMClient([]), config)
    article, _ = make_article_and_evidence(peer_source, url="https://www.jpmorganchase.com/x", title="t", text="AI news.")
    assert agent.supports(article) is True

    from research_assistant.models.enums import SourcePriority, SourceType
    from research_assistant.retrieval.models import SourceCategory, SourceDefinition

    unrelated_source = SourceDefinition(
        name="Not A Peer Bank", domain="example.com", urls=[], source_type=SourceType.SECONDARY,
        priority=SourcePriority.LOW, category=SourceCategory.SECONDARY,
    )
    other_article, _ = make_article_and_evidence(unrelated_source, url="https://example.com/x", title="t", text="AI news.")
    assert agent.supports(other_article) is False


def test_mismatched_evidence_and_article_raises_value_error(config, peer_source):
    article_a, evidence_a = make_article_and_evidence(peer_source, url="https://www.jpmorganchase.com/a", title="A", text=RELEVANT_TEXT)
    article_b, evidence_b = make_article_and_evidence(peer_source, url="https://www.jpmorganchase.com/b", title="B", text=RELEVANT_TEXT)
    agent = PeerAgent(FakeLLMClient([]), config)

    with pytest.raises(ValueError):
        agent.analyze(article_a, evidence_b)
