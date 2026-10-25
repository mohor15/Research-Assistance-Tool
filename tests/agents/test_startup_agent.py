"""StartupAgent: funding/acquisition/partnership/product-launch/enterprise-deal,
irrelevant/ambiguous articles, unsupported-value pass-through, provenance,
malformed output, empty evidence, mocked Ollama failure — all via FakeLLMClient."""

from __future__ import annotations

from pathlib import Path

import pytest

import research_assistant.agents.startup as startup_module
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.agents.startup import StartupAgent
from research_assistant.llm.errors import LLMResponseError, LLMUnavailableError
from research_assistant.models.enums import EventType
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
)

from tests.agents.conftest import FakeLLMClient, make_article_and_evidence


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


def test_relevant_funding_article_produces_research_item(config, startup_source):
    article, evidence = make_article_and_evidence(
        startup_source, url="https://acmeai.example.com/press/series-b",
        title="Acme AI raises Series B",
        text=(
            "Acme AI, a financial AI startup, today announced it has raised "
            "$30 million in a Series B funding round led by a venture capital firm."
        ),
    )
    assessment = _assessment(
        RelevanceLabel.RELEVANT, "Concrete funding round with disclosed amount.",
        event_type=EventType.FUNDING, deal_value="$30 million",
    )
    agent = StartupAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.event_type == EventType.FUNDING
    assert item.deal_value == "$30 million"
    assert item.company == "Acme AI"


def test_irrelevant_startup_article_is_rejected(config, startup_source):
    article, evidence = make_article_and_evidence(
        startup_source, url="https://acmeai.example.com/press/career-day",
        title="Acme AI CEO speaks at career day",
        text=(
            "Acme AI's CEO gave a general talk about the history of artificial "
            "intelligence at a local high school career day, with no mention of "
            "the company's own products, plans, or business."
        ),
    )
    assessment = _assessment(RelevanceLabel.IRRELEVANT, "General talk, no company-specific development.")
    llm = FakeLLMClient([assessment])
    agent = StartupAgent(llm, config)

    item = agent.analyze(article, evidence)

    assert item is None
    assert len(llm.calls) == 1


def test_acquisition_article_produces_research_item(config, startup_source):
    article, evidence = make_article_and_evidence(
        startup_source, url="https://acmeai.example.com/press/acquired",
        title="Acme AI acquired",
        text=(
            "Acme AI, an enterprise AI startup, announced today that it has "
            "been acquired by a larger technology company for an undisclosed amount."
        ),
    )
    assessment = _assessment(
        RelevanceLabel.RELEVANT, "Confirmed acquisition.", event_type=EventType.ACQUISITION,
    )
    agent = StartupAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.event_type == EventType.ACQUISITION


def test_partnership_article_produces_research_item(config, startup_source):
    article, evidence = make_article_and_evidence(
        startup_source, url="https://acmeai.example.com/press/partnership",
        title="Acme AI partners with major bank",
        text=(
            "Acme AI announced a new partnership with a major bank to deploy "
            "its AI agents platform for customer service automation."
        ),
    )
    assessment = _assessment(
        RelevanceLabel.RELEVANT, "Named partnership with a financial institution.",
        event_type=EventType.PARTNERSHIP, partnerships=["a major bank"],
    )
    agent = StartupAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.event_type == EventType.PARTNERSHIP
    assert item.partnerships == ["a major bank"]


def test_product_launch_article_produces_research_item(config, startup_source):
    article, evidence = make_article_and_evidence(
        startup_source, url="https://acmeai.example.com/press/launch",
        title="Acme AI launches new platform",
        text=(
            "Acme AI launched a new AI infrastructure platform today, enabling "
            "enterprises to deploy custom foundation models more efficiently."
        ),
    )
    assessment = _assessment(
        RelevanceLabel.RELEVANT, "Concrete product launch.", event_type=EventType.PRODUCT_LAUNCH,
        products=["AI infrastructure platform"],
    )
    agent = StartupAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.event_type == EventType.PRODUCT_LAUNCH
    assert item.products == ["AI infrastructure platform"]


def test_enterprise_deal_article_produces_research_item(config, startup_source):
    article, evidence = make_article_and_evidence(
        startup_source, url="https://acmeai.example.com/press/deal",
        title="Acme AI signs enterprise deal",
        text=(
            "Acme AI signed a new enterprise deal with a Fortune 500 financial "
            "institution to deploy its AI agents platform."
        ),
    )
    assessment = _assessment(
        RelevanceLabel.RELEVANT, "Named enterprise deal.", event_type=EventType.ENTERPRISE_DEAL,
    )
    agent = StartupAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.event_type == EventType.ENTERPRISE_DEAL


def test_ambiguous_startup_article_is_surfaced_not_dropped(config, startup_source):
    article, evidence = make_article_and_evidence(
        startup_source, url="https://acmeai.example.com/press/roadmap",
        title="Acme AI discusses roadmap",
        text=(
            "Acme AI executives discussed the company's AI research roadmap at "
            "a conference, without disclosing specific products, customers, or timelines."
        ),
    )
    assessment = _assessment(
        RelevanceLabel.AMBIGUOUS, "Mentions AI research direction but no concrete development disclosed."
    )
    llm = FakeLLMClient([assessment])
    agent = StartupAgent(llm, config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.relevance.label == RelevanceLabel.AMBIGUOUS
    assert len(llm.calls) == 1


def test_unsupported_funding_amount_is_left_null_not_guessed(config, startup_source):
    article, evidence = make_article_and_evidence(
        startup_source, url="https://acmeai.example.com/press/undisclosed-round",
        title="Acme AI closes funding round",
        text=(
            "Acme AI announced today that it has closed a new funding round, "
            "without disclosing the amount raised or the names of investors."
        ),
    )
    # A well-behaved LLM leaves deal_value null when the evidence doesn't state one.
    assessment = _assessment(
        RelevanceLabel.RELEVANT, "Funding event, but no amount disclosed.",
        event_type=EventType.FUNDING, deal_value=None,
    )
    agent = StartupAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.deal_value is None


def test_unsupported_partnership_or_customer_is_left_empty_not_guessed(config, startup_source):
    article, evidence = make_article_and_evidence(
        startup_source, url="https://acmeai.example.com/press/strategic-initiative",
        title="Acme AI announces strategic initiative",
        text=(
            "Acme AI announced a new strategic initiative around agentic AI for "
            "financial services, without naming any specific partners or customers."
        ),
    )
    assessment = _assessment(
        RelevanceLabel.RELEVANT, "Strategic initiative, no named partners disclosed.",
        event_type=EventType.STRATEGIC_INITIATIVE, partnerships=[], entities=[],
    )
    agent = StartupAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.partnerships == []
    assert item.entities == []


def test_provenance_is_preserved_on_the_research_item(config, startup_source):
    article, evidence = make_article_and_evidence(
        startup_source, url="https://acmeai.example.com/press/series-b",
        title="Acme AI raises Series B",
        text="Acme AI, a financial AI startup, raised $30 million in a Series B round.",
    )
    assessment = _assessment(RelevanceLabel.RELEVANT, "Funding round.", event_type=EventType.FUNDING)
    agent = StartupAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item.article_id == article.id
    assert item.evidence == [evidence.evidence_id]
    assert item.source_quality.source_type == article.source_type
    assert item.source_quality.source_priority == article.source_priority
    assert item.company == "Acme AI"


def test_malformed_llm_output_raises_llm_response_error(config, startup_source):
    article, evidence = make_article_and_evidence(
        startup_source, url="https://acmeai.example.com/press/series-b",
        title="Acme AI raises Series B",
        text="Acme AI, a financial AI startup, raised $30 million in a Series B round.",
    )
    agent = StartupAgent(FakeLLMClient([LLMResponseError("model returned malformed JSON")]), config)

    with pytest.raises(LLMResponseError):
        agent.analyze(article, evidence)


def test_empty_evidence_content_is_handled_gracefully_without_calling_llm(config, startup_source):
    article, evidence = make_article_and_evidence(
        startup_source, url="https://acmeai.example.com/press/x", title="x", text="A brief note.",
    )
    llm = FakeLLMClient([])  # no AI keywords in "A brief note." -> pre-filter should short-circuit
    agent = StartupAgent(llm, config)

    item = agent.analyze(article, evidence)

    assert item is None
    assert len(llm.calls) == 0


def test_mocked_ollama_failure_propagates_cleanly(config, startup_source):
    article, evidence = make_article_and_evidence(
        startup_source, url="https://acmeai.example.com/press/series-b",
        title="Acme AI raises Series B",
        text="Acme AI, a financial AI startup, raised $30 million in a Series B round.",
    )
    agent = StartupAgent(FakeLLMClient([LLMUnavailableError("Ollama not running")]), config)

    with pytest.raises(LLMUnavailableError):
        agent.analyze(article, evidence)


def test_supports_accepts_any_article_discovery_driven(config, startup_source):
    agent = StartupAgent(FakeLLMClient([]), config)
    article, _ = make_article_and_evidence(startup_source, url="https://acmeai.example.com/x", title="t", text="AI news.")
    assert agent.supports(article) is True


def test_no_hardcoded_startup_universe_in_module_source():
    """StartupAgent must not embed a giant hardcoded company list."""
    source = Path(startup_module.__file__).read_text(encoding="utf-8")
    assert "monitored_names=None" in source  # discovery-driven, not a fixed registry
    assert "JPMorgan" not in source
    assert "OpenAI" not in source


def test_startup_agent_never_imports_retrieval_or_makes_http_calls():
    lines = Path(startup_module.__file__).read_text(encoding="utf-8").splitlines()
    import_lines = [line.strip() for line in lines if line.strip().startswith(("import ", "from "))]
    assert not any("research_assistant.retrieval" in line for line in import_lines)
    assert not any(line == "import requests" for line in import_lines)


def test_config_driven_keywords_include_configured_startup_categories_and_funding_events(config):
    agent = StartupAgent(FakeLLMClient([]), config)
    assert "financial ai" in agent._keywords
    assert "series a" in agent._keywords  # normalized from config's "series_a"
    assert "enterprise deal" in agent._keywords  # normalized from config's "enterprise_deal"
