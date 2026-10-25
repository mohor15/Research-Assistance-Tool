"""Deterministic recall safeguard: Qwen's deep-analysis `relevance.label`
is an imperfect model judgment, not ground truth. An IRRELEVANT verdict on
a candidate that deterministic evidence establishes as clearly within the
product's configured intelligence scope must not silently discard it.

These tests exercise the real production functions directly
(`agents.base._broad_scope_category_hits`, `_resolve_relevance_override`)
plus the full `ResearchAgent.analyze()` integration path -- never just
prompt-string inspection. No live Ollama anywhere -- `FakeLLMClient` only.
"""

from __future__ import annotations

from research_assistant.agents.base import _broad_scope_category_hits, _resolve_relevance_override
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.agents.startup import StartupAgent
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


def _startup_source() -> SourceDefinition:
    """Same shape as the `startup_source` fixture, callable directly (a
    pytest fixture cannot be called outside injection)."""
    return SourceDefinition(
        name="Acme AI", domain="acmeai.example.com", urls=[],
        source_type=SourceType.PRIMARY, priority=SourcePriority.MEDIUM, category=SourceCategory.STARTUP,
    )

# The real JumpCloud release text (forensic investigation subject, run
# 8c605b0881e8edf0) -- genuinely dated ("today released") and genuinely
# touching multiple scope categories (AI agents, governance/identity,
# research findings, enterprise).
JUMPCLOUD_TEXT = (
    "LOUISVILLE, Colo., Oct. 7, 2026 /PRNewswire/ -- JumpCloud Inc. today released a new study, "
    "Enterprise AI Access Is Outpacing Control. The report finds AI agents are gaining access to "
    "critical systems faster than organizations are extending identity and governance controls. "
    "More than half of organizations now use AI agents that can change systems, permissions, or files."
)

# Thin, generic, single-category text: mentions "AI" but nothing else --
# must NOT trigger the override (fails the >=2-category requirement).
THIN_GENERIC_TEXT = "This company uses AI in some of its products."

# Multi-category but NOT dated -- an evergreen page describing AI agent
# governance/security capabilities in the present tense, no announcement
# language anywhere. Must NOT trigger the override (fails the dated-
# development requirement) even though it hits >=2 categories.
MULTI_CATEGORY_BUT_UNDATED_TEXT = (
    "Our platform provides enterprise AI agent governance, identity, and security "
    "capabilities for organizations adopting AI at scale."
)


def _irrelevant_assessment(**overrides) -> AgentAssessment:
    kwargs = dict(
        relevance=RelevanceDecision(label=RelevanceLabel.IRRELEVANT, reason="Qwen judged this not significant enough."),
        category=TopicCategory.ENTERPRISE_AI,
        is_dated_development=False,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
    )
    kwargs.update(overrides)
    return AgentAssessment(**kwargs)


# ---------------------------------------------------------------------------
# A. _broad_scope_category_hits -- the category-breadth signal
# ---------------------------------------------------------------------------


def test_jumpcloud_text_hits_at_least_two_distinct_scope_categories():
    hits = _broad_scope_category_hits(JUMPCLOUD_TEXT)
    assert len(hits) >= 2
    assert "ai_core" in hits
    assert "governance_security" in hits


def test_thin_generic_text_hits_fewer_than_two_categories():
    hits = _broad_scope_category_hits(THIN_GENERIC_TEXT)
    assert len(hits) < 2


def test_empty_text_hits_no_categories():
    assert _broad_scope_category_hits("") == set()
    assert _broad_scope_category_hits(None) == set()


# ---------------------------------------------------------------------------
# B. _resolve_relevance_override -- the full gate (categories AND dated)
# ---------------------------------------------------------------------------


def test_override_fires_for_a_genuinely_dated_multi_category_candidate():
    source = _startup_source()
    _article, evidence = make_article_and_evidence(
        source, url="https://example.com/jumpcloud", title="JumpCloud study", text=JUMPCLOUD_TEXT,
    )
    assessment = _irrelevant_assessment()

    fired, diagnostic = _resolve_relevance_override(assessment, evidence)

    assert fired is True
    assert diagnostic is not None
    assert "qwen_relevance=irrelevant" in diagnostic
    assert "deterministic_scope_signal=in_scope" in diagnostic
    assert "final_relevance=relevant" in diagnostic


def test_override_does_not_fire_for_thin_single_category_text():
    source = _startup_source()
    _article, evidence = make_article_and_evidence(
        source, url="https://example.com/thin", title="Thin article", text=THIN_GENERIC_TEXT,
    )
    assessment = _irrelevant_assessment()

    fired, diagnostic = _resolve_relevance_override(assessment, evidence)

    assert fired is False
    assert diagnostic is None


def test_override_does_not_fire_for_undated_evergreen_text_even_with_multiple_categories():
    """The >=2-category breadth requirement alone is not sufficient -- an
    evergreen capability description that happens to touch several
    categories must still not override without genuine announcement
    language, keeping the safeguard narrow."""
    source = _startup_source()
    _article, evidence = make_article_and_evidence(
        source, url="https://example.com/evergreen", title="Evergreen platform page",
        text=MULTI_CATEGORY_BUT_UNDATED_TEXT,
    )
    assert len(_broad_scope_category_hits(MULTI_CATEGORY_BUT_UNDATED_TEXT)) >= 2  # breadth alone IS satisfied
    assessment = _irrelevant_assessment()

    fired, diagnostic = _resolve_relevance_override(assessment, evidence)

    assert fired is False
    assert diagnostic is None


# ---------------------------------------------------------------------------
# C. Full ResearchAgent.analyze() integration -- the override must actually
# change the agent's real output, including persisted `relevance_override`.
# ---------------------------------------------------------------------------


def test_analyze_overrides_a_qwen_irrelevant_verdict_for_jumpcloud_like_content(config):
    source = _startup_source()
    article, evidence = make_article_and_evidence(
        source, url="https://example.com/jumpcloud", title="JumpCloud study", text=JUMPCLOUD_TEXT,
    )
    client = FakeLLMClient([_irrelevant_assessment()])
    agent = StartupAgent(client, config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.relevance.label == RelevanceLabel.RELEVANT
    assert item.relevance_override is not None
    assert "qwen_relevance=irrelevant" in item.relevance_override
    assert item.is_dated_development is True


def test_analyze_still_rejects_a_genuinely_irrelevant_article(config):
    """The safeguard must not become a blanket override -- a thin,
    single-category, Qwen-rejected article is still correctly dropped."""
    source = _startup_source()
    article, evidence = make_article_and_evidence(
        source, url="https://example.com/thin", title="Thin article", text=THIN_GENERIC_TEXT,
    )
    client = FakeLLMClient([_irrelevant_assessment()])
    agent = StartupAgent(client, config)

    item = agent.analyze(article, evidence)

    assert item is None


def test_analyze_never_sets_relevance_override_when_qwen_already_said_relevant(config):
    """`relevance_override` must stay None for the ordinary, non-disagreement
    path -- it is set ONLY when an override actually fired."""
    source = _startup_source()
    article, evidence = make_article_and_evidence(
        source, url="https://example.com/jumpcloud", title="JumpCloud study", text=JUMPCLOUD_TEXT,
    )
    relevant_assessment = AgentAssessment(
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="Clearly on-topic."),
        category=TopicCategory.ENTERPRISE_AI,
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="x"),
    )
    client = FakeLLMClient([relevant_assessment])
    agent = StartupAgent(client, config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.relevance_override is None
