"""Source-aware, domain-specialized routing refactor.

Covers what changed: (1) `config/sources.yaml`'s per-source
`research_domain` field now sends Financial Times to the AI/Technology
specialist (`AgentName.AI_PRODUCT`) and PR Newswire to the Startup/Funding
specialist (`AgentName.STARTUP`) instead of the generic
`AgentName.FINANCIAL_NEWS`; (2) CURRENT's bounded triage funnel
(`pipeline.nodes._run_current_triage_by_domain`) never mixes more than one
specialist domain's candidates into a single LLM batch, and fairly splits
the bounded triage budget across domains instead of letting one domain
consume all of it. Fixtures/`FakeTriageLLMClient` only -- no live
Ollama/network anywhere."""

from __future__ import annotations

from datetime import timedelta

from research_assistant.agents.router import route_articles
from research_assistant.config import get_config
from research_assistant.evidence.conversion import article_to_evidence
from research_assistant.evidence.normalizer import normalize_article
from research_assistant.models.enums import AgentName, SourcePriority, SourceType
from research_assistant.pipeline.nodes import (
    _passes_current_deterministic_gate,
    _run_current_triage_by_domain,
)
from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition
from research_assistant.agents.triage import TriageCandidate

from tests.pipeline.conftest import D, FakeTriageLLMClient

CONFIG = get_config()


def _real_secondary_source(name: str) -> SourceDefinition:
    entry = next(s for s in CONFIG.sources.secondary_sources if s.name == name)
    return SourceDefinition(
        name=entry.name, domain=entry.domain, urls=entry.urls,
        source_type=entry.source_type, priority=entry.priority, category=SourceCategory.SECONDARY,
        search=entry.search,
    )


def _real_peer_source(name: str) -> SourceDefinition:
    entry = next(s for s in CONFIG.sources.peers if s.name == name)
    return SourceDefinition(
        name=entry.name, domain=entry.domain, urls=entry.urls,
        source_type=entry.source_type, priority=entry.priority, category=SourceCategory.PEER,
    )


def _real_analyst_source(name: str) -> SourceDefinition:
    entry = next(s for s in CONFIG.sources.analysts if s.name == name)
    return SourceDefinition(
        name=entry.name, domain=entry.domain, urls=entry.urls,
        source_type=entry.source_type, priority=entry.priority, category=SourceCategory.ANALYST,
    )


def _article(source: SourceDefinition, *, url: str, title: str, text: str):
    extracted = ExtractedArticle(url=url, title=title, text=text, published_at=D - timedelta(hours=1))
    return normalize_article(extracted, source=source)


_SUBSTANTIVE = " This is a substantive piece of reporting with enough body text to clear extraction thresholds." * 5


# ---------------------------------------------------------------------------
# A. Routing: each domain's real registry source routes to its own specialist
# ---------------------------------------------------------------------------


def test_financial_institution_source_routes_to_peer_agent():
    jpm = _real_peer_source("JPMorgan Chase")
    article = _article(jpm, url="https://jpmorganchase.com/a1", title="JPMorgan Chase deploys new AI platform", text=_SUBSTANTIVE)

    routing = route_articles([article], CONFIG)

    assert article.id in {a.id for a in routing.by_agent[AgentName.PEER]}


def test_ft_source_routes_to_ai_technology_agent_not_financial_news():
    ft = _real_secondary_source("Financial Times")
    article = _article(ft, url="https://www.ft.com/content/aaaa", title="Enterprise AI platform launch", text=_SUBSTANTIVE)

    routing = route_articles([article], CONFIG)

    assert article.id in {a.id for a in routing.by_agent[AgentName.AI_PRODUCT]}
    assert article.id not in {a.id for a in routing.by_agent.get(AgentName.FINANCIAL_NEWS, [])}


def test_pr_newswire_source_routes_to_startup_agent_not_financial_news():
    prn = _real_secondary_source("PR Newswire")
    article = _article(
        prn, url="https://www.prnewswire.com/news/release-1",
        title="AI startup announces Series C funding round", text=_SUBSTANTIVE,
    )

    routing = route_articles([article], CONFIG)

    assert article.id in {a.id for a in routing.by_agent[AgentName.STARTUP]}
    assert article.id not in {a.id for a in routing.by_agent.get(AgentName.FINANCIAL_NEWS, [])}


def test_unflagged_secondary_source_still_routes_to_financial_news():
    """Reuters has no `research_domain` override -- confirms the default
    (unchanged, pre-refactor) behavior still holds for every secondary
    source that doesn't opt into a more specific domain."""
    reuters = _real_secondary_source("Reuters")
    article = _article(reuters, url="https://www.reuters.com/a1", title="Bank announces AI deal", text=_SUBSTANTIVE)

    routing = route_articles([article], CONFIG)

    assert article.id in {a.id for a in routing.by_agent[AgentName.FINANCIAL_NEWS]}


def test_consulting_source_routes_to_analyst_agent():
    mckinsey = _real_analyst_source("McKinsey")
    article = _article(
        mckinsey, url="https://www.mckinsey.com/insights/1",
        title="How banks are operationalizing enterprise AI", text=_SUBSTANTIVE,
    )

    routing = route_articles([article], CONFIG)

    assert article.id in {a.id for a in routing.by_agent[AgentName.ANALYST]}


# ---------------------------------------------------------------------------
# B. Isolation: a specialist never receives another domain's candidate
# merely because of an incidental content signal
# ---------------------------------------------------------------------------


def test_ft_candidate_mentioning_a_bank_is_not_routed_to_peer_without_a_strong_signal():
    """A generic bank mention ("a major bank") is not one of the configured
    peer registry's own names/aliases -- it must not earn a PEER route."""
    ft = _real_secondary_source("Financial Times")
    article = _article(
        ft, url="https://www.ft.com/content/bbbb",
        title="A major bank explores generative AI for customer service", text=_SUBSTANTIVE,
    )

    routing = route_articles([article], CONFIG)

    assert article.id not in {a.id for a in routing.by_agent.get(AgentName.PEER, [])}
    assert article.id in {a.id for a in routing.by_agent[AgentName.AI_PRODUCT]}


def test_financial_institution_candidate_is_not_routed_to_startup_agent():
    jpm = _real_peer_source("JPMorgan Chase")
    article = _article(jpm, url="https://jpmorganchase.com/a2", title="JPMorgan Chase AI strategy update", text=_SUBSTANTIVE)

    routing = route_articles([article], CONFIG)

    assert article.id not in {a.id for a in routing.by_agent.get(AgentName.STARTUP, [])}


# ---------------------------------------------------------------------------
# C. Triage batches are domain-pure: no LLM call ever mixes domains
# ---------------------------------------------------------------------------


def _triage_candidate(primary_agent, *, candidate_id: str, source: SourceDefinition, title: str):
    article = _article(source, url=f"https://example.test/{candidate_id}", title=title, text=_SUBSTANTIVE)
    evidence = article_to_evidence(article)
    assert _passes_current_deterministic_gate(article, evidence)
    return TriageCandidate(candidate_id=article.id, article=article, evidence=evidence, primary_agent=primary_agent)


def _fake_secondary(name: str) -> SourceDefinition:
    return SourceDefinition(
        name=name, domain=f"{name.lower()}.example.com", urls=[],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
    )


def test_triage_batches_never_mix_candidates_from_more_than_one_domain():
    peer_pool = [
        _triage_candidate(AgentName.PEER, candidate_id=f"peer{i}", source=_fake_secondary("PeerSrc"), title=f"PEERDOMAIN story {i}")
        for i in range(3)
    ]
    ai_pool = [
        _triage_candidate(AgentName.AI_PRODUCT, candidate_id=f"ai{i}", source=_fake_secondary("AiSrc"), title=f"AIDOMAIN story {i}")
        for i in range(3)
    ]
    gated_pool = peer_pool + ai_pool
    client = FakeTriageLLMClient()
    diagnostics = []

    winners, metrics = _run_current_triage_by_domain(
        gated_pool, triage_client=client, batch_size=2, triage_pool_bound=20, diagnostics=diagnostics,
    )

    assert len(winners) == 6
    # Every prompt sent must contain exactly one domain's marker, never both.
    for prompt in client.calls:
        has_peer = "PEERDOMAIN" in prompt
        has_ai = "AIDOMAIN" in prompt
        assert has_peer != has_ai, f"a single triage batch mixed domains: {prompt[:200]}"


def test_triage_batches_carry_a_domain_specific_hint():
    pool = [
        _triage_candidate(AgentName.STARTUP, candidate_id=f"su{i}", source=_fake_secondary("StartupSrc"), title=f"Startup story {i}")
        for i in range(2)
    ]
    client = FakeTriageLLMClient()

    _run_current_triage_by_domain(pool, triage_client=client, batch_size=2, triage_pool_bound=10, diagnostics=[])

    assert len(client.calls) == 1
    assert "Startup/funding developments" in client.calls[0]


# ---------------------------------------------------------------------------
# D. Domain-fair budget allocation: one prolific domain cannot starve another
# ---------------------------------------------------------------------------


def test_domain_fair_allocation_represents_every_domain_under_a_tight_budget():
    """10 PEER candidates vs. 1 AI_PRODUCT candidate, triage budget of only
    4 -- plain rank-order truncation would let the prolific PEER domain take
    all 4 slots; domain-fair allocation must still give AI_PRODUCT its one
    candidate a slot."""
    peer_pool = [
        _triage_candidate(AgentName.PEER, candidate_id=f"peer{i}", source=_fake_secondary("PeerSrc"), title=f"PEERDOMAIN story {i}")
        for i in range(10)
    ]
    ai_pool = [
        _triage_candidate(AgentName.AI_PRODUCT, candidate_id="ai0", source=_fake_secondary("AiSrc"), title="AIDOMAIN story 0")
    ]
    gated_pool = peer_pool + ai_pool
    client = FakeTriageLLMClient()

    winners, metrics = _run_current_triage_by_domain(
        gated_pool, triage_client=client, batch_size=2, triage_pool_bound=4, diagnostics=[],
    )

    assert metrics["candidates_sent_to_triage"] == 4
    assert any(c.primary_agent == AgentName.AI_PRODUCT for c, _ in winners)
    assert metrics["candidates_truncated_before_triage"] == 11 - 4


def test_truncated_candidates_each_emit_a_budget_skip_diagnostic():
    """Instrumentation task (forensic finding, CURRENT run
    `e818e08adc2e4dd1`): the 7 candidates never offered a triage slot at
    all (truncated by the pool-bound budget, not rejected by triage) must
    each get their own `budget_skip:triage` diagnostic -- distinguishable
    from `triage:irrelevant`, which only ever fires for a candidate that
    actually received an LLM decision."""
    peer_pool = [
        _triage_candidate(AgentName.PEER, candidate_id=f"peer{i}", source=_fake_secondary("PeerSrc"), title=f"PEERDOMAIN story {i}")
        for i in range(10)
    ]
    ai_pool = [
        _triage_candidate(AgentName.AI_PRODUCT, candidate_id="ai0", source=_fake_secondary("AiSrc"), title="AIDOMAIN story 0")
    ]
    gated_pool = peer_pool + ai_pool
    client = FakeTriageLLMClient()
    diagnostics = []

    winners, metrics = _run_current_triage_by_domain(
        gated_pool, triage_client=client, batch_size=2, triage_pool_bound=4, diagnostics=diagnostics,
    )

    budget_skip_messages = [d.message for d in diagnostics if d.message.startswith("budget_skip:triage")]
    assert len(budget_skip_messages) == metrics["candidates_truncated_before_triage"] == 7

    triaged_ids = {c.candidate_id for c, _ in winners}
    for candidate in gated_pool:
        has_skip_diagnostic = any(f"candidate={candidate.candidate_id} " in m for m in budget_skip_messages)
        if candidate.candidate_id in triaged_ids:
            assert not has_skip_diagnostic, f"{candidate.candidate_id} was triaged but also got a budget_skip diagnostic"
        else:
            assert has_skip_diagnostic, f"{candidate.candidate_id} was truncated but got no budget_skip diagnostic"

    # No duplication against the existing triage:relevant diagnostics.
    triage_messages = [d.message for d in diagnostics if d.message.startswith("triage:")]
    assert len(triage_messages) == 4  # exactly the 4 candidates actually sent to triage


def test_domain_fair_allocation_with_a_single_domain_sends_every_candidate_up_to_the_bound():
    """With only one domain present, domain-fair allocation reduces to
    plain bounded selection -- the pre-refactor single-pool behavior."""
    pool = [
        _triage_candidate(AgentName.PEER, candidate_id=f"peer{i}", source=_fake_secondary("PeerSrc"), title=f"story {i}")
        for i in range(5)
    ]
    client = FakeTriageLLMClient()

    winners, metrics = _run_current_triage_by_domain(
        pool, triage_client=client, batch_size=2, triage_pool_bound=3, diagnostics=[],
    )

    assert metrics["candidates_sent_to_triage"] == 3
    assert len(winners) == 3
