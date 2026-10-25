"""Deterministic agent routing (`agents.router`): the fix for the real-run
regression where `agent_analysis: peer candidates=0` despite peer sources
being successfully retrieved, and for AI Product Agent inspecting articles
outside its domain. Routing must be a pure, LLM-free function of
Article + AppConfig.
"""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.agents.router import ROUTABLE_AGENTS, ArticleRouter, route_articles
from research_assistant.config import get_config
from research_assistant.models.article import Article
from research_assistant.models.enums import AgentName, SourcePriority, SourceType

CONFIG = get_config()


def _article(
    *, url: str, source_name: str, source_type: SourceType, title: str = "", text: str = ""
) -> Article:
    return Article(
        id=f"id-{url}",
        url=url,
        canonical_url=url,
        domain="example.com",
        source_name=source_name,
        source_type=source_type,
        source_priority=SourcePriority.HIGH,
        title=title,
        clean_text=text,
        retrieved_at=datetime.now(timezone.utc),
    )


def _routed(article: Article) -> set:
    result = route_articles([article], CONFIG)
    return {agent for agent in ROUTABLE_AGENTS if article in result.by_agent[agent]}


# ---------------------------------------------------------------------------
# A-D: primary-registry membership routing
# ---------------------------------------------------------------------------


def test_a_peer_article_routes_to_peer_only():
    article = _article(
        url="https://jpmorganchase.com/a", source_name="JPMorgan Chase", source_type=SourceType.PRIMARY,
        title="JPMorgan launches AI copilot", text="JPMorgan Chase rolled out a new generative AI copilot.",
    )
    assert _routed(article) == {AgentName.PEER}


def test_b_ai_provider_article_routes_to_ai_product_only():
    article = _article(
        url="https://openai.com/a", source_name="OpenAI", source_type=SourceType.PRIMARY,
        title="OpenAI releases a new model", text="OpenAI announced a new frontier model release today.",
    )
    assert _routed(article) == {AgentName.AI_PRODUCT}


def test_c_analyst_article_routes_to_analyst_only():
    article = _article(
        url="https://mckinsey.com/a", source_name="McKinsey", source_type=SourceType.PRIMARY,
        title="The state of enterprise AI", text="McKinsey's latest research on enterprise AI adoption.",
    )
    assert _routed(article) == {AgentName.ANALYST}


def test_d_startup_discovery_article_routes_to_startup():
    # TechCrunch is itself a configured secondary_sources member, so it ALSO
    # always routes to FinancialNewsAgent (case 3) -- independent of, and
    # additive to, the startup content-signal route checked here.
    article = _article(
        url="https://techcrunch.com/a", source_name="TechCrunch", source_type=SourceType.SECONDARY,
        title="AI infra startup raises funding",
        text="The AI infrastructure startup announced a Series B funding round led by a major VC firm.",
    )
    assert _routed(article) == {AgentName.STARTUP, AgentName.FINANCIAL_NEWS}


# ---------------------------------------------------------------------------
# E-H: secondary-source strong-signal routing
# ---------------------------------------------------------------------------


def test_e_reuters_article_mentioning_jpmorgan_routes_to_peer():
    # Reuters is itself a configured secondary_sources member, so it ALSO
    # always routes to FinancialNewsAgent (case 3), independent of the
    # peer content-signal match this test is really about.
    article = _article(
        url="https://reuters.com/a", source_name="Reuters", source_type=SourceType.SECONDARY,
        title="JPMorgan expands AI tooling for advisors", text="JPMorgan is rolling out new internal AI tools.",
    )
    assert _routed(article) == {AgentName.PEER, AgentName.FINANCIAL_NEWS}


def test_f_reuters_article_mentioning_anthropic_routes_to_ai_product():
    article = _article(
        url="https://reuters.com/b", source_name="Reuters", source_type=SourceType.SECONDARY,
        title="Anthropic unveils new safety research", text="Anthropic published new interpretability research.",
    )
    assert _routed(article) == {AgentName.AI_PRODUCT, AgentName.FINANCIAL_NEWS}


def test_g_reuters_article_mentioning_jpmorgan_and_anthropic_routes_to_both_not_startup_or_analyst():
    article = _article(
        url="https://reuters.com/c", source_name="Reuters", source_type=SourceType.SECONDARY,
        title="JPMorgan partners with Anthropic on AI tools",
        text="JPMorgan partners with Anthropic to bring generative AI tools to its advisors.",
    )
    assert _routed(article) == {AgentName.PEER, AgentName.AI_PRODUCT, AgentName.FINANCIAL_NEWS}


def test_h_generic_article_does_not_route_to_all_agents():
    # No peer/ai_provider/analyst/startup signal at all -- but Reuters is
    # still a configured secondary_sources member, so FinancialNewsAgent
    # (and ONLY FinancialNewsAgent) is the correct route: source-identity
    # membership for case 3 is unconditional, unlike the content-signal
    # routes for the other four agents.
    article = _article(
        url="https://reuters.com/d", source_name="Reuters", source_type=SourceType.SECONDARY,
        title="AI and banks", text="Banks are increasingly using AI and technology to improve finance operations.",
    )
    routed = _routed(article)
    assert routed != set(ROUTABLE_AGENTS)
    assert routed == {AgentName.FINANCIAL_NEWS}


def test_h2_secondary_source_earnings_article_is_not_stolen_by_startup_signal():
    """Confirmed real-run regression (2026-10-15): a genuine CNBC Q3-earnings
    story ("ASML ... profit smashes expectations") was misrouted to
    StartupAgent -- and rejected as irrelevant there -- because its body
    text happened to contain "growth", one of `config.sources.startups.
    funding_events`' bare, generic stage-label words. CNBC's own case-3
    FINANCIAL_NEWS route must still fire alongside the catch-all STARTUP
    signal (routing itself is additive, never suppressed), but
    `ROUTABLE_AGENTS` order must pick FINANCIAL_NEWS -- not STARTUP -- as
    the single `primary_agent` once the candidate pool dedupes."""
    article = _article(
        url="https://cnbc.com/asml", source_name="CNBC", source_type=SourceType.SECONDARY,
        title="ASML says it's getting order interest through 2030 as profit smashes expectations",
        text="ASML reported third-quarter profit that beat expectations, citing continued growth "
             "in AI-related chip demand from its customers.",
    )
    article.content_hash = "test-hash-asml"
    routed = _routed(article)
    assert routed == {AgentName.FINANCIAL_NEWS, AgentName.STARTUP}

    from research_assistant.pipeline.nodes import _build_current_candidate_pool
    from research_assistant.agents.router import RoutingResult
    from research_assistant.evidence.conversion import article_to_evidence

    routing = RoutingResult()
    routing.by_agent[AgentName.FINANCIAL_NEWS].append(article)
    routing.by_agent[AgentName.STARTUP].append(article)
    routing.multi_routed_count = 1
    evidence = article_to_evidence(article)

    pool = _build_current_candidate_pool(
        routing, {article.id: evidence}, {AgentName.FINANCIAL_NEWS, AgentName.STARTUP},
    )
    assert len(pool) == 1
    assert pool[0].primary_agent == AgentName.FINANCIAL_NEWS


def test_h3_pr_newswire_article_mentioning_a_peer_bank_still_picks_its_own_domain_as_primary():
    """Real bug fix (confirmed 2026-10-19, GMIF2026 memory/storage conference
    article): PR Newswire quoted "Daniel Yen of Morgan Stanley" once in
    passing, which matched PEER's content signal purely because Morgan
    Stanley is a configured peer bank. The OLD fixed `ROUTABLE_AGENTS`
    order (PEER before STARTUP) picked PEER as `primary_agent`, so the
    article was judged only against "is this about a peer bank?"
    (correctly: no) and never reached STARTUP -- PR Newswire's own
    configured `research_domain` and the actually relevant lens. The
    article's home agent (STARTUP, from `ArticleRouter.home_agent`) must
    now win regardless of `ROUTABLE_AGENTS` order."""
    article = _article(
        url="https://prnewswire.com/gmif", source_name="PR Newswire", source_type=SourceType.PRESS_DISTRIBUTION,
        title="GMIF2026: AI Inference Powers the Token Factory",
        text="Daniel Yen of Morgan Stanley projected data center capex trends at the memory summit.",
    )
    article.content_hash = "test-hash-gmif"
    routed = _routed(article)
    assert routed == {AgentName.PEER, AgentName.STARTUP}

    from research_assistant.evidence.conversion import article_to_evidence
    from research_assistant.pipeline.nodes import _build_current_candidate_pool

    result = route_articles([article], CONFIG)
    assert result.home_agent_by_article_id[article.id] == AgentName.STARTUP

    evidence = article_to_evidence(article)
    pool = _build_current_candidate_pool(result, {article.id: evidence}, {AgentName.PEER, AgentName.STARTUP})
    assert len(pool) == 1
    assert pool[0].primary_agent == AgentName.STARTUP


def test_h4_home_agent_is_none_for_a_pure_content_signal_route():
    """A non-member source (not peer/ai_provider/analyst/secondary_sources)
    routed purely by content signal has no configured identity at all --
    `home_agent` must be absent, and the old fixed-order fallback still
    decides (unchanged behavior for this case)."""
    article = _article(
        url="https://random-blog.example.com/a", source_name="Random Blog", source_type=SourceType.SECONDARY,
        title="JPMorgan and Anthropic", text="JPMorgan partners with Anthropic on new AI tools.",
    )
    result = route_articles([article], CONFIG)
    assert article.id not in result.home_agent_by_article_id


# ---------------------------------------------------------------------------
# I: provenance survives article -> evidence -> routing
# ---------------------------------------------------------------------------


def test_i_peer_source_metadata_survives_article_to_evidence_to_routing():
    from research_assistant.evidence.service import NormalizationService
    from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition

    source = SourceDefinition(
        name="JPMorgan Chase", domain="jpmorganchase.com", urls=["https://jpmorganchase.com/newsroom"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )
    extracted = ExtractedArticle(
        url="https://jpmorganchase.com/a", title="JPMorgan launches AI copilot",
        text="JPMorgan Chase today launched a new generative AI copilot for advisors.",
        published_at=datetime.now(timezone.utc),
    )
    service = NormalizationService()
    article = service.normalize(extracted, source=source)
    evidence = service.to_evidence(article)

    assert article.source_name == "JPMorgan Chase"
    assert evidence.article_id == article.id
    result = route_articles([article], CONFIG)
    assert article in result.by_agent[AgentName.PEER]


# ---------------------------------------------------------------------------
# J-L: routing mechanics
# ---------------------------------------------------------------------------


def test_j_routing_makes_zero_llm_calls():
    """route_articles/ArticleRouter never import or call into the llm package."""
    from pathlib import Path

    import research_assistant.agents.router as router_module

    lines = Path(router_module.__file__).read_text(encoding="utf-8").splitlines()
    import_lines = [line.strip() for line in lines if line.strip().startswith(("import ", "from "))]
    assert not any("llm" in line.lower() for line in import_lines)
    assert "generate_structured" not in router_module.__dict__

    # Behavioral confirmation: routing never touches an LLM client even
    # indirectly -- ArticleRouter/route_articles take only Article + AppConfig.
    import inspect

    assert "llm" not in inspect.signature(route_articles).parameters
    assert "llm" not in inspect.signature(ArticleRouter.__init__).parameters


def test_k_each_agent_receives_only_its_routed_subset():
    peer = _article(
        url="https://jpmorganchase.com/a", source_name="JPMorgan Chase", source_type=SourceType.PRIMARY,
    )
    ai_provider = _article(
        url="https://openai.com/a", source_name="OpenAI", source_type=SourceType.PRIMARY,
    )
    result = route_articles([peer, ai_provider], CONFIG)

    assert result.by_agent[AgentName.PEER] == [peer]
    assert result.by_agent[AgentName.AI_PRODUCT] == [ai_provider]
    assert peer not in result.by_agent[AgentName.AI_PRODUCT]
    assert ai_provider not in result.by_agent[AgentName.PEER]


def test_l_duplicate_article_ids_do_not_appear_twice_inside_the_same_bucket():
    article = _article(
        url="https://jpmorganchase.com/a", source_name="JPMorgan Chase", source_type=SourceType.PRIMARY,
    )
    result = route_articles([article, article], CONFIG)

    assert result.by_agent[AgentName.PEER].count(article) == 2  # caller-level dedup is normalize()'s job, not routing's
    # Routing itself never invents a second route for the same article beyond what route() returns once per call:
    router_result_for_one = route_articles([article], CONFIG)
    assert len(router_result_for_one.by_agent[AgentName.PEER]) == 1


def test_unrouted_and_multi_routed_counts_are_tracked():
    # Genuinely unrouted: NOT from any configured peer/ai_provider/analyst/
    # secondary_sources registry (unlike Reuters, which always routes to
    # FinancialNewsAgent by membership), and with no content signal either.
    unrouted = _article(
        url="https://random-blog.example.com/e", source_name="Random Blog", source_type=SourceType.SECONDARY,
        title="Weather report", text="It rained heavily across the region today.",
    )
    multi = _article(
        url="https://reuters.com/f", source_name="Reuters", source_type=SourceType.SECONDARY,
        title="JPMorgan partners with Anthropic", text="JPMorgan partners with Anthropic on new AI tools.",
    )
    result = route_articles([unrouted, multi], CONFIG)

    assert result.unrouted_count == 1
    assert result.multi_routed_count == 1
    assert result.by_agent[AgentName.FINANCIAL_NEWS] == [multi]
