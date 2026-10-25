"""Deterministic agent routing: `Article -> {AgentName}`, entirely
config-driven and LLM-free.

This sits between evidence construction and per-agent analysis (see
`pipeline.nodes.make_agent_analysis_node`): it decides *which* agents a
given article is even eligible for, before any agent's own keyword
pre-filter or LLM relevance call ever runs. It replaces the previous
approach of asking each agent `agent.supports(article)` independently at
the pipeline layer — the routing *decision* now lives in one place, while
`ResearchAgent.supports()` itself is unchanged and still independently
correct/tested (see `tests/agents/test_*_agent.py`).

Two routing paths, in order:

1. Primary-source membership: an article retrieved from one of the
   configured peer/ai_provider/analyst sources (`config/sources.yaml`)
   always routes to that domain's agent. This is an exact `source_name`
   equality check against the registry — the same strings flow unchanged
   from `config.sources.*[].name` through `SourceDefinition.name` to
   `Article.source_name` (see `retrieval.service.build_source_definitions`
   and `evidence.normalizer.normalize_article`), so this can never
   mismatch due to a naming typo or transformation.

2. Secondary-source strong-signal detection: an article that did NOT match
   a peer/ai_provider/analyst registry by source-name membership only
   routes to one of THOSE THREE domain agents when its title/content
   contains a strong, specific signal for that domain: the configured
   company/firm's full name, or a short alias derived from it (its first
   word, when that word is distinctive enough — see
   `_GENERIC_FIRST_WORDS`). A generic category word ("AI", "bank",
   "technology", "finance") is never by itself a signal, since it is never
   one of these specific, configured names.

3. Secondary-source membership (financial/business news, Final Production
   Gate "Live Current" extension, specialized further by the
   source-aware-routing refactor): an article retrieved from one of the
   configured `secondary_sources` (Reuters, Financial Times, Bloomberg,
   CNBC, Barron's, Wall Street Journal, TechCrunch, The Information, PR
   Newswire) routes by the same exact source-name membership check as case
   1 -- never requiring primary-source confirmation, never depending on
   content signals -- but to WHICH agent is now config-driven per source
   (`config/sources.yaml`'s `research_domain` field, see
   `_RESEARCH_DOMAIN_TO_AGENT`), not hardcoded here: a source with no
   `research_domain` configured (the default; today Reuters, Bloomberg,
   CNBC, Barron's, WSJ, TechCrunch, The Information) still routes to
   `AgentName.FINANCIAL_NEWS` exactly as before, while Financial Times
   (`research_domain: ai_technology`) routes to `AgentName.AI_PRODUCT` and
   PR Newswire (`research_domain: startup_funding`) routes to
   `AgentName.STARTUP` instead -- each secondary source routes to exactly
   ONE agent this way (never both FINANCIAL_NEWS and its specialist), so a
   mixed, domain-blind candidate pool is never built from these sources.
   This is independent of, and additive to, case 2: a secondary-source
   article can ALSO route to Peer/AIProduct/Analyst when it independently
   carries a strong signal for one of them (e.g. a Reuters story naming a
   configured peer bank), so this case-3 route never suppresses that.

`StartupAgent` has no fixed source registry to check membership against —
`config/sources.yaml`'s `startups` section is discovery categories/funding
metadata, not named sources (see
`retrieval.service._SCRAPABLE_CATEGORIES`, which deliberately excludes
`startups`). It is therefore routed like a fourth "secondary-signal" case:
only a secondary/press-distribution article (the only place startup
coverage can arrive from in this registry) that *also* contains one of the
configured startup categories/funding-events/event-types (the same
vocabulary `StartupAgent`'s own keyword pre-filter already uses) earns a
route — a generic secondary article ("JPMorgan partners with Anthropic",
with no funding/acquisition/category signal) is not sent to Startup by
default, matching the same "strong, specific signal only" standard applied
to the other three agents.

An article may route to more than one agent when independent strong
signals for each are genuinely present (e.g. a Reuters story naming both a
configured peer bank and a configured AI provider).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from research_assistant.config.schema import AppConfig
from research_assistant.models.article import Article
from research_assistant.models.enums import AgentName, SourceType

# The only agents this module routes to. AgentName.REPORT is a synthesis
# agent, not a research-candidate target, and is deliberately excluded.
#
# FINANCIAL_NEWS is ordered BEFORE STARTUP deliberately (confirmed
# regression, 2026-10-15): StartupAgent's catch-all content signal (case 4
# above) is built from `config.sources.startups.categories`/`funding_events`
# which include bare generic words ("growth", "AI infrastructure") that
# routinely appear in ordinary earnings/economic articles with no startup
# angle at all (e.g. a CNBC ASML Q3-earnings story mentioning "growth").
# When such an article ALSO carries CNBC/Reuters/Bloomberg/etc.'s own
# explicit case-3 FINANCIAL_NEWS route, `_build_current_candidate_pool`'s
# "first match wins" dedup must prefer that explicit, source-configured
# route over the generic catch-all -- otherwise the article's only triage
# attempt goes to StartupAgent, which correctly judges it irrelevant to
# startup funding, and the article is lost even though FinancialNewsAgent
# would have found it relevant. This was confirmed against real CURRENT
# run data: ASML's Oct 14 earnings story, and two other legitimate
# CNBC financial-news articles, were all misrouted to `startup` this way
# and rejected as irrelevant.
ROUTABLE_AGENTS: List[AgentName] = [
    AgentName.PEER, AgentName.AI_PRODUCT, AgentName.ANALYST, AgentName.FINANCIAL_NEWS, AgentName.STARTUP,
]

_SECONDARY_SOURCE_TYPES = {SourceType.SECONDARY, SourceType.PRESS_DISTRIBUTION}

# config/sources.yaml's per-source `research_domain` string -> the one
# specialist agent a secondary/press-distribution source's articles route
# to (case 3 above). Any value not in this mapping (including the default
# `None`) falls back to `AgentName.FINANCIAL_NEWS` -- the original,
# pre-refactor behavior for every secondary source that doesn't opt into a
# more specific domain.
_RESEARCH_DOMAIN_TO_AGENT: Dict[str, AgentName] = {
    "financial_institutions": AgentName.PEER,
    "ai_technology": AgentName.AI_PRODUCT,
    "startup_funding": AgentName.STARTUP,
    "analyst": AgentName.ANALYST,
    "financial_news": AgentName.FINANCIAL_NEWS,
}

# A first-word alias is only used when it is NOT one of these generic
# corporate/category nouns -- these must never alone trigger routing.
_GENERIC_FIRST_WORDS = {
    "bank", "group", "financial", "holdings", "company", "corp", "inc",
    "corporation", "technologies", "technology", "systems", "solutions",
    "capital", "global", "international",
}
_MIN_ALIAS_LENGTH = 4


def _aliases_for(name: str) -> Set[str]:
    """The full configured name, plus (when safe) a short first-word alias
    -- e.g. "JPMorgan Chase" -> {"jpmorgan chase", "jpmorgan"}, but
    "Bank of America" -> {"bank of america"} only ("bank" is generic)."""
    aliases = {name.strip().lower()}
    words = name.split()
    if len(words) > 1:
        first = words[0].lower()
        if len(first) >= _MIN_ALIAS_LENGTH and first not in _GENERIC_FIRST_WORDS:
            aliases.add(first)
    return aliases


def _compile_matcher(names: Set[str]) -> Optional[re.Pattern]:
    all_aliases: Set[str] = set()
    for name in names:
        all_aliases |= _aliases_for(name)
    if not all_aliases:
        return None
    # Longest-first so a multi-word alias is preferred over a shorter one
    # it happens to contain (cosmetic for matching, not a correctness need
    # since every alias is matched at word boundaries independently).
    pattern = "|".join(re.escape(alias) for alias in sorted(all_aliases, key=len, reverse=True))
    return re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)


def _haystack(article: Article) -> str:
    return f"{article.title or ''}\n{article.clean_text or ''}"


def _compile_phrase_matcher(phrases: List[str]) -> Optional[re.Pattern]:
    """Like `_compile_matcher`, but for already-specific phrases (not
    company names needing alias derivation) — used for the startup/funding
    signal, built directly from `config.sources.startups`."""
    normalized = {p.strip().lower() for p in phrases if p.strip()}
    if not normalized:
        return None
    pattern = "|".join(re.escape(p) for p in sorted(normalized, key=len, reverse=True))
    return re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)


class ArticleRouter:
    """Built once per pipeline run from `AppConfig`; reused across every
    article in that run. Construction is pure config parsing — no I/O, no
    LLM call, ever."""

    def __init__(self, config: AppConfig) -> None:
        self._peer_names = {s.name for s in config.sources.peers}
        self._ai_provider_names = {s.name for s in config.sources.ai_providers}
        self._analyst_names = {s.name for s in config.sources.analysts}
        self._secondary_names = {s.name for s in config.sources.secondary_sources}
        self._secondary_domain_agent: Dict[str, AgentName] = {
            s.name: _RESEARCH_DOMAIN_TO_AGENT.get(s.research_domain, AgentName.FINANCIAL_NEWS)
            for s in config.sources.secondary_sources
        }

        self._peer_matcher = _compile_matcher(self._peer_names)
        self._ai_provider_matcher = _compile_matcher(self._ai_provider_names)
        self._analyst_matcher = _compile_matcher(self._analyst_names)

        startups = config.sources.startups
        startup_phrases = [p.replace("_", " ") for p in (list(startups.categories) + list(startups.funding_events) + list(startups.event_types))]
        self._startup_matcher = _compile_phrase_matcher(startup_phrases)

    def route(self, article: Article) -> Set[AgentName]:
        agents: Set[AgentName] = set()
        primary_matched = False

        if article.source_name in self._peer_names:
            agents.add(AgentName.PEER)
            primary_matched = True
        if article.source_name in self._ai_provider_names:
            agents.add(AgentName.AI_PRODUCT)
            primary_matched = True
        if article.source_name in self._analyst_names:
            agents.add(AgentName.ANALYST)
            primary_matched = True

        if not primary_matched:
            # Not from a configured peer/ai_provider/analyst registry --
            # only a strong, specific content signal earns a route to one
            # of THOSE THREE agents; never assumed. (Checked on
            # `primary_matched`, not on `agents` being non-empty, so that
            # the FinancialNewsAgent membership route below -- which is
            # itself independent of these three -- never suppresses this
            # fallback for a secondary-source article.)
            haystack = _haystack(article)
            if self._peer_matcher and self._peer_matcher.search(haystack):
                agents.add(AgentName.PEER)
            if self._ai_provider_matcher and self._ai_provider_matcher.search(haystack):
                agents.add(AgentName.AI_PRODUCT)
            if self._analyst_matcher and self._analyst_matcher.search(haystack):
                agents.add(AgentName.ANALYST)

        if article.source_name in self._secondary_domain_agent:
            # Reuters/FT/Bloomberg/CNBC/etc. -- exact membership, same
            # standard as case 1, independent of the above: never requires
            # primary-source confirmation, and never suppressed by (or
            # suppressing) a peer/ai_provider/analyst content-signal match.
            # WHICH agent is this source's own configured `research_domain`
            # (default FINANCIAL_NEWS) -- see `_RESEARCH_DOMAIN_TO_AGENT`.
            agents.add(self._secondary_domain_agent[article.source_name])

        if article.source_type in _SECONDARY_SOURCE_TYPES and self._startup_matcher:
            # Startup has no fixed source registry to match on (see module
            # docstring), so it needs its own strong, specific signal too —
            # the same funding/acquisition/category vocabulary
            # StartupAgent's own pre-filter already uses — rather than
            # unconditionally seeing every secondary article.
            if self._startup_matcher.search(_haystack(article)):
                agents.add(AgentName.STARTUP)

        return agents

    def home_agent(self, article: Article) -> Optional[AgentName]:
        """The one agent that is this article's own configured identity --
        exact source-name membership (case 1) or the source's configured
        `research_domain` (case 3) -- as opposed to an incidental
        content-signal match (case 2/4), which says a name merely appears
        somewhere in the text. `None` when the article carries no
        configured identity at all (e.g. a non-member source routed purely
        by content signal).

        Real bug this exists to let `pipeline.nodes._build_current_candidate_pool`
        fix (confirmed 2026-10-19): PR Newswire's GMIF2026 memory/storage
        conference article quoted "Daniel Yen of Morgan Stanley" once in
        passing, which matched the PEER content-signal (case 2) purely
        because Morgan Stanley is a configured peer bank. `ROUTABLE_AGENTS`
        fixed order then picked PEER as `primary_agent` (PEER sorts before
        STARTUP), so the article was judged ONLY against "is this about a
        peer bank?" (correctly: no) and never reached STARTUP -- PR
        Newswire's own configured `research_domain`, and the actually
        relevant lens. `home_agent` lets the candidate-pool builder prefer
        a multi-routed article's own configured identity over a same-text
        mention of an unrelated configured name.
        """
        if article.source_name in self._peer_names:
            return AgentName.PEER
        if article.source_name in self._ai_provider_names:
            return AgentName.AI_PRODUCT
        if article.source_name in self._analyst_names:
            return AgentName.ANALYST
        return self._secondary_domain_agent.get(article.source_name)


@dataclass
class RoutingResult:
    by_agent: Dict[AgentName, List[Article]] = field(default_factory=lambda: {name: [] for name in ROUTABLE_AGENTS})
    unrouted_count: int = 0
    multi_routed_count: int = 0
    # article.id -> its home-domain agent (see `ArticleRouter.home_agent`),
    # only present when the article has one. Lets a multi-routed article's
    # OWN configured identity be preferred over an incidental content-signal
    # match when a single `primary_agent` must be picked (see
    # `pipeline.nodes._build_current_candidate_pool`).
    home_agent_by_article_id: Dict[str, AgentName] = field(default_factory=dict)


def route_articles(articles: List[Article], config: AppConfig) -> RoutingResult:
    """Pure, deterministic, LLM-free: every article's agent membership is
    decided by config-driven name/signal matching only."""
    router = ArticleRouter(config)
    result = RoutingResult()
    for article in articles:
        agents = router.route(article)
        if not agents:
            result.unrouted_count += 1
        if len(agents) > 1:
            result.multi_routed_count += 1
        for agent_name in agents:
            result.by_agent[agent_name].append(article)
        home = router.home_agent(article)
        if home is not None:
            result.home_agent_by_article_id[article.id] = home
    return result
