"""Thin orchestration nodes — each one wraps an already-existing,
already-tested service. No retrieval, normalization, scoring, or clustering
logic is reimplemented here; `PipelineServices` is plain dependency
injection (not global state) so tests can supply fakes for every service.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from research_assistant.agents.base import ResearchAgent
from research_assistant.agents.link_grounded import build_link_grounded_research_item
from research_assistant.agents.report import ReportAgent, build_link_grounded_report
from research_assistant.agents.router import ROUTABLE_AGENTS, route_articles
from research_assistant.agents.triage import TriageCandidate, TriageItemResult, run_triage_batch
from research_assistant.config.schema import AppConfig
from research_assistant.events.clock import Clock, SystemClock
from research_assistant.events.service import EventService
from research_assistant.evidence.conversion import evidence_rejection_reason
from research_assistant.llm.errors import LLMUnavailableError
from research_assistant.models.article import DatePrecision
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, BacklogStatus, GroundingType
from research_assistant.models.report import Report, ReportFailure
from research_assistant.models.research_item import ResearchItem
from research_assistant.pipeline.state import BacklogUpdate, ExtractedBundle, PipelineDiagnostic, PipelineState
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.classification import classify_many
from research_assistant.retrieval.exceptions import SourceUnreachableError
from research_assistant.retrieval.models import CachePolicy, ExtractionStatus

logger = logging.getLogger(__name__)

# A title containing a pipe reads as "{Site} | {Section}" (a breadcrumb/nav
# pattern observed on real OpenAI newsroom category pages, e.g. "OpenAI
# Newsroom | Research") rather than a specific headline.
_HUB_TITLE_MARKERS = ("|",)

# Below this, an extracted body reads as a teaser/stub rather than full
# article content -- the real NVIDIA "News Archive" hub page from Phase 11
# extracted to only 349 characters (just the top headline's own teaser).
MIN_SUBSTANTIVE_BODY_CHARS = 400

# Forensic finding (CURRENT run `31fa192e4d7a7175`, 2026-10-12): a real
# Financial Times candidate (`7204dd86f225bad3`, "The huge lurking leverage
# of AI hyperscalers") extracted to nothing but subscription-paywall
# marketing copy ("Subscribe to unlock this article", "Save 50% on Standard
# Digital", "Complete digital access with exclusive insights...") -- no
# actual article prose was ever retrieved. This easily cleared
# `MIN_SUBSTANTIVE_BODY_CHARS` (paywall copy is long) and had no hub-title
# marker, so it consumed a full triage slot and a full deep-analysis call
# (one of only 4 that run) for content that was never the real article --
# Qwen's own "irrelevant" verdict on it was a CORRECT response to this
# specific, genuinely content-free input, not a relevance-logic defect.
# Deliberately generic phrase set (observed across major subscription-
# gated news publishers' own templated paywall copy, not one specific
# domain) and requires at least two independent matches -- a single
# incidental mention (e.g. a "Subscribe to our newsletter" footer link on
# an otherwise real, substantive article) must never alone reject real
# content; this only fires when the page is, structurally, predominantly
# paywall marketing.
_PAYWALL_BOILERPLATE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"subscribe to unlock this article",
        r"save\s+\d+%\s+on\s+(?:standard|premium|digital)",
        r"complete digital access",
        r"explore (?:our|more) (?:full range of )?subscriptions",
        r"digital access to trusted .{0,40} journalism",
        r"start your free trial",
        r"already have (?:free )?access via your university or organi[sz]ation",
        r"for individuals.{0,80}for multiple readers",
    )
)
_PAYWALL_BOILERPLATE_MIN_MATCHES = 2


def _is_paywall_boilerplate(body: str) -> bool:
    return sum(1 for pattern in _PAYWALL_BOILERPLATE_PATTERNS if pattern.search(body)) >= _PAYWALL_BOILERPLATE_MIN_MATCHES


def _flatten_topic_keywords(topics) -> List[str]:
    """Same AI + financial-services vocabulary already defined in
    `config/topics.yaml`, reused here (not duplicated or invented) purely to
    rank already-routed candidates before `top_k` truncates them -- see
    `_candidate_rank_key`."""
    buckets = [
        topics.ai_topics.core, topics.ai_topics.models, topics.ai_topics.infrastructure,
        topics.ai_topics.enterprise, topics.ai_topics.governance,
        topics.financial_services.core, topics.financial_services.markets,
        topics.financial_services.wealth, topics.financial_services.risk,
        topics.financial_services.operations,
    ]
    return [keyword.lower() for bucket in buckets for keyword in bucket]


def _flatten_event_type_phrases(topics) -> List[str]:
    """`config/topics.yaml`'s `event_types` (e.g. "product_launch",
    "partnership", "funding") as natural-language phrases -- the existing,
    already-configured taxonomy of what a genuine dated development looks
    like, reused (not duplicated) as a ranking signal distinguishing a
    specific development from a generic topic/capability overview page
    (Phase 13E)."""
    return [event_type.replace("_", " ").lower() for event_type in topics.event_types]


# A short, generic title (at most this many words) with no action/event
# signal anywhere in the evidence reads as a topic/capability landing page
# (real examples: BCG's "AI Agents", "Generative AI", "Responsible AI" --
# each a standing practice-area overview, not a specific piece of research)
# rather than a specific article -- demoted in ranking, never hard-rejected
# (Section 13: do not destroy recall; if nothing stronger exists in the
# candidate pool, it can still be selected).
_GENERIC_TITLE_MAX_WORDS = 3


def _candidate_rank_key(article, evidence, keywords: List[str], event_type_phrases: List[str] = ()):
    """Composite, deterministic, no-LLM ranking so `top_k` selects the
    strongest few candidates rather than just the first/most-recent few
    pages discovered (Phase 12, strengthened Phase 13E). Higher sorts first
    (used with `reverse=True`):

      1. hub/listing demotion -- a short extracted body or a "Site |
         Section" style title reads as an index/category page rather than
         a specific article (real discovered examples: NVIDIA's "News
         Archive", OpenAI's "OpenAI Newsroom | Research"), and is ranked
         below every genuine candidate.
      2. action/event-type language -- the evidence text itself describes a
         specific action (the same announcement-language patterns the
         date-grounding guardrail uses, e.g. "today announced", "has
         launched") or names one of the configured `event_types`
         (partnership, funding, model release, ...). This is what
         distinguishes a genuine development from an incidental AI mention.
      3. NOT a short, generic topic/capability-overview title with no
         action signal (e.g. "AI Agents", "Generative AI") -- demoted
         below genuine headlines, never hard-excluded.
      4. topical keyword hits in title+body -- more AI/financial-services
         vocabulary present is a stronger relevance signal.
      5. recency -- the original sort key, now a tiebreaker rather than the
         sole criterion.
    """
    from research_assistant.agents.base import _contains_dated_announcement_language

    title = article.title or ""
    body = evidence.content or ""
    is_substantive = len(body) >= MIN_SUBSTANTIVE_BODY_CHARS and not any(
        marker in title for marker in _HUB_TITLE_MARKERS
    )
    haystack = f"{title}\n{body}".lower()
    has_action_signal = _contains_dated_announcement_language(body) or any(
        phrase in haystack for phrase in event_type_phrases
    )
    looks_like_generic_topic_page = len(title.split()) <= _GENERIC_TITLE_MAX_WORDS and not has_action_signal
    keyword_hits = sum(1 for keyword in keywords if keyword in haystack)
    recency = article.publication_date or datetime.min.replace(tzinfo=timezone.utc)
    return (is_substantive, has_action_signal, not looks_like_generic_topic_page, keyword_hits, recency)


# CURRENT-only ranking signal (Part B2): a bounded [now-24h, now) window is
# meaningless for a candidate whose publication date isn't actually known
# to be exact -- MONTH precision and missing dates are already excluded
# from ever qualifying as a bounded event by the existing, unmodified
# EventBuilder/ResearchItem date guard (see agents.base._build_research_item);
# this just makes sure the scarce CURRENT LLM budget is spent on exact-dated
# candidates FIRST, rather than discovering only after an LLM call that a
# candidate could never have qualified anyway. EXACT and DATE_ONLY are
# treated as equally trustworthy (both are real, source-confirmed dates --
# DATE_ONLY just lacks a time-of-day); MONTH ranks below both; a missing
# date ranks last.
_CURRENT_DATE_PRECISION_RANK: Dict[DatePrecision, int] = {
    DatePrecision.EXACT: 2,
    DatePrecision.DATE_ONLY: 2,
    DatePrecision.MONTH: 1,
    DatePrecision.UNKNOWN: 0,
}


def _current_candidate_rank_key(article, evidence, keywords: List[str], event_type_phrases):
    """Same composite ranking as `_candidate_rank_key`, with CURRENT's own
    exact-date-first priority as the leading sort key -- used only for
    `DigestType.CURRENT` runs, never for Morning/Evening (Part B1: "Only
    CURRENT gets a dedicated candidate-selection policy")."""
    precision_rank = _CURRENT_DATE_PRECISION_RANK.get(article.publication_date_precision, 0)
    return (precision_rank, *_candidate_rank_key(article, evidence, keywords, event_type_phrases))


# Only EXACT/DATE_ONLY are trustworthy enough to be checked against CURRENT's
# exact [period_start, period_end) rolling-24h window -- a MONTH-precision
# date (stored as day-1-of-month purely as a storage placeholder, see
# evidence.normalizer) cannot be verified to fall inside or outside a
# window this narrow, and UNKNOWN means no date was ever extracted at all.
_CURRENT_TRUSTWORTHY_PRECISIONS = (DatePrecision.EXACT, DatePrecision.DATE_ONLY)


def _date_gate_diagnostic(article, reason: str) -> "PipelineDiagnostic":
    """One line per article dropped by `_apply_current_date_gate` -- same
    `stage="agent_analysis"`/severity="warning" convention and
    `candidate=<id> source=<repr> title=<repr> url=<url>` shape the
    existing `triage:irrelevant`/`deep_analysis:irrelevant` diagnostics
    already use (forensic finding, CURRENT run `e818e08adc2e4dd1`: 194 of
    201 period-eligible articles were dropped here with no persisted
    per-article record of which one or why, only the aggregate
    `current_dropped_undated` count)."""
    title = (article.title or "")[:80]
    return PipelineDiagnostic(
        stage="agent_analysis",
        message=(
            f"date_gate:{reason} candidate={article.id} source={article.source_name!r} "
            f"title={title!r} url={article.url} "
            f"publication_date={article.publication_date.isoformat() if article.publication_date else 'None'} "
            f"precision={article.publication_date_precision.value}"
        ),
        severity="warning",
    )


def _apply_current_date_gate(
    articles: list, period_start, period_end, *, diagnostics: Optional[List[PipelineDiagnostic]] = None,
) -> Tuple[list, Dict[str, int]]:
    """CURRENT's hard upstream date gate (never applied to Morning/Evening --
    see this function's only call site in `agent_analysis()`, gated on
    `state.digest_type == DigestType.CURRENT`).

    Confirmed real-production defect this closes: a 2026-10-07 live CURRENT
    scan (run `2a55bf02ff1b7d1f`) counted 300 articles as "inside the
    period" purely because `_filter_by_period` retains any article whose
    `publication_date` is unknown -- 279 of those 300 were in fact undated,
    yet all 279 still flowed through evidence creation, domain routing, and
    competed for the bounded triage budget, before being dropped only at
    the very last step (event formation, `publication_date is None`
    exclusion). This function moves that same, pre-existing drop rule much
    earlier: an article without a trustworthy (EXACT/DATE_ONLY) publication
    date already inside `[period_start, period_end)` is removed HERE, before
    `route_articles()` is ever called -- so it can never reach routing,
    triage, specialist analysis, ResearchItem creation, or Event creation.
    `is_dated_development`/`_resolve_dated_development`/event-date
    resolution (`events.builder`) are completely unchanged and remain the
    separate, final authority on whether an article's CONTENT describes a
    genuinely new dated action -- this gate only ever checks the article's
    own extracted metadata date, never its content, and never invents or
    rescues a date via the LLM or the scan's current time.

    Returns `(kept_articles, counts)` where `counts` has keys "undated",
    "stale", "future" -- real, observable reasons, never guessed -- for
    diagnostics (never fabricated: every count comes from an article that
    was actually dropped, matching the reason given).

    `diagnostics`, when supplied, additionally gets one per-article
    `date_gate:{undated,stale,future}` entry appended for every dropped
    article (optional and purely additive -- every pre-existing call site
    that omits it keeps working unchanged, and the kept/counts
    classification itself is never affected by whether it is supplied)."""
    kept: list = []
    undated: list = []
    stale: list = []
    future: list = []
    for article in articles:
        if (
            article.publication_date is None
            or article.publication_date_precision not in _CURRENT_TRUSTWORTHY_PRECISIONS
        ):
            undated.append(article)
            if diagnostics is not None:
                diagnostics.append(_date_gate_diagnostic(article, "undated"))
        elif article.publication_date < period_start:
            stale.append(article)
            if diagnostics is not None:
                diagnostics.append(_date_gate_diagnostic(article, "stale"))
        elif article.publication_date >= period_end:
            future.append(article)
            if diagnostics is not None:
                diagnostics.append(_date_gate_diagnostic(article, "future"))
        else:
            kept.append(article)
    return kept, {"undated": len(undated), "stale": len(stale), "future": len(future)}


def _select_with_source_diversity(ranked_candidates: list, top_k: int) -> list:
    """CURRENT-only selection (Part B4): `ranked_candidates` is already
    sorted strongest-first. Plain `candidates[:top_k]` can let one prolific
    source (e.g. a single peer bank's newsroom) consume the entire budget
    even when other sources have perfectly good current candidates of their
    own waiting just below the cutoff. This takes each source's best
    remaining candidate in turn, in the order that candidate would rank
    overall, before any source gets a second pick -- so rank still decides
    who goes first, but no single source can exhaust the budget while
    others go completely unrepresented. With only one source in the pool,
    this is identical to `ranked_candidates[:top_k]`."""
    buckets: Dict[str, list] = {}
    source_order: List[str] = []
    for pair in ranked_candidates:
        name = pair[0].source_name
        if name not in buckets:
            buckets[name] = []
            source_order.append(name)
        buckets[name].append(pair)

    selected: list = []
    round_index = 0
    while len(selected) < top_k:
        progressed = False
        for name in source_order:
            if round_index < len(buckets[name]):
                selected.append(buckets[name][round_index])
                progressed = True
                if len(selected) >= top_k:
                    break
        if not progressed:
            break
        round_index += 1
    return selected


# ---------------------------------------------------------------------------
# CURRENT's bounded LLM funnel (RETRIEVAL -> deterministic gate -> batched
# triage -> bounded deep analysis -> existing EventBuilder/Report/Digest).
# Replaces, for CURRENT only, the previous design where every specialist
# agent independently ran a full LLM deep-analysis call on every one of its
# routed candidates -- unbounded in the number of discovered articles and
# the real cause of multi-hour CURRENT runs on a CPU-only machine. Never
# applied to Morning/Evening, which keep their existing per-agent
# top_k/current_top_k loop below completely unchanged.
# ---------------------------------------------------------------------------


def _passes_current_deterministic_gate(article, evidence) -> bool:
    """Structural-only filter (Part A): removes candidates no triage/deep
    analysis could usefully reason about -- an empty/missing title, a
    hub/index/navigation page (a body this thin reads as a teaser/stub,
    not a full article -- the real NVIDIA "News Archive" hub page
    extracted to only 349 characters), or a page that extracted to
    nothing but subscription-paywall marketing copy instead of real
    article content (see `_is_paywall_boilerplate`). Deliberately NEVER a
    keyword-presence/topic check -- that would let the gate itself decide
    relevance instead of triage, which the task explicitly forbids; this
    only ever removes content that was never actually retrieved in the
    first place.

    A "Site | Section"-style title is NOT used here as a hard veto, only
    as a SOFT demotion signal in `_candidate_rank_key` -- confirmed
    regression (2026-10-14): a pipe is the near-universal "Headline |
    Site Name" pattern most news sites' <title> tags carry, so hard-
    excluding on it alone silently dropped genuine, substantive articles
    (a real 6,127-char NatWest press release) before they ever reached
    triage, with no diagnostic trail. Body length alone is what
    distinguishes a real hub shell from a real article."""
    title = (article.title or "").strip()
    if not title:
        return False
    body = evidence.content or ""
    is_substantive = len(body) >= MIN_SUBSTANTIVE_BODY_CHARS
    return is_substantive and not _is_paywall_boilerplate(body)


def _build_current_candidate_pool(
    routing, evidence_by_article_id: Dict[str, Any], enabled_agent_names,
) -> List[TriageCandidate]:
    """One (article, evidence) pair per DISTINCT article, deduplicated
    across every agent it was routed to -- the same article independently
    reaching multiple specialist agents (a real, logged case -- see
    `routing.multi_routed_count`) must cost at most one triage slot and at
    most one deep-analysis call, never one per agent it matched.

    Picking that single `primary_agent`: `routing.home_agent_by_article_id`
    (the article's OWN configured identity -- see `ArticleRouter.home_agent`)
    wins whenever it is one of the agents actually matched, regardless of
    `ROUTABLE_AGENTS` order. Real bug this fixes (confirmed 2026-10-19):
    PR Newswire's GMIF2026 memory/storage conference article quoted
    "Daniel Yen of Morgan Stanley" once in passing, which matched PEER's
    content signal purely because Morgan Stanley is a configured peer
    bank -- the OLD fixed-order "first match wins" (PEER sorts before
    STARTUP in `ROUTABLE_AGENTS`) picked PEER as `primary_agent`, so the
    article was judged only against "is this about a peer bank?"
    (correctly: no) and never reached STARTUP, PR Newswire's own
    configured `research_domain` and the actually relevant lens. Only when
    no home agent is matched (a pure content-signal-only route, or the
    home agent isn't enabled this run) does `ROUTABLE_AGENTS`'s fixed order
    still decide -- same fallback as before, including the documented
    FINANCIAL_NEWS-before-STARTUP precedence for that case. Only agents
    actually present in `services.agents` are ever picked as
    `primary_agent`, matching the original per-agent loop's
    `for agent_name, agent in services.agents.items()` -- an article routed
    only to an agent that isn't configured this run is skipped, exactly as
    it always was."""
    matched_agents_by_article_id: Dict[str, Set[AgentName]] = {}
    for agent_name in ROUTABLE_AGENTS:
        if agent_name not in enabled_agent_names:
            continue
        for article in routing.by_agent.get(agent_name, []):
            matched_agents_by_article_id.setdefault(article.id, set()).add(agent_name)

    seen: Dict[str, TriageCandidate] = {}
    for agent_name in ROUTABLE_AGENTS:
        if agent_name not in enabled_agent_names:
            continue
        for article in routing.by_agent.get(agent_name, []):
            if article.id in seen:
                continue
            evidence = evidence_by_article_id.get(article.id)
            if evidence is None:
                continue  # no retrieved content -- handled entirely by the separate, unchanged LINK-grounded path
            matched_agents = matched_agents_by_article_id[article.id]
            home_agent = routing.home_agent_by_article_id.get(article.id)
            primary_agent = home_agent if home_agent in matched_agents else agent_name
            seen[article.id] = TriageCandidate(
                candidate_id=article.id, article=article, evidence=evidence, primary_agent=primary_agent,
            )
    return list(seen.values())


def _apply_source_diversity_to_pool(pool: List[TriageCandidate], top_k: int) -> List[TriageCandidate]:
    """CURRENT is an intelligence-UNIVERSE scan, not a plain top-N article
    ranking -- reuses the EXISTING `_select_with_source_diversity` exactly
    (no new diversity algorithm) so one prolific source cannot consume the
    entire bounded triage pool while other sources' perfectly good
    candidates go completely unrepresented, same as it always has for
    CURRENT's per-agent top_k before the bounded funnel replaced it.
    `pool` must already be gated and ranked strongest-first; `top_k` is the
    bounded triage-pool size (`triage_batch_size * max_triage_batches`)."""
    pairs = [(c.article, c.evidence) for c in pool]
    selected_pairs = _select_with_source_diversity(pairs, top_k)
    by_candidate_id = {c.candidate_id: c for c in pool}
    return [by_candidate_id[article.id] for article, _ in selected_pairs]


# Specialist-routing refactor: per-domain context line handed to
# `run_triage_batch`'s `domain_hint`, so a triage call -- even though it is
# still a cheap, single Qwen call -- reasons about ONE named, specialist
# domain rather than a domain-blind mixed pool of banks/AI/startups/
# analyst-research all at once (the "one general-purpose LLM stage
# reasoning over a mixed pool" failure mode this refactor removes). Text
# intentionally mirrors each specialist agent's own scope (see
# agents.peer/ai_product/startup/analyst/financial_news docstrings) without
# duplicating their full system prompts.
_DOMAIN_TRIAGE_HINTS: Dict[AgentName, str] = {
    AgentName.PEER: (
        "Financial institutions/banks -- AI adoption or strategy, technology "
        "transformation, AI products, partnerships, major investments, "
        "acquisitions, capital markets or digital banking technology."
    ),
    AgentName.AI_PRODUCT: (
        "AI/technology developments -- generative or enterprise AI, AI "
        "models/infrastructure/products, AI partnerships or investments, AI "
        "regulation with material business impact, fintech technology, "
        "enterprise technology strategy, major technology launches."
    ),
    AgentName.ANALYST: (
        "Consulting/analyst research -- AI strategy, financial services, "
        "banking, capital markets, enterprise AI, technology transformation, "
        "or operating-model insights."
    ),
    AgentName.STARTUP: (
        "Startup/funding developments -- venture funding rounds, IPOs, "
        "acquisitions, strategic investments, startup product launches, "
        "major partnerships involving startups."
    ),
    AgentName.FINANCIAL_NEWS: (
        "General financial/technology news -- substantive AI, fintech, or "
        "financial-services developments reported by an independent outlet."
    ),
}


def _allocate_domain_budget(domain_groups: Dict[AgentName, list], total_budget: int) -> Dict[AgentName, int]:
    """Round-robin, one slot per domain per round (in `ROUTABLE_AGENTS`
    order), until `total_budget` slots are handed out or every domain's own
    candidates are exhausted -- the exact same fairness guarantee
    `_select_with_source_diversity` already gives across SOURCES within one
    domain, applied one level up, across DOMAINS, so one specialist's
    candidates can never consume the entire bounded triage pool while
    another specialist's perfectly good candidates go completely
    unrepresented. With only one domain present (every pre-refactor test,
    and any real run routing to a single agent), this allocates the whole
    budget to that one domain -- identical to the pre-refactor behavior."""
    order = [name for name in ROUTABLE_AGENTS if name in domain_groups]
    caps = {name: 0 for name in order}
    allocated = 0
    while allocated < total_budget:
        progressed = False
        for name in order:
            if allocated >= total_budget:
                break
            if caps[name] < len(domain_groups[name]):
                caps[name] += 1
                allocated += 1
                progressed = True
        if not progressed:
            break
    return caps


def _run_current_triage_by_domain(
    gated_pool: List[TriageCandidate], *, triage_client, batch_size: int, triage_pool_bound: int,
    diagnostics: List[PipelineDiagnostic],
) -> Tuple[List[Tuple[TriageCandidate, TriageItemResult]], Dict[str, int]]:
    """Replaces a single `_apply_source_diversity_to_pool` + `_run_current_triage`
    call over the WHOLE (potentially multi-domain) `gated_pool` with:
    domain-fair budget allocation (`_allocate_domain_budget`) -> per-domain
    source-diversity selection (the existing, unchanged
    `_apply_source_diversity_to_pool`) -> a SEPARATE, domain-pure triage
    batch run per domain (the existing, unchanged `_run_current_triage`,
    with a domain-specific hint) -- so no single LLM call ever mixes
    candidates from more than one specialist domain. `gated_pool` is
    already gated and ranked strongest-first; that order is preserved
    within each domain group and restored across domains afterward (the
    merged winners are re-ranked before being returned), so the global
    `triage_pool_bound`/budget semantics are unchanged -- only HOW the
    bounded pool is split into LLM calls changes.

    With only one domain present, this reduces to exactly the pre-refactor
    call sequence (one domain's full share of `triage_pool_bound`, one
    diversity pass, one triage run) -- confirmed by every pre-existing
    single-domain CURRENT funnel test."""
    domain_groups: Dict[AgentName, List[TriageCandidate]] = {}
    for candidate in gated_pool:
        domain_groups.setdefault(candidate.primary_agent, []).append(candidate)

    domain_caps = _allocate_domain_budget(domain_groups, triage_pool_bound)

    winners: List[Tuple[TriageCandidate, TriageItemResult]] = []
    totals = {
        "triage_batches_attempted": 0, "triage_candidates": 0, "triage_relevant": 0,
        "triage_irrelevant": 0, "triage_failures": 0, "candidates_sent_to_triage": 0,
    }
    offered_ids: Set[str] = set()
    for agent_name in ROUTABLE_AGENTS:
        group = domain_groups.get(agent_name)
        cap = domain_caps.get(agent_name, 0)
        if not group or cap <= 0:
            continue
        domain_pool = _apply_source_diversity_to_pool(group, cap)
        offered_ids.update(c.candidate_id for c in domain_pool)
        # `max_batches` sized to exactly cover `domain_pool` -- it was
        # already bounded to `cap` above, so no second truncation happens
        # here; this only controls how many `batch_size`-sized calls this
        # one domain's own pool is split into.
        domain_max_batches = -(-len(domain_pool) // batch_size) if domain_pool else 0
        domain_winners, domain_metrics = _run_current_triage(
            domain_pool, triage_client=triage_client, batch_size=batch_size, max_batches=domain_max_batches,
            diagnostics=diagnostics, domain_hint=_DOMAIN_TRIAGE_HINTS.get(agent_name),
        )
        winners.extend(domain_winners)
        for key in totals:
            totals[key] += domain_metrics[key]

    # Global truncation (candidates never offered a triage slot at all) is
    # computed once, at the pool level -- every per-domain call above saw
    # an already-capped group, so each domain_metrics' own
    # "candidates_truncated_before_triage" is always 0.
    totals["candidates_truncated_before_triage"] = max(0, len(gated_pool) - totals["candidates_sent_to_triage"])

    # Per-candidate observability (forensic finding, CURRENT run
    # `e818e08adc2e4dd1`): previously only the aggregate
    # "candidates_truncated_before_triage" count existed -- never which
    # specific candidate(s) never got a triage slot at all, purely because
    # the domain-fair/pool-bound budget allocation above capped them out
    # before any LLM call. Never confused with `triage:irrelevant` (that
    # requires an actual triage decision): these candidates by definition
    # never appear in `offered_ids`.
    if diagnostics is not None:
        for candidate in gated_pool:
            if candidate.candidate_id in offered_ids:
                continue
            title = (candidate.article.title or "")[:80]
            diagnostics.append(
                PipelineDiagnostic(
                    stage="agent_analysis",
                    message=(
                        f"budget_skip:triage candidate={candidate.candidate_id} "
                        f"agent={candidate.primary_agent.value} source={candidate.article.source_name!r} "
                        f"title={title!r} url={candidate.article.url} reason=triage_pool_bound"
                    ),
                    severity="warning",
                )
            )
    return winners, totals


def _run_current_triage(
    pool: List[TriageCandidate], *, triage_client, batch_size: int, max_batches: int,
    diagnostics: List[PipelineDiagnostic], domain_hint: Optional[str] = None,
) -> Tuple[List[Tuple[TriageCandidate, TriageItemResult]], Dict[str, int]]:
    """`pool` is already gated and ranked strongest-first. Bounds the
    candidates actually sent to the LLM to `batch_size * max_batches`
    (default 3*8=24) regardless of how many more exist -- the rest are
    reported as unprocessed, never silently analyzed anyway via some other
    path. A batch that fails outright (timeout/connection/schema error)
    costs that batch's candidates and moves on to the next batch -- no
    retry, consistent with Part G's "a timeout consumes that candidate's
    bounded budget and moves forward"."""
    bounded_pool = pool[: batch_size * max_batches]
    winners: List[Tuple[TriageCandidate, TriageItemResult]] = []
    batches_attempted = 0
    candidates_triaged = 0
    triage_relevant = 0
    triage_irrelevant = 0
    triage_failures = 0

    for start in range(0, len(bounded_pool), batch_size):
        batch = bounded_pool[start : start + batch_size]
        batches_attempted += 1
        try:
            results = run_triage_batch(triage_client, batch, domain_hint=domain_hint)
        except Exception as exc:
            triage_failures += len(batch)
            diagnostics.append(
                PipelineDiagnostic(
                    stage="agent_analysis",
                    message=f"triage batch {batches_attempted} ({len(batch)} candidate(s)) failed: {exc}",
                    severity="warning",
                )
            )
            logger.warning("agent_analysis(CURRENT): triage batch %d failed: %s", batches_attempted, exc)
            continue
        candidates_triaged += len(batch)
        for candidate in batch:
            result = results.get(candidate.candidate_id)
            title = (candidate.article.title or "")[:80]
            if result is None or not result.relevant:
                triage_irrelevant += 1
                # Lightweight per-candidate observability (forensic finding,
                # run 8c605b0881e8edf0): previously NOTHING recorded which
                # candidate triage rejected or why -- only aggregate counts.
                # One line per rejected candidate, reusing the existing
                # diagnostics_json mechanism (no schema/table change).
                diagnostics.append(
                    PipelineDiagnostic(
                        stage="agent_analysis",
                        message=(
                            f"triage:irrelevant candidate={candidate.candidate_id} "
                            f"agent={candidate.primary_agent.value} source={candidate.article.source_name!r} "
                            f"title={title!r} url={candidate.article.url} "
                            f"confidence={result.confidence if result else 'n/a'}"
                        ),
                        severity="warning",
                    )
                )
                continue
            triage_relevant += 1
            winners.append((candidate, result))
            diagnostics.append(
                PipelineDiagnostic(
                    stage="agent_analysis",
                    message=(
                        f"triage:relevant candidate={candidate.candidate_id} "
                        f"agent={candidate.primary_agent.value} source={candidate.article.source_name!r} "
                        f"title={title!r} url={candidate.article.url} confidence={result.confidence}"
                    ),
                    severity="warning",
                )
            )

    metrics = {
        "triage_batches_attempted": batches_attempted,
        "triage_candidates": candidates_triaged,
        "triage_relevant": triage_relevant,
        "triage_irrelevant": triage_irrelevant,
        "triage_failures": triage_failures,
        "candidates_sent_to_triage": len(bounded_pool),
        "candidates_truncated_before_triage": max(0, len(pool) - len(bounded_pool)),
    }
    return winners, metrics


def _select_deep_analysis_winners(
    triage_winners: List[Tuple[TriageCandidate, TriageItemResult]], budget: int,
    *, diagnostics: Optional[List[PipelineDiagnostic]] = None,
) -> Tuple[List[Tuple[TriageCandidate, TriageItemResult]], int]:
    """Domain-fair selection of which triage winners spend the scarce,
    CPU-bound deep-analysis budget. With `max_deep_analysis` deliberately
    small on CPU-only qwen2.5:3b (Part 3), a plain global-rank cutoff could
    let one specialist domain that happens to rank strongest consume the
    ENTIRE budget, leaving every other domain's perfectly good triage
    winners completely unanalyzed. Reuses the exact same round-robin
    fairness `_allocate_domain_budget` already gives triage's bounded pool,
    applied one stage later. `triage_winners` is already strongest-first;
    that order is preserved within each domain group, and the returned
    selection is re-sorted strongest-first overall by the caller. Returns
    `(selected, dropped_count)` -- `dropped_count` is how many triage
    winners never got a deep-analysis slot purely due to this selection
    (never silently lost from `current_funnel_metrics`).

    `diagnostics`, when supplied, additionally gets one per-candidate
    `budget_skip:deep_analysis_selection` entry for every triage winner
    dropped here -- optional and purely additive, never changing which
    candidates are selected."""
    if len(triage_winners) <= budget:
        return triage_winners, 0
    domain_groups: Dict[AgentName, List[Tuple[TriageCandidate, TriageItemResult]]] = {}
    for pair in triage_winners:
        domain_groups.setdefault(pair[0].primary_agent, []).append(pair)
    # Within each domain, rank by the triage model's own confidence
    # (descending, stable) before taking that domain's cap -- confirmed
    # real-run defect (CURRENT run beeb562aba597399): the pre-fix code
    # sliced `domain_groups[name][:cap]` in pre-triage rank order, so a
    # weaker candidate (confidence=0.85, "Faraday Future") won a "startup"
    # deep-analysis slot ahead of a stronger one from the SAME domain
    # (confidence=0.95/1.0, the real dated SWI Group capital-increase
    # announcement) purely because of discovery/pre-triage ordering, never
    # because triage itself judged it weaker. Domain-fairness (which domain
    # gets how many slots) is unchanged; only the ranking WITHIN a domain's
    # own allocated slots now reflects triage confidence.
    for group in domain_groups.values():
        group.sort(key=lambda pair: (pair[1].confidence if pair[1] is not None else 0.0), reverse=True)
    domain_caps = _allocate_domain_budget(domain_groups, budget)
    selected: List[Tuple[TriageCandidate, TriageItemResult]] = []
    for name, cap in domain_caps.items():
        selected.extend(domain_groups[name][:cap])
    if diagnostics is not None:
        selected_ids = {pair[0].candidate_id for pair in selected}
        for candidate, _result in triage_winners:
            if candidate.candidate_id in selected_ids:
                continue
            title = (candidate.article.title or "")[:80]
            diagnostics.append(
                PipelineDiagnostic(
                    stage="agent_analysis",
                    message=(
                        f"budget_skip:deep_analysis_selection candidate={candidate.candidate_id} "
                        f"agent={candidate.primary_agent.value} source={candidate.article.source_name!r} "
                        f"title={title!r} url={candidate.article.url} reason=domain_fair_selection"
                    ),
                    severity="warning",
                )
            )
    return selected, len(triage_winners) - len(selected)


# ---------------------------------------------------------------------------
# CURRENT candidate backlog: a candidate that passes every deterministic
# gate and has not been conclusively rejected, but whose run ran out of
# triage/deep-analysis budget before reaching it, is never discarded -- see
# `storage.repositories.backlog`. These helpers only ever READ via
# `services.persistence_service` (never write mid-run -- the actual write
# happens once, atomically, alongside everything else this run produces,
# via `make_persistence_node`), so they stay fully unit-testable without a
# real database and never break the existing "persist once, at the end"
# architecture.
# ---------------------------------------------------------------------------

_BACKLOG_AGING_CAP = 5
"""Bounded aging boost ("prioritise, do not discard... allow waiting
candidates to progress through bounded aging/fairness rules"): each prior
deferral OR failed attempt raises a backlog candidate's effective rank by
one tier, capped here so an arbitrarily old candidate cannot win every
remaining slot outright forever -- once the cap is reached it competes on
ordinary strength ranking, it is simply never ranked below a brand-new
candidate with equivalent strength signals.

Reliability patch (pre-live-scan audit finding): the boost must be fed by
`defer_count + attempt_count`, never `attempt_count` alone -- a candidate
that is ALWAYS deferred purely by a budget limit (never actually reaching
a deep-analysis attempt) previously never accumulated any `attempt_count`
at all, so it never gained priority across repeated runs. `defer_count`
(see `pipeline.state.BacklogUpdate.defer_count`) closes that gap by
counting the lost-opportunity-to-budget case specifically."""


def _aging_boosted_rank_key(article, evidence, keywords, event_type_phrases, backlog_fairness_counts: Dict[str, int]):
    boost = min(backlog_fairness_counts.get(article.id, 0), _BACKLOG_AGING_CAP)
    return (boost, *_current_candidate_rank_key(article, evidence, keywords, event_type_phrases))


def _resumable_backlog_candidates(services, state: PipelineState, *, diagnostics: List[PipelineDiagnostic]):
    """Reads (never writes) this run's bounded share of the durable
    backlog (`current_llm.backlog_claim_limit`), revalidates each candidate
    against THIS run's own CURRENT period/deterministic gate ("revalidated
    ... before it is included" -- a candidate deferred under an older
    period is never smuggled into a newer one merely because it remained
    pending), and splits survivors into:

      - never-triaged candidates: merge into the normal gated pool, go
        through triage exactly like a fresh candidate.
      - already-triaged candidates: a valid persisted `triage_relevant=
        True` result -- bypass triage entirely ("avoid rerunning triage
        for candidates with a valid persisted triage result").

    Reliability patch: EVERY per-record step below -- not only the initial
    repository read -- is isolated. A single malformed row (e.g. a
    `triage_confidence` outside the valid `[0, 1]` range stored by some
    future bug, which the database schema itself does not constrain) can
    never raise out of this loop and abort the whole run's fresh-candidate
    processing: it is caught, diagnosed by article id, and the record is
    explicitly quarantined (`BacklogStatus.QUARANTINED`) rather than
    silently skipped-and-left-looking-pending-forever or, worse, marked
    completed/irrelevant. Every OTHER valid record in the same batch is
    still processed normally.

    Returns `(never_triaged, already_triaged_pairs, fairness_counts,
    records_by_article_id, extra_updates)` -- `extra_updates` is a
    ready-made list of `BacklogUpdate`s (status `REJECTED_DATE_GATE` or
    `QUARANTINED`) for the caller to persist directly, so a candidate that
    no longer qualifies this run is explicitly closed out rather than left
    looking pending forever."""
    never_triaged: List[TriageCandidate] = []
    already_triaged: List[Tuple[TriageCandidate, TriageItemResult]] = []
    fairness_counts: Dict[str, int] = {}
    records_by_article_id: Dict[str, Any] = {}
    extra_updates: List[BacklogUpdate] = []

    if services.persistence_service is None or state.period_start is None or state.period_end is None:
        return never_triaged, already_triaged, fairness_counts, records_by_article_id, extra_updates

    current_llm_settings = services.config.settings.current_llm
    agent_values = [a.value for a in services.agents.keys()]
    try:
        records = services.persistence_service.get_pending_backlog_candidates(
            limit=current_llm_settings.backlog_claim_limit, agents=agent_values,
        )
    except Exception as exc:  # a backlog read failure must never abort an otherwise-healthy run
        diagnostics.append(PipelineDiagnostic(
            stage="agent_analysis", message=f"backlog:read_failed {exc}", severity="warning",
        ))
        return never_triaged, already_triaged, fairness_counts, records_by_article_id, extra_updates

    for record in records:
        # Isolation boundary (per record, not just per batch): everything
        # from here to the bottom of the loop body is one candidate's own
        # work. `article_id_for_diagnostics` is captured defensively BEFORE
        # anything that could raise, so even a malformed `record` itself
        # still produces an identifiable diagnostic/quarantine entry
        # instead of a bare, unattributed exception.
        article_id_for_diagnostics = "unknown"
        agent_for_diagnostics = "unknown"
        row_for_diagnostics = None
        try:
            article, evidence, row = record.article, record.evidence, record.backlog
            article_id_for_diagnostics = article.id
            agent_for_diagnostics = row.agent
            row_for_diagnostics = row
            records_by_article_id[article.id] = record
            fairness_counts[article.id] = row.attempt_count + row.defer_count

            in_period = (
                article.publication_date is not None
                and article.publication_date_precision in _CURRENT_TRUSTWORTHY_PRECISIONS
                and state.period_start <= article.publication_date < state.period_end
            )
            if not in_period or not _passes_current_deterministic_gate(article, evidence):
                diagnostics.append(PipelineDiagnostic(
                    stage="agent_analysis",
                    message=(
                        f"backlog:rejected_date_gate candidate={article.id} agent={row.agent} "
                        f"title={(article.title or '')[:80]!r} url={article.url} "
                        f"publication_date={article.publication_date.isoformat() if article.publication_date else 'None'}"
                    ),
                    severity="warning",
                ))
                extra_updates.append(BacklogUpdate(
                    article_id=article.id, agent=row.agent, status=BacklogStatus.REJECTED_DATE_GATE.value,
                    reason="no longer falls inside this run's CURRENT period on resume",
                    attempt_count=row.attempt_count, defer_count=row.defer_count, triage_relevant=row.triage_relevant,
                    triage_confidence=row.triage_confidence, first_seen_run_id=row.first_seen_run_id,
                    first_seen_at=row.first_seen_at,
                ))
                continue

            try:
                agent_name = AgentName(row.agent)
            except ValueError:
                continue  # the routed agent no longer exists in this config -- never a crash

            diagnostics.append(PipelineDiagnostic(
                stage="agent_analysis",
                message=(
                    f"backlog:resumed candidate={article.id} agent={row.agent} prior_status={row.status} "
                    f"attempt={row.attempt_count} defer={row.defer_count} title={(article.title or '')[:80]!r}"
                ),
                severity="warning",
            ))
            candidate = TriageCandidate(candidate_id=article.id, article=article, evidence=evidence, primary_agent=agent_name)
            if row.triage_relevant is True:
                already_triaged.append((
                    candidate,
                    TriageItemResult(
                        candidate_id=article.id, relevant=True,
                        confidence=row.triage_confidence if row.triage_confidence is not None else 0.5,
                    ),
                ))
            else:
                never_triaged.append(candidate)
        except Exception as exc:
            # One malformed record is isolated here, never allowed to
            # abort the whole run. Never marked completed/irrelevant --
            # QUARANTINED is excluded from automatic `list_pending` retry
            # (a structurally malformed row would just fail identically
            # every run) but remains fully inspectable and explicitly
            # recoverable by a deliberate operator action, never silently
            # dropped and never silently left claiming to still be
            # ordinary pending work.
            diagnostics.append(PipelineDiagnostic(
                stage="agent_analysis",
                message=(
                    f"backlog:quarantined candidate={article_id_for_diagnostics} "
                    f"agent={agent_for_diagnostics} error={exc!r}"
                ),
                severity="error",
            ))
            if article_id_for_diagnostics != "unknown":
                extra_updates.append(BacklogUpdate(
                    article_id=article_id_for_diagnostics, agent=agent_for_diagnostics,
                    status=BacklogStatus.QUARANTINED.value,
                    reason=f"could not be safely resumed: {exc}",
                    attempt_count=row_for_diagnostics.attempt_count if row_for_diagnostics else 0,
                    defer_count=row_for_diagnostics.defer_count if row_for_diagnostics else 0,
                    first_seen_run_id=row_for_diagnostics.first_seen_run_id if row_for_diagnostics else None,
                    first_seen_at=row_for_diagnostics.first_seen_at if row_for_diagnostics else None,
                ))

    return never_triaged, already_triaged, fairness_counts, records_by_article_id, extra_updates


_CANDIDATE_ID_RE = re.compile(r"candidate=(\S+)")
_DEEP_ANALYSIS_FAILURE_RE = re.compile(r"^deep_analysis:[^:]+:([0-9a-fA-F]+):")


def _candidate_ids_for_prefix(diagnostics: List[PipelineDiagnostic], prefix: str) -> Set[str]:
    ids: Set[str] = set()
    for d in diagnostics:
        if d.message.startswith(prefix):
            match = _CANDIDATE_ID_RE.search(d.message)
            if match:
                ids.add(match.group(1))
    return ids


def _deep_analysis_failure_ids(diagnostics: List[PipelineDiagnostic]) -> Set[str]:
    ids: Set[str] = set()
    for d in diagnostics:
        match = _DEEP_ANALYSIS_FAILURE_RE.match(d.message)
        if match:
            ids.add(match.group(1))
    return ids


def _compute_backlog_updates(
    *,
    considered_ids: Set[str],
    gated_pool: List[TriageCandidate],
    triage_winners: List[Tuple[TriageCandidate, TriageItemResult]],
    research_items: List[ResearchItem],
    diagnostics: List[PipelineDiagnostic],
    records_by_article_id: Dict[str, Any],
    backlog_max_attempts: int,
) -> List[BacklogUpdate]:
    """Derives every considered candidate's disposition purely from this
    run's own diagnostics and outputs -- never by re-deriving/duplicating
    the triage/deep-analysis selection logic itself (zero changes to
    `_run_current_triage_by_domain`/`_select_deep_analysis_winners`/
    `_run_current_deep_analysis`). Guarantees every considered candidate
    gets exactly one explicit disposition -- no eligible candidate
    disappears without one."""
    success_ids = {item.article_id: item.id for item in research_items}
    triage_irrelevant_ids = _candidate_ids_for_prefix(diagnostics, "triage:irrelevant")
    # Reliability fix (CURRENT run `31fa192e4d7a7175`): a deep-analysis call
    # that COMPLETED normally and produced a genuine, considered
    # non-relevant verdict (`agents.base.ResearchAgent.analyze` returned
    # `None`) must be its own terminal disposition -- it was previously
    # unmatched by any category here and fell into the generic fallback,
    # which is meant for a candidate this run genuinely never resolved at
    # all, not a real, final, negative answer. Checked explicitly BEFORE
    # `failure_ids`/budget-skip below so it is never mistaken for either.
    deep_analysis_irrelevant_ids = _candidate_ids_for_prefix(diagnostics, "deep_analysis:irrelevant")
    budget_skip_triage_ids = _candidate_ids_for_prefix(diagnostics, "budget_skip:triage ")
    budget_skip_deep_selection_ids = _candidate_ids_for_prefix(diagnostics, "budget_skip:deep_analysis_selection")
    budget_skip_deep_ids = _candidate_ids_for_prefix(diagnostics, "budget_skip:deep_analysis ")
    failure_ids = _deep_analysis_failure_ids(diagnostics)
    confidence_by_id = {c.candidate_id: r.confidence for c, r in triage_winners}
    agent_by_id = {c.candidate_id: c.primary_agent.value for c in gated_pool}
    agent_by_id.update({c.candidate_id: c.primary_agent.value for c, _ in triage_winners})

    updates: List[BacklogUpdate] = []
    for candidate_id in considered_ids:
        prior = records_by_article_id.get(candidate_id)
        prior_attempts = prior.backlog.attempt_count if prior else 0
        prior_defers = prior.backlog.defer_count if prior else 0
        prior_run_id = prior.backlog.first_seen_run_id if prior else None
        prior_first_seen_at = prior.backlog.first_seen_at if prior else None
        agent_value = agent_by_id.get(candidate_id, prior.backlog.agent if prior else "unknown")

        if candidate_id in success_ids:
            if prior is not None:  # close out a resumed backlog row; a brand-new success needs no backlog row at all
                updates.append(BacklogUpdate(
                    article_id=candidate_id, agent=agent_value, status=BacklogStatus.COMPLETED.value,
                    reason="deep analysis succeeded", research_item_id=success_ids[candidate_id],
                    attempt_count=prior_attempts + 1, defer_count=prior_defers,
                    first_seen_run_id=prior_run_id, first_seen_at=prior_first_seen_at,
                ))
            continue

        if candidate_id in deep_analysis_irrelevant_ids:
            # A real, completed, considered negative verdict -- terminal,
            # never retried automatically, and never confused with a
            # timeout/failure (`failure_ids`, checked next) or a budget
            # deferral. `attempt_count` increments because a genuine
            # analysis attempt really happened.
            updates.append(BacklogUpdate(
                article_id=candidate_id, agent=agent_value,
                status=BacklogStatus.REJECTED_DEEP_ANALYSIS_IRRELEVANT.value,
                reason="deep analysis completed and judged the content not relevant",
                attempt_count=prior_attempts + 1, defer_count=prior_defers, triage_relevant=True,
                triage_confidence=confidence_by_id.get(candidate_id),
                first_seen_run_id=prior_run_id, first_seen_at=prior_first_seen_at,
            ))
            continue

        if candidate_id in failure_ids:
            attempts = prior_attempts + 1
            terminal = attempts >= backlog_max_attempts
            updates.append(BacklogUpdate(
                article_id=candidate_id, agent=agent_value,
                status=(BacklogStatus.REJECTED_RETRY_EXHAUSTED if terminal else BacklogStatus.PENDING_RETRY).value,
                reason=(
                    f"deep analysis failed/timed out {attempts} time(s)"
                    + (f" (backlog_max_attempts={backlog_max_attempts} reached)" if terminal else "; recoverable")
                ),
                attempt_count=attempts, defer_count=prior_defers, triage_relevant=True,
                triage_confidence=confidence_by_id.get(candidate_id),
                first_seen_run_id=prior_run_id, first_seen_at=prior_first_seen_at,
            ))
            continue

        # Genuine lost-opportunity-to-budget cases (Section 1, pre-live-
        # scan reliability patch): `defer_count` increments here, and ONLY
        # here -- never for a discovery/load, never for an explicit
        # irrelevant/date-gate rejection, never for a real analysis attempt
        # (`attempt_count`'s job, above). This is what lets the bounded
        # aging boost actually strengthen across repeated pure-budget
        # deferrals, closing the gap the pre-scan audit found.
        if candidate_id in budget_skip_deep_selection_ids or candidate_id in budget_skip_deep_ids:
            updates.append(BacklogUpdate(
                article_id=candidate_id, agent=agent_value, status=BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value,
                reason="triage relevant, but this run's deep-analysis budget/selection did not reach it",
                attempt_count=prior_attempts, defer_count=prior_defers + 1, triage_relevant=True,
                triage_confidence=confidence_by_id.get(candidate_id),
                first_seen_run_id=prior_run_id, first_seen_at=prior_first_seen_at,
            ))
            continue

        if candidate_id in triage_irrelevant_ids:
            updates.append(BacklogUpdate(
                article_id=candidate_id, agent=agent_value, status=BacklogStatus.REJECTED_TRIAGE_IRRELEVANT.value,
                reason="explicit triage result: irrelevant", attempt_count=prior_attempts, defer_count=prior_defers,
                triage_relevant=False, first_seen_run_id=prior_run_id, first_seen_at=prior_first_seen_at,
            ))
            continue

        if candidate_id in budget_skip_triage_ids:
            updates.append(BacklogUpdate(
                article_id=candidate_id, agent=agent_value, status=BacklogStatus.PENDING_TRIAGE_BUDGET.value,
                reason="passed the deterministic gate but this run's triage-pool budget was exhausted",
                attempt_count=prior_attempts, defer_count=prior_defers + 1,
                first_seen_run_id=prior_run_id, first_seen_at=prior_first_seen_at,
            ))
            continue

        # Defensive fallback -- a candidate reaching here was considered but
        # matched none of the categories above (e.g. its routed agent is
        # disabled in THIS run's config, or its whole triage BATCH raised
        # rather than a per-candidate result). Not a budget deferral in
        # the narrow sense above, so `defer_count` is deliberately left
        # unchanged here -- never silently dropped regardless.
        updates.append(BacklogUpdate(
            article_id=candidate_id, agent=agent_value, status=BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value,
            reason="considered this run but not resolved by any known stage (e.g. its agent is disabled this run, "
                   "or its triage batch itself failed)",
            attempt_count=prior_attempts, defer_count=prior_defers,
            first_seen_run_id=prior_run_id, first_seen_at=prior_first_seen_at,
        ))
    return updates


def _is_timeout_message(exc: Exception) -> bool:
    return "timed out" in str(exc).lower() or "timeout" in str(exc).lower()


def _run_current_deep_analysis(
    triage_winners: List[Tuple[TriageCandidate, TriageItemResult]],
    *, agents: Dict[AgentName, ResearchAgent], deep_client, budget: int,
    diagnostics: List[PipelineDiagnostic],
) -> Tuple[List[ResearchItem], Dict[str, int], bool, int]:
    """`triage_winners` is already ranked strongest-first (inherited from
    the ranked pool triage was run over). `budget` is a GLOBAL cap across
    ALL specialist agents combined (Part D) -- never 8-per-agent. Stops
    the instant the budget is spent; every remaining winner is reported as
    unprocessed rather than silently analyzed anyway."""
    research_items: List[ResearchItem] = []
    attempted = 0
    success = 0
    timeouts = 0
    failures = 0

    for index, (candidate, _triage_result) in enumerate(triage_winners):
        if attempted >= budget:
            unprocessed = len(triage_winners) - index
            # Per-candidate observability (forensic finding, CURRENT run
            # `e818e08adc2e4dd1`): the global deep-analysis budget cutoff
            # previously reported only the aggregate `unprocessed` count --
            # never which specific triage winner(s) never got a slot,
            # distinguishable from an actual deep-analysis call that ran
            # and failed/timed out/judged irrelevant.
            for skipped_candidate, _skipped_result in triage_winners[index:]:
                title = (skipped_candidate.article.title or "")[:80]
                diagnostics.append(
                    PipelineDiagnostic(
                        stage="agent_analysis",
                        message=(
                            f"budget_skip:deep_analysis candidate={skipped_candidate.candidate_id} "
                            f"agent={skipped_candidate.primary_agent.value} "
                            f"source={skipped_candidate.article.source_name!r} title={title!r} "
                            f"url={skipped_candidate.article.url} reason=deep_analysis_budget"
                        ),
                        severity="warning",
                    )
                )
            return research_items, {
                "deep_analysis_attempted": attempted, "deep_analysis_success": success,
                "deep_analysis_timeouts": timeouts, "deep_analysis_failures": failures,
            }, True, unprocessed

        agent = agents.get(candidate.primary_agent)
        if agent is None:  # the routed-to agent is disabled this run
            continue
        attempted += 1
        try:
            item = agent.analyze(candidate.article, candidate.evidence, llm_client=deep_client)
        except Exception as exc:
            if isinstance(exc, LLMUnavailableError) and _is_timeout_message(exc):
                timeouts += 1
            else:
                failures += 1
            diagnostics.append(
                PipelineDiagnostic(
                    stage="agent_analysis",
                    message=f"deep_analysis:{candidate.primary_agent.value}:{candidate.candidate_id}: {exc}",
                    severity="warning",
                )
            )
            logger.warning(
                "agent_analysis(CURRENT): deep analysis failed on %s (%s): %s",
                candidate.candidate_id, candidate.primary_agent.value, exc,
            )
            continue
        if item is not None:
            success += 1
            research_items.append(item)
            if item.relevance_override is not None:
                # The deterministic recall safeguard (agents.base.
                # _resolve_relevance_override) overrode a Qwen IRRELEVANT
                # verdict -- always surfaced here too, never only buried in
                # the persisted ResearchItem's own column.
                diagnostics.append(
                    PipelineDiagnostic(
                        stage="agent_analysis",
                        message=(
                            f"relevance_override candidate={candidate.candidate_id} "
                            f"agent={candidate.primary_agent.value} source={candidate.article.source_name!r} "
                            f"url={candidate.article.url} {item.relevance_override}"
                        ),
                        severity="warning",
                    )
                )
        else:
            # Lightweight per-candidate observability (forensic finding,
            # run 8c605b0881e8edf0): a deep-analysis call that completes
            # without an exception but returns None (the specialist's own
            # `relevance.label == IRRELEVANT` verdict) previously left NO
            # trace anywhere -- `deep_analysis_success/timeouts/failures`
            # all correctly stay at 0 for this outcome (it is not a
            # failure), but that made it indistinguishable from a silent
            # drop without reading this diagnostic.
            diagnostics.append(
                PipelineDiagnostic(
                    stage="agent_analysis",
                    message=(
                        f"deep_analysis:irrelevant candidate={candidate.candidate_id} "
                        f"agent={candidate.primary_agent.value} source={candidate.article.source_name!r} "
                        f"title={(candidate.article.title or '')[:80]!r} url={candidate.article.url}"
                    ),
                    severity="warning",
                )
            )

    return research_items, {
        "deep_analysis_attempted": attempted, "deep_analysis_success": success,
        "deep_analysis_timeouts": timeouts, "deep_analysis_failures": failures,
    }, False, 0


def _run_current_link_grounded(
    routing, evidence_by_article_id: Dict[str, Any], agents_settings, config: AppConfig,
    enabled_agent_names,
) -> Tuple[List[ResearchItem], int]:
    """Identical rule/order to the pre-existing per-agent loop's
    link-grounded branch -- only extracted so it can run once, independent
    of the new CONTENT-path funnel above. Zero LLM calls, never gated by
    triage/deep-analysis budget (Part F). `enabled_agent_names` mirrors the
    original loop's `for agent_name, agent in services.agents.items()`:
    only agents actually present in `services.agents` ever produce a
    LINK-grounded item, exactly as before."""
    research_items: List[ResearchItem] = []
    link_grounded_total = 0
    for agent_name in ROUTABLE_AGENTS:
        if agent_name not in enabled_agent_names:
            continue
        settings = getattr(agents_settings, agent_name.value)
        if not settings.enabled:
            continue
        routed_articles = routing.by_agent.get(agent_name, [])
        link_grounded_candidates = [a for a in routed_articles if a.id not in evidence_by_article_id]
        for article in link_grounded_candidates:
            link_item = build_link_grounded_research_item(article, config=config, agent=agent_name)
            if link_item is not None:
                research_items.append(link_item)
                link_grounded_total += 1
    return research_items, link_grounded_total


@dataclass
class PipelineServices:
    """Everything a node needs, injected once at graph-build time. Duck-typed
    (not `isinstance`-checked) so tests can supply fakes for
    `retrieval_service`/`normalization_service` without subclassing the real
    Phase 1/2 services."""

    config: AppConfig
    retrieval_service: Any  # .list_sources() -> List[SourceDefinition]; .retrieve_source(source) -> List[ExtractedArticle]
    normalization_service: Any  # .normalize(extracted, source=...) -> Article; .deduplicate(articles); .to_evidence_many(articles)
    agents: Dict[AgentName, ResearchAgent]
    event_service: EventService
    report_agent: ReportAgent
    digest_service: DigestService
    clock: Clock = field(default_factory=SystemClock)
    persistence_service: Any = None  # .persist_pipeline_result(...) -> None; None disables persistence entirely
    # CURRENT-only bounded LLM funnel (see `make_agent_analysis_node`'s
    # CURRENT-mode path): dedicated LLM clients with their own, SHORTER
    # timeouts (current_llm.triage_timeout_seconds / .deep_analysis_timeout_
    # seconds) -- deliberately separate instances from each agent's own
    # `self._llm` (the global, unchanged 150s timeout Morning/Evening/manual
    # keep using). None in every existing test/construction site; real
    # CURRENT runs get them from `PipelineRunner.build()`. Unused entirely
    # for Morning/Evening.
    triage_llm_client: Any = None
    deep_analysis_llm_client: Any = None


def make_load_config_node(services: PipelineServices):
    def load_config(state: PipelineState) -> dict:
        return {"config": services.config}

    return load_config


def make_discover_sources_node(services: PipelineServices):
    def discover_sources(state: PipelineState) -> dict:
        sources = services.retrieval_service.list_sources()
        logger.info("discover_sources: %d source(s) configured", len(sources))
        return {"sources": sources, "metrics": {**state.metrics, "sources_discovered": len(sources)}}

    return discover_sources


def make_retrieve_node(services: PipelineServices):
    def retrieve(state: PipelineState) -> dict:
        bundles: List[ExtractedBundle] = []
        diagnostics: List[PipelineDiagnostic] = []
        succeeded = 0
        failed = 0
        # CURRENT's [now-24h, now) contract requires fresh retrieval --
        # PageCache has no TTL (see retrieval.cache's own docstring), so
        # without this a page fetched once would be reused forever,
        # structurally hiding anything published since that snapshot from
        # CURRENT regardless of ranking/budget/LLM behavior. Morning/Evening/
        # manual runs are completely unaffected -- CachePolicy.CACHE,
        # passed here for every other digest_type, is the exact same
        # cache-reading behavior this already had before CachePolicy existed.
        cache_policy = CachePolicy.FRESH if state.digest_type == DigestType.CURRENT else CachePolicy.CACHE
        for source in state.sources:
            try:
                extracted_list = services.retrieval_service.retrieve_source(source, cache_policy=cache_policy)
            except Exception as exc:  # one bad source must not abort the whole run
                failed += 1
                # Honest classification (TLS_CERTIFICATE_ERROR, HTTP_401,
                # ANTI_BOT_ACCESS_BLOCKED, ...) appended after the reason --
                # never before it, since dashboard_data._latest_retrieve_failures
                # parses this message as exactly `"{source_name}: {reason}"`
                # (split on the first ": ") and must keep getting a bare
                # source name. Only SourceUnreachableError carries the
                # structured per-URL failures this needs; any other
                # exception still gets a diagnostic, just unclassified.
                labels = classify_many(exc.failures) if isinstance(exc, SourceUnreachableError) else []
                suffix = f" [{', '.join(labels)}]" if labels else ""
                message = f"{source.name}: {exc}{suffix}"
                diagnostics.append(PipelineDiagnostic(stage="retrieve", message=message))
                logger.warning("retrieve: %s failed (%s): %s", source.name, labels or "unclassified", exc)
                continue
            succeeded += 1
            if not extracted_list:
                # Phase 13E: a source that was reachable but yielded zero
                # candidates is a categorically different outcome from a
                # fetch failure -- without this, both looked identical
                # ("no articles for this source"), making it impossible to
                # tell "genuinely nothing new" apart from "something is
                # silently wrong with this source's discovery".
                diagnostics.append(
                    PipelineDiagnostic(
                        stage="retrieve", severity="warning",
                        message=f"{source.name}: reachable but yielded zero candidates this run [ZERO_CANDIDATES]",
                    )
                )
            bundles.extend(ExtractedBundle(source=source, extracted=extracted) for extracted in extracted_list)
        logger.info(
            "retrieve: sources_attempted=%d sources_succeeded=%d sources_failed=%d candidates_discovered=%d",
            len(state.sources), succeeded, failed, len(bundles),
        )
        return {
            "extracted_bundles": bundles,
            "diagnostics": state.diagnostics + diagnostics,
            "metrics": {
                **state.metrics,
                "sources_attempted": len(state.sources),
                "sources_succeeded": succeeded,
                "sources_failed": failed,
                "candidates_discovered": len(bundles),
            },
        }

    return retrieve


def make_normalize_node(services: PipelineServices):
    def normalize(state: PipelineState) -> dict:
        articles = []
        diagnostics: List[PipelineDiagnostic] = []
        failures = 0
        for bundle in state.extracted_bundles:
            try:
                articles.append(services.normalization_service.normalize(bundle.extracted, source=bundle.source))
            except Exception as exc:  # one bad article must not abort the whole run
                failures += 1
                diagnostics.append(
                    PipelineDiagnostic(stage="normalize", message=f"{bundle.extracted.url}: {exc}")
                )
                logger.warning("normalize: %s failed: %s", bundle.extracted.url, exc)
        deduped = services.normalization_service.deduplicate(articles)

        # Phase 9: article HISTORY/PERSISTENCE uses a wider retention window
        # than any single digest's own reporting period -- the existing,
        # previously-unused `research.lookback_hours` config value. This is
        # deliberately separate from the narrower window `agent_analysis`
        # applies before any LLM call (see that node), so widening history
        # retention here never increases LLM call volume.
        lookback_start = _compute_lookback_start(
            state.period_start, state.period_end, services.config.settings.research.lookback_hours
        )
        # calendar_date_aware only for Morning/Evening (Part 1 fix): CURRENT's
        # exact [now-24h, now) contract is deliberately left conservative/
        # unchanged -- see _filter_by_period's docstring.
        in_history, out_of_history = _filter_by_period(
            deduped, lookback_start, state.period_end,
            calendar_date_aware=state.digest_type != DigestType.CURRENT,
        )
        # Discovery vs. content-retrieval separation: an article whose body
        # could not be retrieved (robots/anti-bot/HTTP/browser failure --
        # see RetrievalService.extract's content_unavailable_reason) is
        # still a genuine, retained discovery -- it survives into
        # `in_history` exactly like any other article, with no clean_text,
        # which `evidence.conversion.article_to_evidence` already refuses
        # to turn into Evidence (unchanged gate). This count is purely an
        # observability metric -- it never gates anything downstream.
        content_unavailable = sum(
            1 for a in in_history if a.extraction_status in (ExtractionStatus.FAILED, ExtractionStatus.NOT_ARTICLE)
        )
        # Diagnostic fix (forensic finding, CURRENT run `e818e08adc2e4dd1`):
        # `articles_excluded_out_of_lookback` was previously one opaque
        # number that could not distinguish "genuinely no publication date"
        # (never excluded here -- see `_filter_by_period`'s own docstring)
        # from "had a real, known publication_date that nonetheless landed
        # outside the lookback window" -- exactly what happened, invisibly,
        # to six real PR Newswire releases whose extracted timestamp had
        # collapsed to midnight UTC. Narrowly additive: counts, never gates
        # anything, and changes no existing metric's meaning.
        excluded_with_known_date = sum(1 for a in out_of_history if a.publication_date is not None)
        logger.info(
            "normalize: normalized=%d after_dedup=%d excluded_out_of_lookback=%d "
            "excluded_out_of_lookback_with_known_date=%d failures=%d content_unavailable=%d",
            len(articles), len(deduped), len(out_of_history), excluded_with_known_date, failures, content_unavailable,
        )
        return {
            "articles": in_history,
            "diagnostics": state.diagnostics + diagnostics,
            "metrics": {
                **state.metrics,
                "articles_normalized": len(articles),
                "articles_after_dedup": len(deduped),
                "articles_excluded_out_of_lookback": len(out_of_history),
                "articles_excluded_out_of_lookback_with_known_date": excluded_with_known_date,
                "normalize_failures": failures,
                "articles_discovered_content_unavailable": content_unavailable,
            },
        }

    return normalize


def _compute_lookback_start(period_start, period_end, lookback_hours: int):
    """The lower bound of the wider article-history retention window (Phase
    9): `lookback_hours` before this run's `period_end`, but never narrower
    than the digest's own `period_start` (history always contains at least
    what this digest itself reports on). `None` when the run has no period
    configured at all, preserving `_filter_by_period`'s existing no-op
    behavior for period-less runs (every pre-Phase-8 test)."""
    if period_end is None:
        return None
    lookback_bound = period_end - timedelta(hours=lookback_hours)
    if period_start is not None:
        return min(period_start, lookback_bound)
    return lookback_bound


def _filter_by_period(articles: list, period_start, period_end, *, calendar_date_aware: bool = False) -> tuple:
    """Keep an article unless its `publication_date` is *known* and falls
    outside `[period_start, period_end]`. An article with no publication
    date is always retained: `retrieved_at` is never substituted as a proxy
    for publication time, so a dateless article can't be proven out of the
    reporting window and is not silently dropped. A no-op (returns every
    article, unfiltered) when `period_start`/`period_end` are both `None`.
    Reused for two distinct purposes (Phase 9): the wide article-history
    window in `normalize()`, and the narrow digest-reporting window in
    `agent_analysis()` — the same logic, different bounds.

    `calendar_date_aware` (Morning/Evening only -- see call sites, never
    passed True for CURRENT): a `DatePrecision.DATE_ONLY` article's
    `publication_date` is stored at UTC midnight purely as a storage
    convention (see `evidence.normalizer`'s docstring) -- it records that
    the article was published SOMEWHERE on that calendar day, never that
    midnight was the observed instant. Comparing that midnight instant
    directly against `period_start` (as the `else` branch below does, and
    as this function always did before this parameter existed) wrongly
    excludes a same-day article merely because the digest's period starts
    later in the day than midnight -- confirmed root cause of the empty
    2026-10-04 Evening digest (FT articles dated that day, Evening's period
    beginning at 02:30 UTC). When True, a DATE_ONLY article is instead kept
    if its full UTC calendar day overlaps the period AT ALL -- never
    inventing a specific publication time (23:59:59 or otherwise), and
    never changing `publication_date`/`publication_date_precision`
    themselves (both pass through unmodified; only this eligibility
    decision changes). An EXACT/UNKNOWN-precision article's handling is
    completely unchanged regardless of this flag."""
    if period_start is None or period_end is None:
        return list(articles), []
    kept, excluded = [], []
    for article in articles:
        if article.publication_date is None:
            kept.append(article)
        elif calendar_date_aware and article.publication_date_precision == DatePrecision.DATE_ONLY:
            if _date_only_calendar_day_overlaps_period(article.publication_date, period_start, period_end):
                kept.append(article)
            else:
                excluded.append(article)
        elif period_start <= article.publication_date <= period_end:
            kept.append(article)
        else:
            excluded.append(article)
    return kept, excluded


def _date_only_calendar_day_overlaps_period(publication_date: datetime, period_start: datetime, period_end: datetime) -> bool:
    """`publication_date` is a DATE_ONLY value: UTC midnight on the known
    calendar day, with no real time-of-day information (see
    `_filter_by_period`'s docstring). Treats it as "the true publication
    instant is SOMEWHERE in `[day_start_utc, day_end_utc)`" -- the full UTC
    calendar day the stored midnight value names -- and asks whether that
    whole day overlaps the digest's `[period_start, period_end)` at all.
    Standard half-open interval overlap: two intervals `[a, b)` and `[c, d)`
    overlap iff `a < d and c < b`. Never narrows to a single invented
    instant."""
    day_start_utc = publication_date  # already UTC midnight for DATE_ONLY
    day_end_utc = day_start_utc + timedelta(days=1)
    return day_start_utc < period_end and period_start < day_end_utc


def _source_category(article, config: AppConfig) -> str:
    """Which configured registry section this article's source belongs to
    -- for diagnostics only (mirrors, without importing, the same
    source_name-membership logic `agents.router` uses for routing). There
    is no fixed "startups" source registry in `config/sources.yaml` (see
    `retrieval.service._SCRAPABLE_CATEGORIES`), so startup-relevant content
    can only ever arrive as a secondary/press-distribution article and is
    reported under "secondary_sources", never a separate "startups" bucket.
    """
    if article.source_name in {s.name for s in config.sources.peers}:
        return "peers"
    if article.source_name in {s.name for s in config.sources.ai_providers}:
        return "ai_providers"
    if article.source_name in {s.name for s in config.sources.analysts}:
        return "analysts"
    if article.source_name in {s.name for s in config.sources.secondary_sources}:
        return "secondary_sources"
    return "other"


def make_evidence_node(services: PipelineServices):
    def build_evidence(state: PipelineState) -> dict:
        # Unchanged from before: the injected normalization_service's own
        # batch conversion is the single source of truth for what evidence
        # actually gets produced -- every test fake already implements this
        # one method, so this call is untouched. The pure
        # `evidence_rejection_reason()` helper below is then used purely to
        # *explain* any article that didn't end up with evidence, never to
        # decide inclusion itself.
        evidence: List = services.normalization_service.to_evidence_many(state.articles)
        evidence_article_ids = {e.article_id for e in evidence}

        diagnostics: List[PipelineDiagnostic] = []
        rejected_missing_clean_text = 0
        rejected_missing_content_hash = 0
        rejected_missing_both = 0
        rejected_other = 0
        by_source_category: Dict[str, int] = {}

        for article in state.articles:
            category = _source_category(article, services.config)
            if article.id in evidence_article_ids:
                by_source_category[category] = by_source_category.get(category, 0) + 1
                continue

            reason = evidence_rejection_reason(article)
            if reason == "missing_clean_text_and_content_hash":
                rejected_missing_both += 1
            elif reason == "missing_clean_text":
                rejected_missing_clean_text += 1
            elif reason == "missing_content_hash":
                rejected_missing_content_hash += 1
            else:
                # The pure gate says this article SHOULD have produced
                # evidence, yet the injected service didn't -- an
                # unexpected discrepancy, surfaced rather than hidden.
                rejected_other += 1
                diagnostics.append(
                    PipelineDiagnostic(
                        stage="evidence",
                        message=f"{article.id}: expected evidence but none was produced (no known rejection reason)",
                        severity="error",
                    )
                )

            # Per-article detail: logs only (never PipelineState.diagnostics,
            # to avoid flooding the run's diagnostic list on a real run with
            # many rejections) -- never the article body itself.
            logger.info(
                "evidence: rejected article_id=%s source_name=%s source_type=%s category=%s "
                "reason=%s title=%r publication_date=%s extraction_status=%s extraction_method=%s "
                "clean_text_length=%d content_hash_present=%s",
                article.id, article.source_name, article.source_type.value, category, reason,
                article.title, article.publication_date.isoformat() if article.publication_date else "none",
                article.extraction_status.value, article.extraction_method.value,
                len(article.clean_text) if article.clean_text else 0, article.content_hash is not None,
            )

        total_rejected = rejected_missing_clean_text + rejected_missing_content_hash + rejected_missing_both + rejected_other
        logger.info(
            "evidence: input_articles=%d evidence_created=%d rejected_missing_clean_text=%d "
            "rejected_missing_content_hash=%d rejected_missing_both=%d rejected_other=%d by_source_category=%s",
            len(state.articles), len(evidence), rejected_missing_clean_text,
            rejected_missing_content_hash, rejected_missing_both, rejected_other, by_source_category,
        )
        if total_rejected > 0:
            diagnostics.append(
                PipelineDiagnostic(
                    stage="evidence",
                    message=(
                        f"input_articles={len(state.articles)} evidence_created={len(evidence)} "
                        f"rejected_missing_clean_text={rejected_missing_clean_text} "
                        f"rejected_missing_content_hash={rejected_missing_content_hash} "
                        f"rejected_missing_both={rejected_missing_both} rejected_other={rejected_other} "
                        f"by_source_category={by_source_category}"
                    ),
                    severity="warning",
                )
            )

        return {
            "evidence": evidence,
            "diagnostics": state.diagnostics + diagnostics,
            "metrics": {
                **state.metrics,
                "evidence_created": len(evidence),
                "evidence_rejected_missing_clean_text": rejected_missing_clean_text,
                "evidence_rejected_missing_content_hash": rejected_missing_content_hash,
                "evidence_rejected_missing_both": rejected_missing_both,
                "evidence_rejected_other": rejected_other,
                **{f"evidence_by_category_{k}": v for k, v in by_source_category.items()},
            },
        }

    return build_evidence


def make_agent_analysis_node(services: PipelineServices):
    def agent_analysis(state: PipelineState) -> dict:
        evidence_by_article_id = {e.article_id: e for e in state.evidence}
        agents_settings = services.config.settings.agents
        research_items: List[ResearchItem] = []
        diagnostics: List[PipelineDiagnostic] = []
        candidates_considered_total = 0
        candidates_analyzed_total = 0
        agent_failures_total = 0
        # CURRENT-only diagnostics (Part B7) -- stay zero/unset for
        # Morning/Evening, never affect their metrics.
        current_candidates_ranked_total = 0
        current_candidates_analyzed_total = 0
        current_candidates_skipped_by_budget_total = 0
        link_grounded_total = 0
        backlog_updates: List[BacklogUpdate] = []

        # Analysis candidate selection (Phase 9): state.articles now holds
        # the WIDER article-history window (see normalize()'s use of
        # research.lookback_hours) -- narrow back down to exactly this
        # run's own digest period before routing/LLM calls, so widening
        # history retention never increases LLM call volume. Reuses the
        # same _filter_by_period() helper normalize() uses, just with the
        # original (narrow) period bounds instead of the lookback-widened
        # ones already applied there.
        period_eligible_articles, out_of_digest_period = _filter_by_period(
            state.articles, state.period_start, state.period_end,
            calendar_date_aware=state.digest_type != DigestType.CURRENT,
        )
        if out_of_digest_period:
            logger.info(
                "agent_analysis: %d article(s) retained for history but outside this digest's own "
                "period are excluded from analysis", len(out_of_digest_period),
            )

        # CURRENT's hard upstream date gate (see `_apply_current_date_gate`'s
        # own docstring): an undated/stale/future-dated article is removed
        # HERE, before routing -- it can never reach routing, triage,
        # specialist analysis, ResearchItem, or Event creation. Morning/
        # Evening are completely untouched (they keep `_filter_by_period`'s
        # existing, unchanged lenient behavior -- this gate never runs for
        # them).
        current_date_gate_metrics = {"current_date_qualified": 0, "current_dropped_undated": 0,
                                      "current_dropped_stale": 0, "current_dropped_future": 0}
        if state.digest_type == DigestType.CURRENT and state.period_start is not None and state.period_end is not None:
            digest_eligible_articles, gate_counts = _apply_current_date_gate(
                period_eligible_articles, state.period_start, state.period_end, diagnostics=diagnostics,
            )
            current_date_gate_metrics = {
                "current_date_qualified": len(digest_eligible_articles),
                "current_dropped_undated": gate_counts["undated"],
                "current_dropped_stale": gate_counts["stale"],
                "current_dropped_future": gate_counts["future"],
            }
            logger.info(
                "agent_analysis(CURRENT date gate): period_candidates=%d date_qualified=%d "
                "dropped_undated=%d dropped_stale=%d dropped_future=%d",
                len(period_eligible_articles), len(digest_eligible_articles),
                gate_counts["undated"], gate_counts["stale"], gate_counts["future"],
            )
        else:
            digest_eligible_articles = period_eligible_articles

        # Deterministic routing happens once, up front, entirely from
        # config + article metadata -- zero LLM calls. This decides *which*
        # agents an article is even eligible for; each agent's own keyword
        # pre-filter and LLM relevance call (below) still runs only on its
        # routed subset, never on the full article pool.
        routing = route_articles(digest_eligible_articles, services.config)
        topic_keywords = _flatten_topic_keywords(services.config.topics)
        event_type_phrases = _flatten_event_type_phrases(services.config.topics)
        logger.info(
            "routing: total_articles=%d peer=%d ai_product=%d analyst=%d startup=%d financial_news=%d "
            "unrouted=%d multi_routed=%d",
            len(digest_eligible_articles),
            len(routing.by_agent.get(AgentName.PEER, [])),
            len(routing.by_agent.get(AgentName.AI_PRODUCT, [])),
            len(routing.by_agent.get(AgentName.ANALYST, [])),
            len(routing.by_agent.get(AgentName.STARTUP, [])),
            len(routing.by_agent.get(AgentName.FINANCIAL_NEWS, [])),
            routing.unrouted_count,
            routing.multi_routed_count,
        )
        # Observability (Part 25): per-domain routed-candidate counts, so an
        # admin can distinguish "retrieval found nothing for this domain"
        # from "retrieval succeeded but this domain's relevance filtering
        # rejected everything" downstream -- real routing counts, never
        # fabricated.
        routing_metrics = {
            "routed_peer": len(routing.by_agent.get(AgentName.PEER, [])),
            "routed_ai_technology": len(routing.by_agent.get(AgentName.AI_PRODUCT, [])),
            "routed_analyst": len(routing.by_agent.get(AgentName.ANALYST, [])),
            "routed_startup_funding": len(routing.by_agent.get(AgentName.STARTUP, [])),
            "routed_financial_news": len(routing.by_agent.get(AgentName.FINANCIAL_NEWS, [])),
            "routed_unrouted": routing.unrouted_count,
            "routed_multi_routed": routing.multi_routed_count,
        }

        if state.digest_type == DigestType.CURRENT and routing.unrouted_count > 0:
            # Per-article observability (forensic finding, CURRENT run
            # `e818e08adc2e4dd1`): previously only `routed_unrouted`'s
            # aggregate count existed -- never which specific article never
            # matched any agent's routing rules. `RoutingResult` itself
            # only carries the count, never the articles, so this is
            # recomputed here from the same two already-available lists
            # (never a routing-logic change): any digest-eligible article
            # whose id does not appear in ANY agent's routed list was
            # unrouted.
            routed_ids = {a.id for articles_for_agent in routing.by_agent.values() for a in articles_for_agent}
            for article in digest_eligible_articles:
                if article.id not in routed_ids:
                    title = (article.title or "")[:80]
                    diagnostics.append(
                        PipelineDiagnostic(
                            stage="agent_analysis",
                            message=(
                                f"routing:unrouted candidate={article.id} source={article.source_name!r} "
                                f"title={title!r} url={article.url}"
                            ),
                            severity="warning",
                        )
                    )

        is_current = state.digest_type == DigestType.CURRENT
        current_funnel_metrics: Dict[str, Any] = {}

        if is_current:
            # ---- BOUNDED LLM FUNNEL (CURRENT only) ----
            current_llm_settings = services.config.settings.current_llm
            pool = _build_current_candidate_pool(routing, evidence_by_article_id, services.agents.keys())
            candidates_before_llm = len(pool)

            # Durable candidate backlog (read-only here; see
            # `_resumable_backlog_candidates`'s own docstring): a candidate
            # deferred by a PRIOR run's exhausted triage/deep-analysis
            # budget is resumed and revalidated against THIS run's own
            # period before being merged in -- never silently lost to a
            # one-run budget, never smuggled into a period it wasn't
            # actually confirmed to fall inside.
            backlog_never_triaged, backlog_already_triaged, backlog_fairness_counts, backlog_records_by_id, \
                backlog_extra_updates = _resumable_backlog_candidates(services, state, diagnostics=diagnostics)

            gated_pool = [c for c in pool if _passes_current_deterministic_gate(c.article, c.evidence)]
            gated_pool_ids = {c.candidate_id for c in gated_pool}
            # A backlog candidate independently re-discovered fresh THIS
            # run is never duplicated into the pool -- the fresh instance
            # (freshly re-retrieved content) always wins; the backlog row
            # itself is reconciled below via `_compute_backlog_updates`
            # once this run's outcome for that article id is known.
            gated_pool.extend(c for c in backlog_never_triaged if c.candidate_id not in gated_pool_ids)
            gated_pool.sort(
                key=lambda c: _aging_boosted_rank_key(
                    c.article, c.evidence, topic_keywords, event_type_phrases, backlog_fairness_counts,
                ),
                reverse=True,
            )
            candidates_considered_total = candidates_before_llm

            # Every candidate this run actually considers (freshly routed,
            # merged from backlog, or resumed already-triaged) -- used
            # afterward to guarantee each one gets an explicit disposition.
            backlog_already_triaged_ids = {c.candidate_id for c, _ in backlog_already_triaged}
            considered_ids = gated_pool_ids | {c.candidate_id for c in backlog_never_triaged} | backlog_already_triaged_ids

            if gated_pool or backlog_already_triaged:
                # Deliberately NEVER falls back to a real Ollama client here
                # -- only `PipelineRunner.build()` (the real, production
                # construction path) ever creates one. A CURRENT run with
                # real candidates to triage but no injected client is a
                # configuration error, not something to silently paper over
                # with a live network call (which would make tests depend
                # on a live Ollama server, explicitly forbidden). A run with
                # NO gated candidates at all (e.g. LINK-only fixtures) never
                # needs a client in the first place -- zero LLM calls either
                # way, so no client requirement either.
                if services.triage_llm_client is None or services.deep_analysis_llm_client is None:
                    raise RuntimeError(
                        "CURRENT requires PipelineServices.triage_llm_client and .deep_analysis_llm_client "
                        "to be set (see PipelineRunner.build()); got None -- refusing to silently fall back "
                        "to a real Ollama client."
                    )
                # Domain-fair budget allocation -> per-domain source
                # diversity -> domain-PURE triage batches (never a single
                # mixed-domain LLM call) -- see `_run_current_triage_by_
                # domain`'s own docstring. Reduces to the exact pre-refactor
                # single-pool call when `gated_pool` only ever routes to one
                # domain (every pre-existing single-domain funnel test).
                triage_pool_bound = current_llm_settings.triage_batch_size * current_llm_settings.max_triage_batches
                triage_winners, triage_metrics = _run_current_triage_by_domain(
                    gated_pool, triage_client=services.triage_llm_client,
                    batch_size=current_llm_settings.triage_batch_size,
                    triage_pool_bound=triage_pool_bound,
                    diagnostics=diagnostics,
                )
                # Resumed backlog candidates with a valid PERSISTED triage
                # result bypass triage entirely ("avoid rerunning triage
                # for candidates with a valid persisted triage result") --
                # merged in as already-decided winners, never re-triaged,
                # never costing another triage LLM call.
                triage_winners.extend(backlog_already_triaged)
                # Winners are gathered domain-by-domain above (each
                # internally still strongest-first); re-sort the MERGED
                # list back into one global strongest-first order so
                # `_run_current_deep_analysis`'s budget cutoff still spends
                # the scarce deep-analysis budget on the overall strongest
                # candidates first, never favoring one domain's weaker
                # candidates merely because that domain was processed first.
                # The same bounded aging boost applied to `gated_pool`
                # above is applied here too, so a repeatedly-deferred
                # backlog candidate is not perpetually out-ranked by a
                # continuous stream of fresher candidates.
                triage_winners.sort(
                    key=lambda pair: _aging_boosted_rank_key(
                        pair[0].article, pair[0].evidence, topic_keywords, event_type_phrases, backlog_fairness_counts,
                    ),
                    reverse=True,
                )
                # Domain-fair cut to the (deliberately small, CPU-only)
                # deep-analysis budget BEFORE the sequential loop, so a
                # small budget still spends across specialist domains
                # instead of being consumed entirely by whichever domain
                # ranked globally strongest (Part 3/"domain-aware selection").
                deep_analysis_pool, winners_dropped_by_selection = _select_deep_analysis_winners(
                    triage_winners, current_llm_settings.max_deep_analysis, diagnostics=diagnostics,
                )
                deep_analysis_pool.sort(
                    key=lambda pair: _aging_boosted_rank_key(
                        pair[0].article, pair[0].evidence, topic_keywords, event_type_phrases, backlog_fairness_counts,
                    ),
                    reverse=True,
                )
                deep_items, deep_metrics, budget_exhausted, unprocessed_after_budget = _run_current_deep_analysis(
                    deep_analysis_pool, agents=services.agents, deep_client=services.deep_analysis_llm_client,
                    budget=current_llm_settings.max_deep_analysis, diagnostics=diagnostics,
                )
                unprocessed_after_budget += winners_dropped_by_selection
                budget_exhausted = budget_exhausted or winners_dropped_by_selection > 0
            else:
                triage_winners, triage_metrics = [], {
                    "triage_batches_attempted": 0, "triage_candidates": 0, "triage_relevant": 0,
                    "triage_irrelevant": 0, "triage_failures": 0,
                    "candidates_sent_to_triage": 0, "candidates_truncated_before_triage": 0,
                }
                deep_items, deep_metrics, budget_exhausted, unprocessed_after_budget = [], {
                    "deep_analysis_attempted": 0, "deep_analysis_success": 0,
                    "deep_analysis_timeouts": 0, "deep_analysis_failures": 0,
                }, False, 0
            research_items.extend(deep_items)

            # Every considered candidate's explicit disposition (Section 9:
            # "no eligible candidate disappears without a disposition"),
            # derived purely from this run's own diagnostics/outputs above
            # -- never a second, parallel decision mechanism. Collected
            # into `state.backlog_updates` and persisted atomically by
            # `make_persistence_node`, never written here mid-run.
            run_backlog_updates = _compute_backlog_updates(
                considered_ids=considered_ids,
                gated_pool=gated_pool,
                triage_winners=triage_winners,
                research_items=deep_items,
                diagnostics=diagnostics,
                records_by_article_id=backlog_records_by_id,
                backlog_max_attempts=current_llm_settings.backlog_max_attempts,
            )
            backlog_pending_total = sum(
                1 for u in run_backlog_updates
                if u.status in (
                    BacklogStatus.PENDING_TRIAGE_BUDGET.value,
                    BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value,
                    BacklogStatus.PENDING_RETRY.value,
                )
            )
            backlog_completed_total = sum(1 for u in run_backlog_updates if u.status == BacklogStatus.COMPLETED.value)
            backlog_date_gate_rejected = sum(
                1 for u in backlog_extra_updates if u.status == BacklogStatus.REJECTED_DATE_GATE.value
            )
            backlog_quarantined_total = sum(
                1 for u in backlog_extra_updates if u.status == BacklogStatus.QUARANTINED.value
            )
            backlog_rejected_total = backlog_date_gate_rejected + sum(
                1 for u in run_backlog_updates
                if u.status in (
                    BacklogStatus.REJECTED_TRIAGE_IRRELEVANT.value,
                    BacklogStatus.REJECTED_DEEP_ANALYSIS_IRRELEVANT.value,
                    BacklogStatus.REJECTED_DATE_GATE.value,
                    BacklogStatus.REJECTED_RETRY_EXHAUSTED.value,
                )
            )
            backlog_updates = backlog_extra_updates + run_backlog_updates

            agent_failures_total = (
                triage_metrics["triage_failures"] + deep_metrics["deep_analysis_timeouts"] + deep_metrics["deep_analysis_failures"]
            )
            candidates_analyzed_total = deep_metrics["deep_analysis_attempted"]
            current_candidates_ranked_total = len(gated_pool)
            current_candidates_analyzed_total = deep_metrics["deep_analysis_attempted"]
            current_candidates_skipped_by_budget_total = max(
                0, len(gated_pool) - deep_metrics["deep_analysis_attempted"]
            )

            llm_budget_exhausted = budget_exhausted or triage_metrics["candidates_truncated_before_triage"] > 0
            candidates_left_unprocessed = triage_metrics["candidates_truncated_before_triage"] + unprocessed_after_budget
            total_llm_calls = triage_metrics["triage_batches_attempted"] + deep_metrics["deep_analysis_attempted"]

            if candidates_left_unprocessed > 0:
                diagnostics.append(
                    PipelineDiagnostic(
                        stage="agent_analysis",
                        message=(
                            f"CURRENT LLM budget applied — {candidates_left_unprocessed} candidate(s) "
                            f"not processed this run (triage_batches<={current_llm_settings.max_triage_batches}, "
                            f"max_deep_analysis={current_llm_settings.max_deep_analysis})"
                        ),
                        severity="warning",
                    )
                )

            current_funnel_metrics = {
                "candidates_before_llm": candidates_before_llm,
                "candidates_after_deterministic_gate": len(gated_pool),
                **triage_metrics,
                "deep_analysis_budget": current_llm_settings.max_deep_analysis,
                **deep_metrics,
                "llm_budget_exhausted": llm_budget_exhausted,
                "candidates_left_unprocessed": candidates_left_unprocessed,
                "total_llm_calls": total_llm_calls,
                # Candidate-backlog observability (never a silent drop --
                # every one of `run_backlog_updates` has an explicit,
                # inspectable status/reason; these counts are UNIQUE
                # candidates, distinct from the per-run processing-attempt
                # counts above).
                "backlog_claimed": len(backlog_records_by_id),
                "backlog_resumed_already_triaged": len(backlog_already_triaged),
                "backlog_resumed_pending_triage": len(backlog_never_triaged),
                "candidates_pending_carried_forward": backlog_pending_total,
                "candidates_completed_from_backlog": backlog_completed_total,
                "candidates_permanently_rejected": backlog_rejected_total,
                "candidates_quarantined": backlog_quarantined_total,
            }
            logger.info(
                "agent_analysis(CURRENT funnel): before_llm=%d after_gate=%d triage_batches=%d deep_attempted=%d "
                "deep_success=%d budget_exhausted=%s unprocessed=%d total_llm_calls=%d",
                candidates_before_llm, len(gated_pool), triage_metrics["triage_batches_attempted"],
                deep_metrics["deep_analysis_attempted"], deep_metrics["deep_analysis_success"],
                llm_budget_exhausted, candidates_left_unprocessed, total_llm_calls,
            )

            link_items, link_grounded_total = _run_current_link_grounded(
                routing, evidence_by_article_id, agents_settings, services.config, services.agents.keys(),
            )
            research_items.extend(link_items)
        else:
            # ---- Morning/Evening: existing per-agent loop, UNCHANGED ----
            for agent_name, agent in services.agents.items():
                settings = getattr(agents_settings, agent_name.value)
                if not settings.enabled:
                    logger.info("agent_analysis: %s is disabled, skipping", agent_name.value)
                    continue

                routed_articles = routing.by_agent.get(agent_name, [])
                candidates = [
                    (article, evidence_by_article_id[article.id])
                    for article in routed_articles
                    if article.id in evidence_by_article_id
                ]
                candidates_considered_total += len(candidates)
                if candidates:
                    source_counts: Dict[str, int] = {}
                    for article, _ in candidates:
                        source_counts[article.source_name] = source_counts.get(article.source_name, 0) + 1
                    logger.info(
                        "agent_analysis: %s candidate sources: %s",
                        agent_name.value, ", ".join(f"{name}={count}" for name, count in sorted(source_counts.items())),
                    )
                # Deterministic top_k: strongest candidates first (see
                # `_candidate_rank_key` -- substantive body, then topical
                # keyword density, then recency as a tiebreaker), so a limited
                # top_k analyzes the strongest few developments rather than
                # just the first/most-recent few pages discovered. A missing
                # date still sorts last among otherwise-equal candidates, never
                # first (never silently prioritized).
                candidates.sort(
                    key=lambda pair: _candidate_rank_key(pair[0], pair[1], topic_keywords, event_type_phrases),
                    reverse=True,
                )
                top_k = settings.top_k if settings.top_k is not None else len(candidates)
                selected = candidates[:top_k]
                candidates_analyzed_total += len(selected)
                if len(candidates) > len(selected):
                    diagnostics.append(
                        PipelineDiagnostic(
                            stage="agent_analysis",
                            message=(
                                f"{agent_name.value}: top_k={top_k} applied — "
                                f"{len(candidates) - len(selected)} candidate(s) not processed this run"
                            ),
                            severity="warning",
                        )
                    )

                agent_failures = 0
                agent_items = 0
                for article, evidence in selected:
                    try:
                        item = agent.analyze(article, evidence)
                    except Exception as exc:  # one article's agent call must never abort the whole run
                        agent_failures += 1
                        diagnostics.append(
                            PipelineDiagnostic(stage="agent_analysis", message=f"{agent_name.value}:{article.id}: {exc}")
                        )
                        logger.warning("agent_analysis: %s failed on %s: %s", agent_name.value, article.id, exc)
                        continue
                    if item is not None:
                        agent_items += 1
                        research_items.append(item)

                agent_failures_total += agent_failures
                logger.info(
                    "agent_analysis: %s candidates=%d analyzed=%d items=%d failures=%d",
                    agent_name.value, len(candidates), len(selected), agent_items, agent_failures,
                )

                # Link-grounded path (content vs. link-grounded intelligence):
                # a routed article with NO Evidence (robots/anti-bot/HTTP/
                # browser content-retrieval failure -- see
                # retrieval.service.RetrievalService.extract's
                # content_unavailable_reason) never reaches the LLM candidate
                # pool above (it has no evidence to analyze), but is still a
                # genuine, legitimately discovered article. Deterministically
                # (no Ollama call, never gated by top_k/budget -- this costs no
                # LLM time) check whether its title/metadata alone establish a
                # substantive, relevant development; if so it becomes a
                # LINK-grounded ResearchItem instead of being silently dropped.
                # See agents.link_grounded for the full truthfulness guardrails.
                link_grounded_candidates = [a for a in routed_articles if a.id not in evidence_by_article_id]
                for article in link_grounded_candidates:
                    link_item = build_link_grounded_research_item(article, config=services.config, agent=agent_name)
                    if link_item is not None:
                        research_items.append(link_item)
                        link_grounded_total += 1

        relevant = sum(1 for item in research_items if item.relevance.label.value == "relevant")
        ambiguous = sum(1 for item in research_items if item.relevance.label.value == "ambiguous")
        # By elimination: analyzed candidates that produced neither a
        # ResearchItem nor a recorded failure were judged irrelevant by the
        # agent (or rejected by its cheap keyword pre-filter before ever
        # reaching the LLM) — this pipeline layer cannot cheaply distinguish
        # the two without changing the stable Phase 3/4 agent contract, so
        # they are reported together (see README/known limitations).
        # link_grounded_total is subtracted here since those items were
        # never part of candidates_analyzed_total (they never reached the
        # LLM at all) -- counting them in `research_items` without
        # excluding them here would make `not_produced` wrong.
        content_grounded_total = len(research_items) - link_grounded_total
        not_produced = candidates_analyzed_total - content_grounded_total - agent_failures_total

        logger.info(
            "agent_analysis: candidates_considered=%d candidates_analyzed=%d relevant=%d ambiguous=%d "
            "not_produced=%d failures=%d link_grounded=%d",
            candidates_considered_total, candidates_analyzed_total, relevant, ambiguous,
            not_produced, agent_failures_total, link_grounded_total,
        )
        metrics = {
            **state.metrics,
            **routing_metrics,
            **current_date_gate_metrics,
            "current_period_candidates": len(period_eligible_articles),
            "link_grounded_items": link_grounded_total,
            "content_grounded_items": content_grounded_total,
            "articles_excluded_from_digest_period": len(out_of_digest_period),
            "agent_candidates_considered": candidates_considered_total,
            "agent_candidates_analyzed": candidates_analyzed_total,
            "research_items_relevant": relevant,
            "research_items_ambiguous": ambiguous,
            "research_items_not_produced": max(0, not_produced),
            "agent_failures": agent_failures_total,
            "research_items_total": len(research_items),
        }

        if state.digest_type == DigestType.CURRENT:
            # Part B7: so an empty CURRENT digest can be told apart from "the
            # system never had a fair chance to find anything" -- these
            # never affect Morning/Evening metrics.
            exact_date_articles = sum(
                1 for a in digest_eligible_articles
                if a.publication_date_precision in (DatePrecision.EXACT, DatePrecision.DATE_ONLY)
            )
            undated_articles = sum(
                1 for a in digest_eligible_articles
                if a.publication_date_precision in (DatePrecision.MONTH, DatePrecision.UNKNOWN)
            )
            events_excluded_missing_date = sum(1 for item in research_items if item.publication_date is None)
            metrics.update({
                "current_articles_in_window": len(digest_eligible_articles),
                "current_exact_date_articles": exact_date_articles,
                "current_undated_articles": undated_articles,
                "current_ranked_candidates": current_candidates_ranked_total,
                "current_candidates_analyzed": current_candidates_analyzed_total,
                "current_candidates_skipped_by_budget": current_candidates_skipped_by_budget_total,
                "current_relevant": relevant,
                "current_research_items_excluded_missing_date": events_excluded_missing_date,
                **current_funnel_metrics,
            })
            logger.info(
                "agent_analysis(CURRENT): articles_in_window=%d exact_date=%d undated=%d "
                "ranked_candidates=%d analyzed=%d skipped_by_budget=%d relevant=%d excluded_missing_date=%d",
                len(digest_eligible_articles), exact_date_articles, undated_articles,
                current_candidates_ranked_total, current_candidates_analyzed_total,
                current_candidates_skipped_by_budget_total, relevant, events_excluded_missing_date,
            )

        if state.run_id is not None:
            research_items = [item.model_copy(update={"run_id": state.run_id}) for item in research_items]

        return {
            "research_items": research_items,
            "diagnostics": state.diagnostics + diagnostics,
            "metrics": metrics,
            "backlog_updates": backlog_updates,
        }

    return agent_analysis


def make_event_scoring_node(services: PipelineServices):
    def event_scoring(state: PipelineState) -> dict:
        events = services.event_service.build_events(state.research_items)
        if state.run_id is not None:
            events = [event.model_copy(update={"run_id": state.run_id}) for event in events]
        logger.info("event_scoring: %d research item(s) -> %d event(s)", len(state.research_items), len(events))
        metrics = {**state.metrics, "events_formed": len(events)}
        if state.digest_type == DigestType.CURRENT:
            metrics["current_events_formed"] = len(events)
        return {"events": events, "metrics": metrics}

    return event_scoring


def make_report_generation_node(services: PipelineServices):
    def report_generation(state: PipelineState) -> dict:
        if not services.config.settings.agents.report.enabled:
            logger.info("report_generation: report agent disabled, skipping")
            return {"reports": [], "report_failures": [], "metrics": {**state.metrics, "reports_succeeded": 0, "reports_failed": 0}}

        items_by_id = {item.id: item for item in state.research_items}
        events_with_items = [
            (event, [items_by_id[rid] for rid in event.research_items if rid in items_by_id])
            for event in state.events
        ]
        # LINK-grounded events (no retrieved article content behind ANY
        # member) get a deterministic, template-based report -- no Ollama
        # call, and no risk of the LLM narrating as if it had read an
        # article body that was never retrieved. Only CONTENT-grounded
        # events (the pre-existing, unchanged path) go to the real
        # ReportAgent/LLM.
        link_grounded_events = [
            (event, items) for event, items in events_with_items if event.grounding_type == GroundingType.LINK
        ]
        content_grounded_events = [
            (event, items) for event, items in events_with_items if event.grounding_type != GroundingType.LINK
        ]
        link_reports = [build_link_grounded_report(event, items) for event, items in link_grounded_events]
        results = services.report_agent.generate_many(content_grounded_events) + link_reports
        reports = [r for r in results if isinstance(r, Report)]
        failures = [r for r in results if isinstance(r, ReportFailure)]
        logger.info("report_generation: requested=%d succeeded=%d failed=%d", len(events_with_items), len(reports), len(failures))
        return {
            "reports": reports,
            "report_failures": failures,
            "metrics": {**state.metrics, "reports_succeeded": len(reports), "reports_failed": len(failures)},
        }

    return report_generation


def make_digest_generation_node(services: PipelineServices):
    def digest_generation(state: PipelineState) -> dict:
        period_end = state.period_end or services.clock.now()
        period_start = state.period_start or (period_end - timedelta(hours=12))

        # Digest-period filtering (Phase 9, tightened in Phase 13D): a
        # final, explicit correctness backstop -- only an event whose own
        # event_date falls inside this digest's period may appear in the
        # digest. In practice agent_analysis's own candidate-selection
        # narrowing already guarantees this, since every constituent
        # research item was already in-period; this exists so the
        # invariant ("an article outside the digest window never appears
        # in that digest merely because it's retained historically") holds
        # explicitly and independently of how upstream filtering was
        # computed. This does not affect which events get persisted as
        # standalone rows -- only digest.events.
        #
        # Phase 13D: a NULL event_date (an evergreen/reference development
        # with no corroborated dated action -- see the date-grounding
        # guardrail in agents.base) is no longer treated as "always
        # in-period". A date-bounded operational digest must never present
        # an undated development as if it fell inside a specific calendar
        # window; it remains fully visible elsewhere (entity pages, event
        # detail, the research database) via the existing, unfiltered
        # Event/ResearchItem queries, which are deliberately unaffected by
        # this change.
        # Phase 13F: half-open interval [period_start, period_end) -- an
        # event dated exactly at a digest's own end boundary (e.g. 08:00:00
        # IST for Morning) belongs to the NEXT digest (Evening), never both.
        in_period_events = [e for e in state.events if e.event_date is not None and period_start <= e.event_date < period_end]
        excluded_from_digest = len(state.events) - len(in_period_events)
        if excluded_from_digest:
            logger.info(
                "digest_generation: excluded %d event(s) whose event_date falls outside this digest's period",
                excluded_from_digest,
            )

        digest = services.digest_service.build_digest(
            in_period_events, digest_type=state.digest_type, period_start=period_start, period_end=period_end
        )
        logger.info(
            "digest_generation: type=%s events=%d themes=%d period=%s..%s",
            digest.digest_type.value, len(digest.events), len(digest.themes), period_start.isoformat(), period_end.isoformat(),
        )
        return {
            "digest": digest,
            "metrics": {
                **state.metrics,
                "digest_event_count": len(digest.events),
                "digest_theme_count": len(digest.themes),
                "events_excluded_from_digest_period": excluded_from_digest,
            },
        }

    return digest_generation


def make_persistence_node(services: PipelineServices):
    """Persists this run's output via `PersistenceService` (Phase 7). A
    failure here is recorded as a diagnostic, never raised — a persistence
    problem must not make the run appear to have produced nothing; the
    computed Events/Reports/Digest are still returned in state either way.
    Persistence is entirely optional: `services.persistence_service is None`
    (the default) means this node is a no-op, preserving Phase 6 behavior
    for callers that don't want SQLite involved at all."""

    def persistence(state: PipelineState) -> dict:
        if services.persistence_service is None:
            return {}

        diagnostics: List[PipelineDiagnostic] = []
        try:
            services.persistence_service.persist_pipeline_result(
                articles=state.articles,
                evidence=state.evidence,
                research_items=state.research_items,
                events=state.events,
                reports=state.reports,
                report_failures=state.report_failures,
                digest=state.digest,
                backlog_updates=state.backlog_updates,
                run_id=state.run_id,
            )
        except Exception as exc:
            diagnostics.append(PipelineDiagnostic(stage="persistence", message=str(exc)))
            logger.error("persistence: failed to persist pipeline result: %s", exc)
        else:
            logger.info(
                "persistence: articles=%d evidence=%d research_items=%d events=%d reports=%d digest=%s",
                len(state.articles), len(state.evidence), len(state.research_items), len(state.events),
                len(state.reports), "yes" if state.digest else "no",
            )

        return {"diagnostics": state.diagnostics + diagnostics}

    return persistence
