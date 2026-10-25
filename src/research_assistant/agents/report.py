"""ReportAgent — Event + its ResearchItems -> a structured Report.

Deliberately NOT a `ResearchAgent` subclass: it operates on one Event at a
time (never a batch of unrelated events, for provenance/context-size/
testability, per the Phase 6 brief) rather than one Article/Evidence pair,
and produces a `Report`, not a `ResearchItem`. It shares the same
`LLMClient` abstraction and prompt-loading convention as every other agent,
and never touches `research_assistant.retrieval` — it only ever reasons
over Event/ResearchItem objects the caller already produced.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from research_assistant.agents.prompt_loader import load_prompt
from research_assistant.agents.schemas import ReportNarrative, TierBReportNarrative
from research_assistant.events.models import Event, SourceRef
from research_assistant.llm.client import LLMClient
from research_assistant.models.enums import EntityType, SourcePriority, SourceType
from research_assistant.models.report import (
    Report,
    ReportDetails,
    ReportFailure,
    ReportSource,
    TierBReport,
    TierBReportFailure,
)
from research_assistant.models.research_item import ResearchItem

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_FILE = "report_system.md"
_TIER_B_SYSTEM_PROMPT_FILE = "report_tier_b_system.md"

# Bounded safety margin on the raw article text sent to the model -- a
# latency/context-size bound only, never a sufficiency judgment (that is
# `reporting.tier_b_reports`'s job, decided BEFORE this agent is ever
# called). Chosen generously above a typical press release's length so a
# normal article is never truncated in practice; only an unusually long
# source is bounded, with the truncation itself stated honestly in the
# prompt rather than silently hidden.
_TIER_B_MAX_SOURCE_TEXT_CHARS = 6000

# ---------------------------------------------------------------------------
# Deterministic report-grounding guardrail (Phase 13C): the ReportAgent's own
# prompt instructions are secondary protection only -- this is the final
# safety boundary before a report is persisted. A report sentence asserting
# a partnership/collaboration survives ONLY when every company-like name it
# mentions is in the deterministic, validated `partnerships` fact list (see
# `agents.base._validate_partnerships_against_evidence`, which already
# narrowed `ResearchItem.partnerships` against the underlying evidence
# text) -- never because the LLM's free-text narrative asserts one.
# ---------------------------------------------------------------------------

# Confirmed regression (real Gemtek/Qualcomm/gFiber event, 2026-10-15): a
# bare `(?<=[.!?])\s+` treats the period in "...Qualcomm Technologies,
# Inc., and gFiber Inc. announced..." as a sentence boundary (the comma
# after "Co."/"Ltd." happens to avoid a false split there, but nothing
# stops the final "Inc." immediately before the verb from splitting) --
# the real subject-list fragment ("...gFiber Inc.") then survives as its
# own "sentence" (no partnership-language match), while the real verb
# clause ("announced a collaboration...") becomes a separate sentence
# that DOES match `_REPORT_PARTNERSHIP_LANGUAGE_RE` and gets dropped,
# leaving a dangling, verb-less fragment as the entire field. Requiring
# the character after the whitespace to look like a genuine new
# sentence's start (uppercase letter, digit, or opening quote) fixes
# this generically for any abbreviation ("Inc.", "Ltd.", "Co.", "Corp.",
# etc.) immediately followed by a lowercase verb, without needing a
# bespoke abbreviation whitelist, and does not change how real
# multi-sentence text splits (every genuine sentence boundary in this
# agent's LLM-generated narrative text starts with a capital letter).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_REPORT_PARTNERSHIP_LANGUAGE_RE = re.compile(
    r"\bpartner(?:s|ed|ship|ships|ing)?\b|\bcollaborat(?:ion|ed|ing)?\b"
    r"|\bjoint\s+venture\b|\bstrategic\s+alliance\b",
    re.IGNORECASE,
)

# Phase 13F "REPORT QUALITY": unsupported, inflated editorializing phrases
# the LLM sometimes adds despite the prompt's grounding instructions --
# ("key player", "leading developer", etc. are never themselves facts
# extracted from evidence; see `_aggregate_facts`, which has no field for
# "market stature"). Deliberately a surgical PHRASE removal, not a
# sentence-drop: the task explicitly warns "Do not use a blind blacklist
# that damages factual prose" -- a sentence like "NVIDIA, a key player in
# AI infrastructure, today announced..." has a real, evidence-grounded fact
# in it (the announcement) alongside the unsupported editorializing clause;
# dropping the whole sentence (as the partnership guardrail does, where the
# ENTIRE sentence's claim is unsupported) would destroy that fact too.
_INFLATED_PHRASES = (
    "key player", "leading developer", "leading provider", "leading player",
    "major market opportunity", "increasingly critical", "increasingly important",
    "strategic investment opportunity", "game changer", "game-changer",
    "industry leader", "cutting-edge", "cutting edge", "world-class", "world class",
    "best-in-class", "best in class", "pioneering", "groundbreaking", "revolutionary",
)
# Confirmed production regression (2026-10-15, real NVIDIA/France event):
# the clause-removal alternative below originally required the phrase to
# be followed specifically by "in" (e.g. "a leading provider IN AI"), so a
# real, very common phrasing -- "a leading provider OF AI..." -- never
# matched it and fell through to the bare-word alternative instead, which
# strips only "leading provider" and leaves a grammatically broken
# dangling article: "NVIDIA, a of AI and high-performance computing
# solutions, announced...". Fixed by not requiring any specific
# preposition -- the clause alternative now removes everything from the
# article through the next comma, regardless of what connects the phrase
# to its object ("of", "in", "for", or nothing at all).
_INFLATED_PHRASE_RE = re.compile(
    r"(?:,\s*(?:a|an|the)\s+(?:" + "|".join(re.escape(p) for p in _INFLATED_PHRASES) + r")\b[^,.;]*,)"
    r"|\b(?:" + "|".join(re.escape(p) for p in _INFLATED_PHRASES) + r")\b",
    re.IGNORECASE,
)
_WHITESPACE_CLEANUP_RE = re.compile(r"\s{2,}")
_DANGLING_PUNCTUATION_RE = re.compile(r"\s+([,.;:])")
# Confirmed production regression #2 (2026-10-15, real NVIDIA/France
# regeneration test run): the comma-bounded clause alternative above only
# fires when the inflated phrase sits inside a comma-delimited appositive
# ("NVIDIA, a leading provider of AI, announced..."). A phrase at the end
# of a clause with NO trailing comma ("...which is a leading player in
# the AGENTIC computing space.") never matches that alternative and falls
# to the bare-word alternative instead, which strips only the phrase
# itself and leaves the SAME class of dangling-article defect as bug #1,
# just without a following comma: "...which is a in the AGENTIC computing
# space." General fix (not phrase-specific, not punctuation-pattern-
# specific): after phrase removal, any leftover "a"/"an"/"the" directly
# followed by a preposition (nothing noun-like survived in between) is
# itself a leftover fragment of a removed phrase and is dropped too. The
# trailing `\s` requirement (not a bare `\b`) deliberately excludes a
# hyphenated compound like "a for-profit company", where "for" is a real
# word of the noun phrase, not a dangling preposition.
_DANGLING_ARTICLE_BEFORE_PREPOSITION_RE = re.compile(
    r"\b(?:a|an|the)\s+(?=(?:in|of|for|with|on|at|to)\s)", re.IGNORECASE,
)


def _strip_inflated_language(text: str) -> str:
    """Removes unsupported editorializing phrases in place, preserving the
    surrounding factual content of the sentence (see module note above)."""
    if not text:
        return text
    stripped = _INFLATED_PHRASE_RE.sub("", text)
    stripped = _DANGLING_ARTICLE_BEFORE_PREPOSITION_RE.sub("", stripped)
    stripped = _DANGLING_PUNCTUATION_RE.sub(r"\1", stripped)
    stripped = _WHITESPACE_CLEANUP_RE.sub(" ", stripped)
    return stripped.strip()


_FALLBACK_TEXT = {
    "opening_context": "The evidence establishes limited detail beyond the structured facts above.",
    "details_narrative": "The evidence establishes limited detail beyond the structured facts above.",
    "strategic_implications": "The evidence does not support a specific strategic interpretation beyond the stated facts.",
    "nomura_opportunity": "This development may warrant monitoring as more detail becomes available.",
}


def _split_sentences(text: str) -> List[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]


# Confirmed production regression #3 (2026-10-15, real NVIDIA/France event,
# zero validated partnerships): this field's entire LLM-generated text is
# routinely just ONE sentence, and for an article literally framed around a
# multi-company agreement, that one sentence is near-guaranteed to use
# partnership/collaboration language -- so dropping "the unsupported
# sentence" is, in practice, dropping the WHOLE field every time, even
# though real, validated, non-partnership facts (products/technologies/
# geography from `_aggregate_facts`) exist and could ground a genuinely
# specific fallback instead of the bare, context-free hedge. Also: when
# EVERY item in `key_insights` independently trips this same rule (a
# distinct insight per unvalidated company, in this real case), each item
# collapsed to the exact same literal fallback string, producing visibly
# duplicated boilerplate lines. Fixed two ways below: (1) a fact-grounded
# fallback (`_fallback_with_facts`) is used instead of the bare generic
# text whenever `facts` has real structured content to cite; (2) list
# items that empty out are DROPPED, never individually replaced with a
# repeated fallback -- at most ONE fallback line is ever added, only if
# the whole list would otherwise be empty.
def _sanitize_partnership_claims(
    text: str, *, validated_partners: List[str], known_companies: List[str], field: str,
    fallback: Optional[str] = None, allow_empty: bool = False,
) -> str:
    """Drop any sentence that frames a relationship as a partnership/
    collaboration unless every company-like name it mentions is a
    validated partner. With no validated partners at all, EVERY
    partnership-framed sentence is dropped outright -- the two proven
    high-impact failures (an entirely unsupported partnership claim, and a
    named-but-unvalidated extra "partner") are both covered by this single
    rule. Unless `allow_empty`, never leaves a field empty (schema requires
    non-empty text): `fallback` (or the generic default) is used if
    sanitization would otherwise empty it out."""
    if not text:
        return text
    validated_lower = {p.lower() for p in validated_partners}
    kept = []
    for sentence in _split_sentences(text):
        if _REPORT_PARTNERSHIP_LANGUAGE_RE.search(sentence):
            lowered = sentence.lower()
            unsupported_company_named = any(
                company.lower() in lowered and company.lower() not in validated_lower
                for company in known_companies
            )
            if not validated_partners or unsupported_company_named:
                logger.info(
                    "report grounding guardrail: dropped unsupported partnership sentence from %s: %r",
                    field, sentence,
                )
                continue
        kept.append(sentence)
    result = " ".join(kept).strip()
    if result:
        return result
    if allow_empty:
        return ""
    return fallback if fallback is not None else _FALLBACK_TEXT[field]


# Facts deliberately excluded from this clause: "companies"/"other_companies"
# (naming one could read as implying an unvalidated relationship with the
# primary company -- the exact risk the partnership guardrail exists to
# avoid) and "partnerships" (by definition unvalidated here, or this
# fallback path would never have been reached). Only genuinely
# relationship-free structured facts are cited.
_GROUNDABLE_FACT_KEYS = ("products", "technologies", "geography")


def _grounded_fallback_clause(facts: Dict) -> Optional[str]:
    """A short, deterministic, comma-joined clause citing real structured
    facts (never LLM prose, never partnerships/companies) -- or `None` if
    `facts` has nothing citable. Bounded to a handful of items total so a
    long entity extraction can't produce an unbounded fallback sentence."""
    bits: List[str] = []
    for key in _GROUNDABLE_FACT_KEYS:
        values = facts.get(key) or []
        bits.extend(values[:3])
        if len(bits) >= 4:
            break
    if facts.get("deal_value") and len(bits) < 4:
        bits.append(str(facts["deal_value"]))
    if not bits:
        return None
    return ", ".join(bits[:4])


_FACT_GROUNDED_FALLBACK_TEMPLATES = {
    "opening_context": "The confirmed facts include {clause}; the evidence establishes limited additional detail beyond that.",
    "details_narrative": "The confirmed facts include {clause}; the evidence establishes limited additional detail beyond that.",
    "strategic_implications": "Based on the confirmed {clause}, the evidence does not support a more specific strategic interpretation beyond those facts.",
    "nomura_opportunity": "Based on the confirmed {clause}, no specific Nomura-relevant opportunity is established beyond that; this may warrant monitoring as more detail becomes available.",
}


def _fallback_with_facts(field: str, facts: Dict) -> str:
    """Prefers a fact-grounded fallback sentence over the bare, generic
    hedge whenever `facts` has real citable content -- see
    `_grounded_fallback_clause`. Falls back to the original generic text
    when `facts` is empty/uncitable (e.g. Tier B, which has no structured
    facts at all) -- unchanged, maximally conservative behavior there."""
    clause = _grounded_fallback_clause(facts)
    if clause is None:
        return _FALLBACK_TEXT[field]
    return _FACT_GROUNDED_FALLBACK_TEMPLATES[field].format(clause=clause)


_TIER_B_CONTEXT_GROUNDED_FALLBACK_TEMPLATES = {
    "strategic_implications": (
        "Based on the development described above, the evidence does not support a more specific "
        "strategic interpretation beyond that."
    ),
    "nomura_opportunity": (
        "Based on the development described above, no specific Nomura-relevant opportunity is "
        "established beyond that; this may warrant monitoring as more detail becomes available."
    ),
}


def _tier_b_fallback_with_context(field: str, sanitized_opening_context: str) -> str:
    """Tier B has no structured `facts` dict to cite (no ResearchItem/Event
    was ever validated) -- `_fallback_with_facts` would always return the
    bare generic hedge here. Instead, when the ALREADY-SANITIZED
    `opening_context` (itself grounded only in the article text, and
    already passed through the same partnership/inflated-language
    guardrails) is substantive, point back to it rather than a
    context-free hedge -- this cites no new fact, just references content
    that already survived grounding. Falls back to the original generic
    text when `opening_context` itself is empty or is itself the bare
    fallback (nothing substantive to point back to)."""
    if not sanitized_opening_context or sanitized_opening_context == _FALLBACK_TEXT["opening_context"]:
        return _FALLBACK_TEXT[field]
    return _TIER_B_CONTEXT_GROUNDED_FALLBACK_TEMPLATES[field]


@dataclass(frozen=True)
class TierBGenerationInput:
    """Everything `ReportAgent.generate_tier_b` needs for ONE Tier B
    article -- deliberately a bare dataclass, not a persisted model: the
    caller (`reporting.tier_b_reports`) has already decided sufficiency
    and resolved the effective text BEFORE constructing this, so this
    agent never has to re-derive either."""

    article_id: str
    title: Optional[str]
    publisher: Optional[str]
    publication_date: Optional[datetime]
    url: str
    source_type: SourceType
    source_priority: SourcePriority
    text: str
    content_partial: bool


class ReportAgent:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._system_prompt = load_prompt(_SYSTEM_PROMPT_FILE)
        self._tier_b_system_prompt = load_prompt(_TIER_B_SYSTEM_PROMPT_FILE)

    def generate(self, event: Event, research_items: List[ResearchItem]) -> Report:
        """Raises `research_assistant.llm.LLMError` on failure — the caller
        decides whether to record a `ReportFailure` and continue (see
        `generate_many`, or the LangGraph report_generation node)."""
        facts = _aggregate_facts(research_items)
        prompt = _build_prompt(event, facts)
        narrative = self._llm.generate_structured(system=self._system_prompt, prompt=prompt, schema=ReportNarrative)
        narrative = _sanitize_narrative(narrative, facts)
        return _build_report(event, research_items, facts, narrative)

    def generate_many(
        self, events_with_items: List[Tuple[Event, List[ResearchItem]]]
    ) -> List[Union[Report, ReportFailure]]:
        """One Event at a time; a single event's LLM failure never blocks the rest."""
        from research_assistant.llm.errors import LLMError

        results: List[Union[Report, ReportFailure]] = []
        for event, items in events_with_items:
            try:
                results.append(self.generate(event, items))
            except LLMError as exc:
                logger.warning("Report generation failed for event %s: %s", event.id, exc)
                results.append(ReportFailure(event_id=event.id, reason=type(exc).__name__, detail=str(exc)))
        return results

    def generate_tier_b(self, item: TierBGenerationInput) -> TierBReport:
        """Raises `research_assistant.llm.LLMError` on failure -- the
        caller (`generate_tier_b_many`, or the newsletter report resolver)
        decides whether to record a `TierBReportFailure` and continue.
        Sufficiency and effective-text selection are the CALLER's
        responsibility (`reporting.tier_b_reports`) -- this method never
        re-fetches, re-extracts, or judges whether `item.text` was enough;
        it only ever reasons over the text it is given, exactly once."""
        bounded_text = _bound_tier_b_source_text(item.text)
        prompt = _build_tier_b_prompt(item, bounded_text)
        narrative = self._llm.generate_structured(
            system=self._tier_b_system_prompt, prompt=prompt, schema=TierBReportNarrative,
        )
        narrative = _sanitize_tier_b_narrative(narrative)
        return TierBReport(
            article_id=item.article_id,
            title=item.title,
            opening_context=narrative.opening_context,
            key_insights=narrative.key_insights,
            strategic_implications=narrative.strategic_implications,
            nomura_opportunity=narrative.nomura_opportunity,
            source=ReportSource(
                publisher=item.publisher, title=item.title, publication_date=item.publication_date,
                url=item.url, source_type=item.source_type, source_priority=item.source_priority,
            ),
            content_partial=item.content_partial,
        )

    def generate_tier_b_many(
        self, items: List[TierBGenerationInput]
    ) -> List[Union[TierBReport, TierBReportFailure]]:
        """One article at a time, strictly sequentially (no model-call
        concurrency) -- a single article's LLM failure never blocks the
        rest, exactly like `generate_many`."""
        from research_assistant.llm.errors import LLMError

        results: List[Union[TierBReport, TierBReportFailure]] = []
        for item in items:
            try:
                results.append(self.generate_tier_b(item))
            except LLMError as exc:
                logger.warning("Tier B report generation failed for article %s: %s", item.article_id, exc)
                results.append(TierBReportFailure(article_id=item.article_id, reason=type(exc).__name__, detail=str(exc)))
        return results


_COMPANY_ENTITY_TYPES = {EntityType.COMPANY, EntityType.ORGANIZATION}
_PRODUCT_ENTITY_TYPES = {EntityType.PRODUCT, EntityType.PLATFORM, EntityType.MODEL}


def _aggregate_facts(items: List[ResearchItem]) -> Dict:
    """Deterministic, Python-only aggregation — no LLM involved. Sorted by
    item id first so the result never depends on input order.

    Phase 13C: typed `entities` (company/organization -> companies,
    product/platform/model -> products, technology -> technologies) are
    folded in alongside the pre-existing `item.company`/`.products`/
    `.technologies` fields -- addressing the Phase 13B finding that this
    function never read `entities` at all, which left reports with
    near-empty facts whenever the agent put real detail there instead of
    the older fields. This never touches `partnerships` -- entities are
    purely descriptive facts, never a source of a new partnership claim."""
    ordered = sorted(items, key=lambda item: item.id)
    primary_companies = {item.company for item in ordered}
    entity_companies = set()
    products = {p for item in ordered for p in item.products}
    technologies = {t for item in ordered for t in item.technologies}
    for item in ordered:
        for entity in item.entities:
            if entity.type in _COMPANY_ENTITY_TYPES:
                entity_companies.add(entity.name)
            elif entity.type in _PRODUCT_ENTITY_TYPES:
                products.add(entity.name)
            elif entity.type == EntityType.TECHNOLOGY:
                technologies.add(entity.name)
    people = sorted({p for item in ordered for p in item.people})
    partnerships = sorted({p for item in ordered for p in item.partnerships})
    geography = sorted({g for item in ordered for g in item.geography})
    deal_values = [item.deal_value for item in ordered if item.deal_value]
    return {
        "companies": sorted(primary_companies | entity_companies),
        # Entity-derived companies ONLY (excludes the primary monitored
        # company/companies, e.g. "NVIDIA") -- the set a report sentence's
        # partnership claim is checked against for an unsupported *other*
        # party. The primary company is the actor making an announcement,
        # never itself "the partner".
        "other_companies": sorted(entity_companies - primary_companies),
        "people": people,
        "products": sorted(products),
        "technologies": sorted(technologies),
        "partnerships": partnerships,
        "geography": geography,
        "deal_value": deal_values[0] if deal_values else None,
    }


def _sanitize_field(
    text: str, *, validated_partners: List[str], known_companies: List[str], field: str,
    fallback: Optional[str] = None,
) -> str:
    """Phrase-level inflated-language stripping runs first (never empties a
    field on its own -- it only ever removes a clause/phrase), THEN the
    existing sentence-level partnership guardrail (Phase 13C), which can
    drop whole sentences and already has its own fallback-text handling.
    `fallback`, when given, is a fact-grounded sentence (see
    `_fallback_with_facts`) preferred over the generic default."""
    de_inflated = _strip_inflated_language(text)
    return _sanitize_partnership_claims(
        de_inflated, validated_partners=validated_partners, known_companies=known_companies, field=field,
        fallback=fallback,
    )


def _sanitize_insight_item(text: str, *, validated_partners: List[str], known_companies: List[str]) -> str:
    """Same guardrails as `_sanitize_field`, but returns "" (never a
    fallback sentence) when nothing survives -- callers must drop empty
    items rather than keep one, so that N independently-unsupported
    insights never turn into N copies of the identical fallback line (see
    `_sanitize_narrative`'s `key_insights` handling)."""
    de_inflated = _strip_inflated_language(text)
    return _sanitize_partnership_claims(
        de_inflated, validated_partners=validated_partners, known_companies=known_companies,
        field="details_narrative", allow_empty=True,
    )


def _sanitize_key_insights(
    items: List[str], *, validated_partners: List[str], known_companies: List[str], facts: Dict,
) -> List[str]:
    """Drops (never individually replaces) any insight that the guardrail
    empties out, then adds AT MOST ONE fact-grounded fallback item if that
    leaves the list empty -- never one repeated fallback line per dropped
    item."""
    survivors = [
        sanitized
        for item in items
        if (sanitized := _sanitize_insight_item(
            item, validated_partners=validated_partners, known_companies=known_companies,
        ))
    ]
    if survivors:
        return survivors
    return [_fallback_with_facts("details_narrative", facts)]


def _reuse_sibling_on_fallback(own_sanitized: str, own_fallback: str, sibling_sanitized: str, sibling_fallback: str) -> str:
    """When `own_sanitized` fell all the way back to its own context-free
    hedge (every sentence of the LLM's real answer was partnership-framed
    with no validated partner -- see `_sanitize_partnership_claims`), but a
    SIBLING field's independently-sanitized text survived with real
    substance (did not itself fall back), surface that already-vetted text
    instead of a second, unrelated hedge. Real production finding
    (2026-10-19, HKU/GCSA centre announcement): `opening_context`'s own
    sentence used "jointly establish..." (partnership-framed, dropped
    outright since this item had no `partnerships_json` entry), while
    `details_narrative` described the exact same announcement with
    non-partnership-framed phrasing ("...jointly announced the
    establishment of...") and survived -- yet the UI only ever showed
    `opening_context`'s empty hedge, discarding the real, already-safe
    content sitting right next to it. Never invents anything new: only
    reuses text that already passed the identical guardrails for this same
    event."""
    if own_sanitized != own_fallback:
        return own_sanitized
    if sibling_sanitized and sibling_sanitized != sibling_fallback:
        return sibling_sanitized
    return own_sanitized


def _sanitize_narrative(narrative: ReportNarrative, facts: Dict) -> ReportNarrative:
    validated_partners = facts["partnerships"]
    known_companies = facts["other_companies"]

    details_fallback = _fallback_with_facts("details_narrative", facts)
    sanitized_details_narrative = _sanitize_field(
        narrative.details_narrative, validated_partners=validated_partners,
        known_companies=known_companies, field="details_narrative", fallback=details_fallback,
    )
    sanitized_key_insights = _sanitize_key_insights(
        narrative.key_insights, validated_partners=validated_partners,
        known_companies=known_companies, facts=facts,
    )

    opening_context_fallback = _fallback_with_facts("opening_context", facts)
    sanitized_opening_context = _reuse_sibling_on_fallback(
        _sanitize_field(
            narrative.opening_context, validated_partners=validated_partners,
            known_companies=known_companies, field="opening_context", fallback=opening_context_fallback,
        ),
        opening_context_fallback, sanitized_details_narrative, details_fallback,
    )

    # strategic_implications has no single-field sibling describing the
    # same event the way details_narrative/opening_context do, but a
    # surviving key_insight is independently-sanitized, real, forward-
    # looking content -- a far better fallback than a context-free hedge.
    strategic_fallback = _fallback_with_facts("strategic_implications", facts)
    sanitized_strategic_implications = _reuse_sibling_on_fallback(
        _sanitize_field(
            narrative.strategic_implications, validated_partners=validated_partners,
            known_companies=known_companies, field="strategic_implications", fallback=strategic_fallback,
        ),
        strategic_fallback, sanitized_key_insights[0] if sanitized_key_insights else "", details_fallback,
    )

    return narrative.model_copy(
        update={
            "opening_context": sanitized_opening_context,
            "details_narrative": sanitized_details_narrative,
            "key_insights": sanitized_key_insights,
            "strategic_implications": sanitized_strategic_implications,
            "nomura_opportunity": _sanitize_field(
                narrative.nomura_opportunity, validated_partners=validated_partners,
                known_companies=known_companies, field="nomura_opportunity",
                fallback=_fallback_with_facts("nomura_opportunity", facts),
            ),
        }
    )


def _sanitize_tier_b_narrative(narrative: TierBReportNarrative) -> TierBReportNarrative:
    """Reuses the exact same guardrails Tier A's `_sanitize_narrative`
    does. A Tier B article has no deterministically-validated
    partnerships or entities at all (no ResearchItem/Event exists to
    validate them against) -- passing empty lists for both means
    `_sanitize_partnership_claims`'s existing "no validated partners at
    all -> drop EVERY partnership-framed sentence" rule applies
    unconditionally, which is exactly the correct, conservative behavior
    here: nothing was ever validated, so no partnership claim can survive.
    No structured `facts` dict exists for Tier B, so `strategic_implications`/
    `nomura_opportunity` use `_tier_b_fallback_with_context` instead of
    `_fallback_with_facts` -- citing the already-sanitized `opening_context`
    (grounded only in the article text) rather than a bare, context-free
    hedge. `opening_context` is sanitized FIRST so that fallback can
    reference its final value. The `key_insights` duplicate-fallback fix
    (`_sanitize_key_insights`) is shared with Tier A."""
    no_facts: Dict = {}
    sanitized_opening = _sanitize_field(
        narrative.opening_context, validated_partners=[], known_companies=[], field="opening_context",
    )
    return narrative.model_copy(
        update={
            "opening_context": sanitized_opening,
            "key_insights": _sanitize_key_insights(
                narrative.key_insights, validated_partners=[], known_companies=[], facts=no_facts,
            ),
            "strategic_implications": _sanitize_field(
                narrative.strategic_implications, validated_partners=[], known_companies=[], field="strategic_implications",
                fallback=_tier_b_fallback_with_context("strategic_implications", sanitized_opening),
            ),
            "nomura_opportunity": _sanitize_field(
                narrative.nomura_opportunity, validated_partners=[], known_companies=[], field="nomura_opportunity",
                fallback=_tier_b_fallback_with_context("nomura_opportunity", sanitized_opening),
            ),
        }
    )


def _bound_tier_b_source_text(text: str) -> str:
    if len(text) <= _TIER_B_MAX_SOURCE_TEXT_CHARS:
        return text
    truncated = text[:_TIER_B_MAX_SOURCE_TEXT_CHARS].rsplit(" ", 1)[0]
    return truncated + " [...text truncated for length...]"


def _build_tier_b_prompt(item: TierBGenerationInput, bounded_text: str) -> str:
    return (
        f"Article title: {item.title or 'unknown'}\n"
        f"Publisher: {item.publisher or 'unknown'}\n"
        f"Publication date: {item.publication_date.isoformat() if item.publication_date else 'unknown'}\n"
        f"Source URL: {item.url}\n"
        f"Retrieved text completeness: "
        f"{'PARTIAL -- this may not be the full article' if item.content_partial else 'full extraction succeeded'}\n\n"
        "Article text -- this is the ONLY factual material you may reference:\n"
        f"{bounded_text}\n\n"
        "Write the four sections (opening_context, key_insights, strategic_implications, "
        "nomura_opportunity) as a single JSON object matching the required schema, grounded "
        "ONLY in the article text above."
    )


def _build_prompt(event: Event, facts: Dict) -> str:
    def _joined(values: List[str]) -> str:
        return ", ".join(values) if values else "none stated"

    all_sources: List[SourceRef] = [event.primary_source, *event.secondary_sources]
    sources_summary = "\n".join(
        f"- {ref.publisher or 'unknown publisher'} ({ref.source_type.value}): {ref.url}" for ref in all_sources
    )

    return (
        f"Event type: {event.event_type.value if event.event_type else 'unknown'}\n"
        f"Primary company: {event.primary_company}\n"
        f"Related companies: {_joined(event.related_companies)}\n"
        f"Event date: {event.event_date.isoformat() if event.event_date else 'unknown'}\n"
        f"Existing description: {event.description or 'none'}\n\n"
        "Structured facts already extracted from evidence — this is the ONLY "
        "factual material you may reference:\n"
        f"Companies: {_joined(facts['companies'])}\n"
        f"People: {_joined(facts['people'])}\n"
        f"Products: {_joined(facts['products'])}\n"
        f"Technologies: {_joined(facts['technologies'])}\n"
        f"Partnerships: {_joined(facts['partnerships'])}\n"
        f"Deal value: {facts['deal_value'] or 'not disclosed'}\n"
        f"Geography: {_joined(facts['geography'])}\n\n"
        f"Sources ({len(all_sources)}):\n{sources_summary}\n\n"
        "Write the four narrative sections as a single JSON object matching "
        "the required schema, grounded ONLY in the facts above."
    )


def _build_report(event: Event, items: List[ResearchItem], facts: Dict, narrative: ReportNarrative) -> Report:
    items_by_id = {item.id: item for item in items}
    # `other_companies` is an internal-only key (the report-grounding
    # guardrail's input, see `_sanitize_narrative`) -- never part of the
    # persisted `ReportDetails` shape.
    report_facts = {k: v for k, v in facts.items() if k != "other_companies"}
    return Report(
        event_id=event.id,
        title=event.canonical_title,
        opening_context=narrative.opening_context,
        details=ReportDetails(**report_facts, narrative=narrative.details_narrative, key_insights=narrative.key_insights),
        strategic_implications=narrative.strategic_implications,
        nomura_opportunity=narrative.nomura_opportunity,
        sources=_build_sources(event, items_by_id),
        relevance_score=event.relevance_score,
        confidence_score=event.confidence_score,
    )


def _build_sources(event: Event, items_by_id: Dict[str, ResearchItem]) -> List[ReportSource]:
    sources = []
    for ref in [event.primary_source, *event.secondary_sources]:
        item = items_by_id.get(ref.research_item_id)
        sources.append(
            ReportSource(
                publisher=ref.publisher,
                title=item.title if item else None,
                publication_date=item.publication_date if item else None,
                url=ref.url,
                source_type=ref.source_type,
                source_priority=ref.source_priority,
            )
        )
    return sources


def build_link_grounded_report(event: Event, items: List[ResearchItem]) -> Report:
    """Deterministic (no LLM/Ollama call -- unlike `ReportAgent.generate`)
    report for a LINK-grounded Event (`event.grounding_type ==
    GroundingType.LINK`): every item's article body was unavailable, so
    there is no real content to summarize/interpret. Every prose field
    below is a fixed, transparent template -- it states only what is
    actually known (title/source/date/URL/why it surfaced) and explicitly
    says the body was never read, rather than generating any narrative
    that could read as if the system had. `details` facts intentionally
    stay empty (no entities/people/products/technologies were ever
    extracted, since none were invented) -- reusing `_aggregate_facts`/
    `_build_sources` would be misleading here (they'd silently produce
    the same empty/near-empty shape anyway, since nothing was extracted),
    so this builds the few fields directly instead."""
    items_by_id = {item.id: item for item in items}
    sources = _build_sources(event, items_by_id)
    titles = sorted({item.title for item in items if item.title})
    publishers = sorted({item.publisher for item in items if item.publisher})

    opening_context = (
        f"{', '.join(publishers) or 'The source'} published "
        f"{'an article' if len(titles) == 1 else f'{len(titles)} articles'} "
        f"titled {', '.join(f'‘{t}’' for t in titles)}. "
        "This development was identified as relevant from its title and publication metadata; "
        "automated access to the article content was unavailable, so no claim below is based on "
        "reading the article body."
    )
    event_type_label = event.event_type.value.replace("_", " ") if event.event_type else "development"
    details_narrative = (
        f"What is known: source, title, and publication date (see sources below), and that the title "
        f"names a {event_type_label}. What is not known: the article body was inaccessible to automated "
        "retrieval, so no further facts, figures, quotes, or partnership/deal details can be stated. "
        "Open the original article for full details."
    )
    strategic_implications = (
        "Strategic implications cannot be assessed without the article content; this is a link-grounded "
        "development surfaced for awareness, not a content-based analysis."
    )
    nomura_opportunity = (
        "Not assessed -- no article content was available to evaluate a Nomura-specific opportunity."
    )

    return Report(
        event_id=event.id,
        title=event.canonical_title,
        opening_context=opening_context,
        details=ReportDetails(
            companies=[], people=[], products=[], technologies=[], partnerships=[], deal_value=None,
            geography=[], narrative=details_narrative,
        ),
        strategic_implications=strategic_implications,
        nomura_opportunity=nomura_opportunity,
        sources=sources,
        relevance_score=event.relevance_score,
        confidence_score=event.confidence_score,
    )
