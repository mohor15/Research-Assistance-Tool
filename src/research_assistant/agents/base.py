"""Shared reasoning logic for every Phase 3+ agent.

Architecture: Article + Evidence (Phase 2) -> agent -> ResearchItem. An
agent never fetches or scrapes anything itself — it only reasons over
already-retrieved, already-normalized evidence supplied by the caller, and
it never touches `research_assistant.retrieval` (see
`tests/agents/test_base.py::test_agent_modules_do_not_import_retrieval`).
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import ClassVar, Dict, List, Optional, Set, Tuple, Type

from research_assistant.agents.prompt_loader import load_prompt
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.config.schema import AppConfig, TopicsRegistry
from research_assistant.llm.client import LLMClient
from research_assistant.models.article import Article, DatePrecision
from research_assistant.models.enums import AgentName
from research_assistant.models.evidence import Evidence
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    ResearchItem,
    SourceQuality,
)

logger = logging.getLogger(__name__)

_ID_LENGTH = 16

# CPU-only local inference (qwen2.5:3b on this machine) measured prompt-eval
# cost scaling poorly with input size: an isolated benchmark showed an
# ~11k-char article pushing a single call to ~90s, and two same-sized real
# articles exceeded the production LLM timeout entirely. Evidence content is
# the only unbounded, variable-size part of the prompt (system prompt and
# schema are both fixed, ~2.8-2.9k chars). Capping it is a deliberate,
# documented trade-off, not an oversight:
#   - keeps the BEGINNING of the article (news/press-release writing is
#     front-loaded -- the who/what/when is almost always in the opening
#     paragraphs), which is the cheapest way to preserve grounding quality
#     without any content-aware summarization;
#   - never caps title/publisher/date/URL -- only the body;
#   - appends an explicit truncation marker so the model (and anyone
#     auditing the prompt) never mistakes a cut body for a complete one;
#   - the full, untruncated text remains in `Evidence.content` in storage --
#     only the LLM-facing prompt is capped.
MAX_EVIDENCE_CONTENT_CHARS = 4000
_TRUNCATION_MARKER = "\n[... evidence truncated at {limit} characters; remainder omitted ...]"


def _capped_evidence_content(content: str, *, limit: int = MAX_EVIDENCE_CONTENT_CHARS) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + _TRUNCATION_MARKER.format(limit=limit)


# ---------------------------------------------------------------------------
# Deterministic post-LLM guardrails (Phase 13C): qwen2.5:3b does not reliably
# obey prompt-only instructions for two grounding-critical judgments --
# whether a page describes a genuinely DATED development, and whether a
# named company is a confirmed PARTNER -- so both are re-derived here from
# the evidence text itself and can only ever NARROW what the LLM claimed,
# never add something it didn't. Generic, pattern-based, source-agnostic:
# no company/page name appears anywhere in this logic.
# ---------------------------------------------------------------------------

# Phrases that plausibly mark a specific, dated action (an announcement,
# launch, or release) as opposed to evergreen/reference prose describing a
# product or capability in the present tense. Calibrated against real
# retrieved press releases/blog posts in this project (e.g. "Bank of
# America today announced...", "CoreWeave announced availability of...",
# "Today, we're releasing...") vs. a real evergreen product page (plain
# present-tense capability descriptions, no announcement phrasing at all).
# `today released` (end-to-end audit finding): the real PR Newswire
# JumpCloud release ("JumpCloud Inc. today released a new study...") used
# this exact word order, which was missing even though every other verb
# here (announced/launched/unveiled) already covers both word orders
# ("X today" and "today X") -- a real article with a real announced,
# same-day release was silently losing its event_date (see
# `_select_event_date` in events/builder.py) purely because of this one
# missing phrase, never reaching the CURRENT digest despite being
# genuinely relevant, dated, and successfully deep-analyzed.
#
# `has achieved` / `today achieved` / `achieved today` (CURRENT run
# 7aab105954d38497 forensic finding): the real PR Newswire Bits In Glass
# release ("Bits In Glass (BIG) has achieved Select partner status...")
# is a one-time, dated milestone/recognition announcement -- the exact
# same family as "has announced"/"has launched"/"has unveiled" (a
# present-perfect verb reporting a just-completed, specific action), just
# a verb this list didn't yet cover. Qwen itself correctly judged this
# `is_dated_development=true` ("Describes a specific action ... with
# announcement language"), but `_resolve_dated_development` narrowed it
# back to False purely because this verb was missing, silently losing the
# event's date exactly like the JumpCloud "today released" gap above.
# ---------------------------------------------------------------------------
# Generic, data-driven verb-tense generation (2026-10-13 refactor): every
# fix above this point was the same recurring shape -- a real press
# release used a KNOWN announcement verb (announce/launch/unveil/
# introduce) in a tense/collocation this list hadn't been given yet (SWI:
# present-tense "announces"; HR Path/ugee maker: bare present-tense with
# no "today" at all; ActivTrak/IFS Softeon: "introduced" in any tense at
# all). Rather than keep discovering and hand-adding one more tense
# variant per verb forever, `_ANNOUNCEMENT_VERB_LEMMAS` + the conjugation
# helpers below generate EVERY temporal-marker collocation (today X / X
# today / has X / is X-ing / today X-s / X-s today) for every lemma at
# once -- adding a lemma now automatically gets full tense coverage, so
# this specific class of gap (a verb covered in SOME tenses but not
# others) cannot recur for any verb already in the list.
_ANNOUNCEMENT_VERB_LEMMAS = ("announce", "launch", "unveil", "introduce")
# `achieve`/`release` are deliberately NOT in this list: both have
# plausible EVERGREEN bare-present-tense uses with no announcement sense
# at all ("the model achieves 95% accuracy", "the fund releases quarterly
# reports") -- unlike announce/launch/unveil/introduce, which are almost
# never used in a non-dated sense. They keep their existing, narrower,
# explicitly-marker-qualified patterns only (today achieved/has achieved,
# today released/has released/is releasing/etc., hand-written below) --
# a deliberate, reasoned exception, not an oversight.


def _past_form(lemma: str) -> str:
    return lemma + "d" if lemma.endswith("e") else lemma + "ed"


def _ing_form(lemma: str) -> str:
    return lemma[:-1] + "ing" if lemma.endswith("e") else lemma + "ing"


def _s_form(lemma: str) -> str:
    if lemma.endswith(("ch", "sh", "ss", "x", "z", "o")):
        return lemma + "es"
    return lemma + "s"


def _generated_verb_tense_patterns() -> Tuple[str, ...]:
    """For each lemma: today/has/is-marked past, present-continuous, and
    present-tense collocations, in both word orders where applicable --
    every one of these requires an explicit temporal marker ("today",
    "has", "is"), so none of them risk matching evergreen/habitual prose;
    they only ever disambiguate WHICH tense of an already-announcement-
    shaped verb is present."""
    patterns: List[str] = []
    for lemma in _ANNOUNCEMENT_VERB_LEMMAS:
        past, ing, s_form = _past_form(lemma), _ing_form(lemma), _s_form(lemma)
        patterns.extend([
            rf"\btoday\s+{past}\b", rf"\b{past}\s+today\b", rf"\bhas\s+{past}\b",
            rf"\bis\s+{ing}\b", rf"\btoday\s+{s_form}\b", rf"\b{s_form}\s+today\b",
        ])
    return tuple(patterns)


# Bare present-tense, third-person-singular, with NO temporal marker at
# all (confirmed HR Path / ugee maker forensic findings, 2026-10-13: "HR
# Path... announces the acquisition of SnapAI...", "ugee maker introduces
# a third... option...") -- genuinely dated, one-time announcements with
# no "today"/"has" anywhere in the sentence. Accepted generically for the
# same four verbs above (reliably announcement-shaped even bare), gated
# by `_RECURRING_CADENCE_RE` below so a genuinely habitual/recurring use
# ("announces its quarterly results every year") is never mistaken for a
# one-time dated development -- the one guard this generic acceptance
# still needs, replacing what used to be a hand-curated, per-verb
# object-noun list (launch/availability/release/acquisition/partnership)
# that could never cover every real object noun a real release might use
# (e.g. ugee maker's "a third... option").
_BARE_PRESENT_VERB_RE = re.compile(
    rf"\b(?:{'|'.join(_s_form(lemma) for lemma in _ANNOUNCEMENT_VERB_LEMMAS)})\b", re.IGNORECASE,
)
# Bare PAST tense, same reasoning (confirmed pre-existing CoreWeave case --
# "CoreWeave announced availability of NVIDIA Vera Rubin NVL72 systems" --
# previously only matched via a hand-curated object-noun list restricted to
# "launch/availability/release"; now generic across all four lemmas, same
# local recurring-cadence guard, so it is never lost again for a noun the
# old curated list didn't happen to include).
_BARE_PAST_VERB_RE = re.compile(
    rf"\b(?:{'|'.join(_past_form(lemma) for lemma in _ANNOUNCEMENT_VERB_LEMMAS)})\b", re.IGNORECASE,
)
_RECURRING_CADENCE_RE = re.compile(
    r"\bevery\s+(day|week|month|quarter|year)\b"
    r"|\b(annually|quarterly|monthly|weekly|daily|biannually|regularly|routinely|periodically)\b",
    re.IGNORECASE,
)
# A second, independent local guard for the bare-PAST-tense check only:
# "unveiled YEARS AGO", "introduced several years ago" are classic
# evergreen/historical framing (a product's known history, not a current
# dated development) using the exact same bare verbs -- confirmed as a
# real false-positive risk the generic bare-past-tense acceptance
# introduced (negative-control test case, 2026-10-13). Distinct from
# `_RECURRING_CADENCE_RE` (that one is about HABITUAL repetition; this one
# is about TEMPORAL DISTANCE from the article's own publication).
_HISTORICAL_DISTANCE_RE = re.compile(
    r"\b(years?|months?|decades?|weeks?)\s+ago\b", re.IGNORECASE,
)
# A third, independent local guard, for BOTH bare checks: "announced IT
# WILL host a panel" / "announces that it plans to..." reports a FUTURE
# INTENTION, not a development that has actually happened -- confirmed
# real-test-fixture false positive (`test_irrelevant_consulting_article_
# is_rejected`, 2026-10-13): a mere "we will do X later" event
# announcement is explicitly NOT a dated development, and bare "announced"
# with no object-noun restriction at all now matches it where the old
# hand-curated noun list (launch/availability/release) never did.
_FUTURE_INTENTION_RE = re.compile(
    r"\b(it|that|which|this)\s+(will|would|could|might|may)\b"
    r"|\b(plans?|intends?)\s+to\b|\bis\s+(set|going)\s+to\b",
    re.IGNORECASE,
)
_RECURRING_CADENCE_CONTEXT_WINDOW = 60
"""Confirmed forensic finding (ugee maker, 2026-10-13): checking for a
recurring-cadence word ANYWHERE in the whole article -- not just near the
bare-verb match -- let an unrelated mention far away in the text (e.g.
"...weekly..." in a completely different paragraph) wrongly veto a real,
one-time "introduces a new X" announcement elsewhere in the same
article. The cadence check only looks at a small window immediately
around each bare-verb match, mirroring `_PARTNERSHIP_CONTEXT_WINDOW`'s
existing local-context approach below."""

_DATED_ANNOUNCEMENT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        *_generated_verb_tense_patterns(),
        r"\brolling\s+out\b", r"\brolled\s+out\b",
        r"\bgenerally\s+available\s+today\b",
        r"\bwe(?:'re|\s+are)\s+releasing\b", r"\bis\s+releasing\b",
        r"\bhas\s+released\b", r"\breleased\s+today\b", r"\btoday\s+released\b",
        r"\bfor\s+immediate\s+release\b",
        r"\btoday\s+achieved\b", r"\bachieved\s+today\b", r"\bhas\s+achieved\b",
        # `announced that` (confirmed KONST forensic finding, 2026-10-12):
        # a reported-speech construction, not a verb-tense variant -- stays
        # hand-written since the generic generator above only produces
        # today/has/is-marked and bare collocations, not this different shape.
        r"\bannounced\s+that\b",
        # `unveiled at <event>` (confirmed ASUS forensic finding,
        # 2026-10-13): "unveiled" as a reduced relative-clause past
        # participle introducing WHERE a dated unveiling happened -- also a
        # different shape from the generic generator's collocations.
        r"\bunveiled\s+at\b",
    )
)


def _contains_dated_announcement_language(text: str) -> bool:
    """True if the evidence text itself contains explicit announcement/
    release language -- the deterministic signal required to trust a
    `is_dated_development=true` judgment (see `_resolve_dated_development`).
    Never looks at a date/title/URL, only the evidence body, so it cannot be
    satisfied merely by page metadata or a dateline-shaped string.

    Checks the fixed, temporal-marker-qualified patterns first (never a
    false-positive risk); only if none of those match does a bare,
    unmarked present-tense announcement verb (`_BARE_PRESENT_VERB_RE`)
    count -- and only when the text does NOT also read as a habitual/
    recurring action (`_RECURRING_CADENCE_RE`), so "announces its
    quarterly results every year" is still correctly rejected while
    "announces the acquisition of X" / "introduces a new option" are
    correctly accepted."""
    if not text:
        return False
    if any(pattern.search(text) for pattern in _DATED_ANNOUNCEMENT_PATTERNS):
        return True
    for bare_verb_re in (_BARE_PRESENT_VERB_RE, _BARE_PAST_VERB_RE):
        for match in bare_verb_re.finditer(text):
            start = max(0, match.start() - _RECURRING_CADENCE_CONTEXT_WINDOW)
            end = match.end() + _RECURRING_CADENCE_CONTEXT_WINDOW
            window = text[start:end]
            if (
                not _RECURRING_CADENCE_RE.search(window)
                and not _HISTORICAL_DISTANCE_RE.search(window)
                and not _FUTURE_INTENTION_RE.search(window)
            ):
                return True  # at least one bare-verb mention is NOT locally habitual/historical
    return False


def _resolve_dated_development(
    assessment: AgentAssessment, evidence: Evidence, article: Optional[Article] = None
) -> Tuple[bool, Optional[str]]:
    """Deterministic guardrail: the LLM's own `is_dated_development` can only
    be trusted when it says True AND the evidence text corroborates it with
    explicit announcement language -- it is never promoted from False to
    True, only narrowed from True to False, so a development the model
    already (correctly or conservatively) called non-dated is left alone.

    Second corroboration path (manager-reported coverage-gap fix,
    2026-10-15): narrative financial journalism (CNBC, Yahoo Finance,
    TradingView, Reuters, ...) routinely reports a real, current
    development without ever using "today announced"/"has launched"-style
    language ("AI Agents to Soon Trade Bonds, HSBC's Investment Bank Head
    Says") -- `_contains_dated_announcement_language` correctly finds
    nothing, and before this fix that silently dropped the item's
    `event_date` to None, which excludes it from every date-bounded digest
    even though it was fully analyzed and judged relevant. The exact same
    metadata-trust rule already applied to LINK-grounded items with no body
    at all (see `agents.link_grounded`: `is_dated_development =
    article.publication_date_precision in (EXACT, DATE_ONLY)`) is reused
    here as a fallback: a precisely dated article (EXACT/DATE_ONLY
    metadata, from a real `<time>`/JSON-LD tag, not a guess) is just as
    reliable a date as explicit announcement language -- only an imprecise
    MONTH/UNKNOWN-precision guess must never stand in for a confirmed date.
    `article` is optional (default None) so every pre-existing direct-call
    test of this function (text-pattern behavior only) is unaffected."""
    if not assessment.is_dated_development:
        return False, assessment.development_date_rationale
    if _contains_dated_announcement_language(evidence.content or ""):
        return True, assessment.development_date_rationale
    if article is not None and article.publication_date_precision in (
        DatePrecision.EXACT, DatePrecision.DATE_ONLY,
    ):
        return True, (
            "No explicit announcement/release language found in the evidence text, but the "
            "article's own publication date is precisely known (EXACT/DATE_ONLY metadata) -- "
            "trusted as the development's date exactly as for metadata-only link-grounded items "
            "(see agents.link_grounded)."
        )
    return False, (
        "Deterministic override: the evidence text contains no explicit "
        "announcement/release language (e.g. 'today announced', 'has "
        "launched'), and the article's own publication date is not precisely known either, so "
        "is_dated_development=true was not corroborated and "
        "is not trusted on the model's judgment alone."
    )


# Explicit partnership/collaboration language -- requires a real relational
# verb, not just two company names appearing near each other.
_PARTNERSHIP_LANGUAGE_RE = re.compile(
    r"\bpartner(?:ed|ing)?\s+with\b|\bin\s+partnership\s+with\b|\bpartnership\s+with\b"
    r"|\bin\s+collaboration\s+with\b|\bcollaborat(?:ed|ing)\s+with\b"
    r"|\bjoint\s+venture\s+with\b|\bstrategic\s+alliance\s+with\b"
    r"|\bteamed\s+up\s+with\b|\bco-engineer(?:ed|ing)?\b|\bjoint\s+development\s+with\b",
    re.IGNORECASE,
)

# The same company name appearing near "available through/on/via", "built
# on", "runs on", "supported by", "compatible with", or "powered by" is a
# distribution/integration reference, not a confirmed partnership -- the
# exact pattern the task calls out (AWS/Amazon SageMaker/Microsoft Foundry
# "availability" language being mistaken for a partnership).
_AVAILABILITY_ONLY_RE = re.compile(
    r"\bavailable\b[^.]{0,25}\b(on|through|via|in)\b"
    r"|\bbuilt\s+on\b|\bruns?\s+on\b|\bsupported\s+by\b|\bcompatible\s+with\b|\bpowered\s+by\b"
    r"|\bintegrat(?:ed|ion)\s+with\b",
    re.IGNORECASE,
)

_PARTNERSHIP_CONTEXT_WINDOW = 120


def _validate_partnerships_against_evidence(partnerships: List[str], evidence_text: str) -> List[str]:
    """Deterministic guardrail: a name only survives as a validated partner
    if the evidence text itself uses real relationship language near that
    name -- never merely because the LLM listed it under `partnerships`.
    A name found only near "available through/on", "built on", "runs on",
    etc. is explicitly NOT a partnership. A name not found in the evidence
    text at all, or found with no corroborating language either way, is
    dropped (conservative: prefer false when evidence is insufficient)."""
    if not evidence_text or not partnerships:
        return []
    lowered = evidence_text.lower()
    validated: List[str] = []
    for name in partnerships:
        idx = lowered.find(name.lower())
        if idx == -1:
            continue
        start = max(0, idx - _PARTNERSHIP_CONTEXT_WINDOW)
        end = idx + len(name) + _PARTNERSHIP_CONTEXT_WINDOW
        window = evidence_text[start:end]
        if _PARTNERSHIP_LANGUAGE_RE.search(window):
            validated.append(name)
        # Availability-only language (or no language at all) never qualifies.
    return validated


def _derive_nomura_relevance(strategic_significance: RelevanceAssessment) -> RelevanceAssessment:
    """`AgentAssessment` no longer asks the model for `nomura_relevance`
    (see `agents.schemas.AgentAssessment`'s comment -- nothing downstream
    reads it besides DB storage), but `ResearchItem.nomura_relevance` and
    the `research_items` table's `nomura_relevance_level`/`_rationale`
    columns are both still required (DB schema is explicitly out of scope
    for this change). Mirroring `strategic_significance` is a deliberate,
    documented stand-in, not a guess: both fields are the agent's judgment
    of how strategically relevant a development is to Nomura specifically,
    and the model was never shown a prompt that distinguished them in the
    first place (neither field name appears in any `prompts/*_system.md`
    file -- the model only ever saw the bare JSON schema)."""
    return RelevanceAssessment(
        level=strategic_significance.level,
        rationale=(
            "Not independently requested from the model (cost reduction); "
            "mirrors the strategic-significance assessment above."
        ),
    )


# ---------------------------------------------------------------------------
# Deterministic recall safeguard: Qwen's deep-analysis `relevance.label`
# is a judgment from an imperfect, low-parameter local model, NOT ground
# truth. An IRRELEVANT verdict on a candidate that deterministic evidence
# establishes as clearly within the product's own configured intelligence
# scope must not be allowed to silently discard it. This is deliberately
# NOT "IRRELEVANT -> RELEVANT for everything" -- it requires BOTH genuine
# topical breadth (hits in at least two independent scope categories, never
# just one generic "AI" mention) AND a deterministically-confirmed dated
# development (the same `_contains_dated_announcement_language` check used
# elsewhere, computed independently of whatever Qwen itself claimed for
# `is_dated_development`). Routing having already placed this candidate
# with THIS specialist agent in the first place is the baseline scope
# signal (every candidate reaching here already matched the agent's own
# domain); these two further, stricter conditions are what keeps the
# override narrow and auditable rather than a blanket override.
# ---------------------------------------------------------------------------

_SCOPE_CATEGORY_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "ai_core": (
        r"\bai\b", r"\bgenai\b", r"generative ai", r"artificial intelligence",
        r"\bagentic\b", r"\bai agents?\b", r"foundation model", r"large language model", r"\bllms?\b",
    ),
    "governance_security": (
        r"governance", r"\bsecurity\b", r"\bidentity\b", r"compliance",
        r"risk management", r"access control", r"\biam\b",
    ),
    "enterprise_adoption": (
        r"enterprise", r"\badoption\b", r"business transformation", r"digital transformation", r"deployment",
    ),
    "research_findings": (
        r"\bresearch\b", r"\bstudy\b", r"\bsurvey\b", r"report finds", r"findings", r"analysis reveals",
    ),
    "deal_activity": (
        r"\bfunding\b", r"\bpartnership\b", r"\bacquisition\b", r"\binvestment\b",
        r"platform launch", r"product launch", r"\binfrastructure\b",
    ),
    "financial_services": (
        r"financial services", r"\bfintech\b", r"capital markets", r"\bbanking\b",
        r"\btrading\b", r"investment bank",
    ),
}
_SCOPE_CATEGORY_RES: Dict[str, Tuple[re.Pattern, ...]] = {
    category: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for category, patterns in _SCOPE_CATEGORY_PATTERNS.items()
}
_MIN_SCOPE_CATEGORIES_FOR_OVERRIDE = 2


def _broad_scope_category_hits(text: str) -> Set[str]:
    """Which of the curated, broad-but-bounded intelligence-scope
    categories the evidence text hits -- never a single bare keyword
    count, always a named category, so the override threshold below means
    something auditable ("hit >=2 distinct categories"), not "mentioned AI
    twice"."""
    if not text:
        return set()
    return {category for category, patterns in _SCOPE_CATEGORY_RES.items() if any(p.search(text) for p in patterns)}


def _resolve_relevance_override(
    assessment: AgentAssessment, evidence: Evidence,
) -> Tuple[bool, Optional[str]]:
    """Only ever called when `assessment.relevance.label == IRRELEVANT`.
    Returns `(override_fired, diagnostic_or_None)`. Never promotes
    AMBIGUOUS (which already reaches construction on its own) and never
    considers anything the model itself already judged relevant -- this
    exists solely to catch the specific "Qwen said irrelevant, but
    deterministic scope evidence disagrees" case, with the disagreement
    always recorded, never hidden, when it fires."""
    text = evidence.content or ""
    category_hits = _broad_scope_category_hits(text)
    is_dated = _contains_dated_announcement_language(text)

    if len(category_hits) < _MIN_SCOPE_CATEGORIES_FOR_OVERRIDE or not is_dated:
        return False, None

    diagnostic = (
        f"qwen_relevance=irrelevant deterministic_scope_signal=in_scope final_relevance=relevant "
        f"override_reason=matched_categories={sorted(category_hits)} dated_development=true "
        f"qwen_reason={assessment.relevance.reason!r}"
    )
    return True, diagnostic


def _generate_research_item_id(agent: AgentName, article_id: str) -> str:
    basis = f"{agent.value}:{article_id}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:_ID_LENGTH]


def _flatten_ai_keywords(topics: TopicsRegistry) -> List[str]:
    buckets = [
        topics.ai_topics.core,
        topics.ai_topics.models,
        topics.ai_topics.infrastructure,
        topics.ai_topics.enterprise,
        topics.ai_topics.governance,
    ]
    return [keyword.lower() for bucket in buckets for keyword in bucket]


class ResearchAgent:
    agent_name: ClassVar[AgentName]
    system_prompt_file: ClassVar[str]
    assessment_schema: ClassVar[Type[AgentAssessment]] = AgentAssessment

    def __init__(
        self, llm_client: LLMClient, config: AppConfig, *, monitored_names: Optional[Set[str]] = None
    ) -> None:
        """`monitored_names=None` means "discovery-driven, no fixed company
        registry" (see StartupAgent) — `supports()` then accepts any article
        and relies entirely on the keyword pre-filter + LLM relevance
        decision, rather than an enormous hardcoded company list."""
        self._llm = llm_client
        self._config = config
        self._monitored_names = monitored_names
        self._system_prompt = load_prompt(self.system_prompt_file)
        self._keywords = _flatten_ai_keywords(config.topics) + [
            k.lower().replace("_", " ") for k in self._extra_keywords()
        ]  # normalizes snake_case config identifiers (e.g. "series_a") to match natural text ("series A")

    def supports(self, article: Article) -> bool:
        """Whether this agent's scope covers this article's source.

        With a fixed company registry (Peer/AIProduct/Analyst), this checks
        membership. With `monitored_names=None` (discovery-driven, Startup),
        any article is in-scope structurally — the pre-filter and the LLM's
        own relevance judgment are what actually narrow things down.
        """
        if self._monitored_names is None:
            return True
        return article.source_name in self._monitored_names

    def _extra_keywords(self) -> List[str]:
        """Hook for a subclass to widen the pre-filter beyond the generic AI
        topic list (e.g. StartupAgent adds config.sources.startups
        categories/funding_events). Empty by default."""
        return []

    def analyze(
        self, article: Article, evidence: Evidence, *, llm_client: Optional[LLMClient] = None,
    ) -> Optional[ResearchItem]:
        """Article/Evidence -> ResearchItem, or None if rejected as irrelevant.

        Raises `research_assistant.llm.LLMError` if the configured LLM is
        unavailable or returns output that fails schema validation — this
        propagates to the caller rather than being swallowed, so a failed
        analysis is never silently mistaken for "no development found".

        `llm_client`, when supplied, is used for this one call INSTEAD of
        the agent's own `self._llm` -- the hook CURRENT's bounded deep-
        analysis funnel uses to apply a shorter, dedicated timeout
        (`current_llm.deep_analysis_timeout_seconds`) without touching the
        agent's own client (still the global 150s timeout) or any other
        caller (Morning/Evening/manual all call `analyze()` with no
        override, completely unaffected).
        """
        if evidence.article_id != article.id:
            raise ValueError("evidence does not belong to the supplied article")

        if not self._passes_keyword_prefilter(article):
            logger.info(
                "%s: pre-filter rejected article %s (%s) — no AI-related keywords found",
                self.agent_name.value, article.id, article.url,
            )
            return None

        prompt = self._build_prompt(article, evidence)
        client = llm_client if llm_client is not None else self._llm
        assessment = client.generate_structured(
            system=self._system_prompt, prompt=prompt, schema=self.assessment_schema
        )

        relevance_override: Optional[str] = None
        if assessment.relevance.label == RelevanceLabel.IRRELEVANT:
            override_fired, override_diagnostic = _resolve_relevance_override(assessment, evidence)
            if not override_fired:
                logger.info(
                    "%s: LLM marked article %s irrelevant (%s)",
                    self.agent_name.value, article.id, assessment.relevance.reason,
                )
                return None
            # Deterministic recall safeguard fired: Qwen's own IRRELEVANT
            # judgment is overridden, never silently -- `relevance_override`
            # is persisted on the resulting ResearchItem (see models.
            # research_item.ResearchItem.relevance_override) so the
            # disagreement is always auditable, never hidden.
            logger.warning(
                "%s: deterministic scope safeguard overrode LLM relevance=irrelevant -> relevant for "
                "article %s (%s)", self.agent_name.value, article.id, override_diagnostic,
            )
            relevance_override = override_diagnostic
            assessment = assessment.model_copy(
                update={
                    "relevance": RelevanceDecision(
                        label=RelevanceLabel.RELEVANT,
                        reason=(
                            f"Deterministic scope safeguard override (model said irrelevant: "
                            f"{assessment.relevance.reason})"
                        ),
                    ),
                    # The override already independently re-confirmed
                    # dated-announcement language via
                    # `_contains_dated_announcement_language` (see
                    # `_resolve_relevance_override`) -- set True here too
                    # so the EXISTING `_resolve_dated_development` guardrail
                    # below corroborates against the same evidence text and
                    # reaches the same conclusion, rather than trusting
                    # whatever (possibly stale/irrelevant-biased) value
                    # Qwen happened to set alongside its IRRELEVANT verdict.
                    "is_dated_development": True,
                }
            )

        is_dated_development, development_date_rationale = _resolve_dated_development(assessment, evidence, article)
        if is_dated_development != assessment.is_dated_development:
            logger.info(
                "%s: deterministic guardrail overrode is_dated_development=%s -> %s for article %s",
                self.agent_name.value, assessment.is_dated_development, is_dated_development, article.id,
            )
        validated_partnerships = _validate_partnerships_against_evidence(
            assessment.partnerships, evidence.content or ""
        )
        if validated_partnerships != assessment.partnerships:
            logger.info(
                "%s: deterministic guardrail narrowed partnerships %s -> %s for article %s",
                self.agent_name.value, assessment.partnerships, validated_partnerships, article.id,
            )

        return self._build_research_item(
            article, evidence, assessment,
            is_dated_development=is_dated_development,
            development_date_rationale=development_date_rationale,
            validated_partnerships=validated_partnerships,
            relevance_override=relevance_override,
        )

    def analyze_many(self, items: List[Tuple[Article, Evidence]]) -> List[ResearchItem]:
        """Analyze several (Article, Evidence) pairs, skipping rejected ones."""
        results: List[ResearchItem] = []
        for article, evidence in items:
            item = self.analyze(article, evidence)
            if item is not None:
                results.append(item)
        return results

    def _passes_keyword_prefilter(self, article: Article) -> bool:
        """Cheap pre-filter to skip an LLM call when nothing AI-related is even
        mentioned. Passing this check is necessary but not sufficient —
        the LLM still makes the actual relevance/irrelevance call."""
        if not self._keywords:
            return True
        haystack = f"{article.title or ''}\n{article.clean_text or ''}".lower()
        return any(keyword in haystack for keyword in self._keywords)

    def _build_prompt(self, article: Article, evidence: Evidence) -> str:
        publication_date = evidence.publication_date.isoformat() if evidence.publication_date else "unknown"
        return (
            "Assess the following evidence. Return only the structured object "
            "matching the provided response schema — no prose, no markdown, "
            "no extra commentary.\n\n"
            f"Company/source being monitored: {article.source_name}\n"
            f"Publisher: {evidence.publisher or 'unknown'}\n"
            f"Title: {evidence.title or 'unknown'}\n"
            f"Publication date: {publication_date}\n"
            f"URL: {evidence.source_url}\n\n"
            "Note on 'Publication date' above: this is page metadata (when the "
            "page was retrieved or last modified) -- it is NOT proof that a "
            "specific, dated event happened on that day. Decide "
            "`is_dated_development` from the EVIDENCE TEXT itself: true only if "
            "the text describes a specific action that was announced/launched/"
            "released/occurred (e.g. 'today announced', 'has launched', a dated "
            "press-release lede); false for an evergreen/reference page that "
            "merely describes or lists a product/model/feature without stating "
            "a specific new action. When false, still extract whatever the "
            "evidence supports, but do not let the 'Publication date' field "
            "above leak into `summary` or anywhere else as if it were a "
            "confirmed announcement date.\n\n"
            "Evidence content (this is the ONLY source of facts you may use):\n"
            "---\n"
            f"{_capped_evidence_content(evidence.content or '')}\n"
            "---\n"
        )

    def _build_research_item(
        self,
        article: Article,
        evidence: Evidence,
        assessment: AgentAssessment,
        *,
        is_dated_development: bool,
        development_date_rationale: Optional[str],
        validated_partnerships: List[str],
        relevance_override: Optional[str] = None,
    ) -> ResearchItem:
        return ResearchItem(
            id=_generate_research_item_id(self.agent_name, article.id),
            agent=self.agent_name,
            article_id=article.id,
            publisher=article.publisher,
            title=article.title,
            url=article.url,
            # Phase 14 date-precision fix: a MONTH-precision date (e.g.
            # Bank of America's URL-derived "September 2026", stored as
            # day 1 purely as a placeholder) must never be carried forward
            # as if it were this item's real, exact publication date --
            # doing so would let it wrongly qualify for an exact bounded
            # operational digest window it was never actually confirmed to
            # fall inside. The Article itself still retains the full
            # month-precision date for broader historical context; only
            # the ResearchItem (which feeds Event.event_date and therefore
            # all period-bound eligibility) treats it as unknown, via the
            # EXISTING, unmodified "no publication_date -> excluded from a
            # bounded digest" rule in events/builder.py -- no change
            # needed there.
            publication_date=(
                article.publication_date
                if article.publication_date_precision != DatePrecision.MONTH
                else None
            ),
            company=article.source_name,
            relevance=assessment.relevance,
            category=assessment.category,
            event_type=assessment.event_type,
            summary=assessment.summary,
            is_dated_development=is_dated_development,
            development_date_rationale=development_date_rationale,
            entities=assessment.entities,
            people=assessment.people,
            products=assessment.products,
            deal_value=assessment.deal_value,
            partnerships=validated_partnerships,
            technologies=assessment.technologies,
            geography=assessment.geography,
            financial_services_relevance=assessment.financial_services_relevance,
            nomura_relevance=_derive_nomura_relevance(assessment.strategic_significance),
            strategic_significance=assessment.strategic_significance,
            evidence=[evidence.evidence_id],
            source_quality=SourceQuality(source_type=article.source_type, source_priority=article.source_priority),
            analyst_findings=getattr(assessment, "analyst_findings", None),
            relevance_override=relevance_override,
        )
