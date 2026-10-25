"""Phase 13C: deterministic post-LLM guardrails. These exercise the actual
production functions (`agents.base._resolve_dated_development`,
`_contains_dated_announcement_language`, `_validate_partnerships_against_evidence`,
`agents.report._sanitize_partnership_claims`, `_aggregate_facts`) directly --
never just prompt-string inspection -- plus one full `ResearchAgent.analyze()`
integration test proving the guardrail applies through the real code path.
"""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.agents.base import (
    _contains_dated_announcement_language,
    _resolve_dated_development,
    _validate_partnerships_against_evidence,
)
from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.report import _aggregate_facts, _sanitize_partnership_claims, _strip_inflated_language
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.evidence.conversion import article_to_evidence
from research_assistant.evidence.normalizer import normalize_article
from research_assistant.models.enums import EntityType
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
    TypedEntity,
)
from research_assistant.retrieval.models import ExtractedArticle

from tests.agents.conftest import FakeLLMClient
from tests.events.conftest import make_item

EVERGREEN_TEXT = (
    "Claude Sonnet 5.5 is a clear upgrade over Sonnet 5 that runs 30% faster "
    "and costs up to 30% less for most work. Anyone can chat with Claude "
    "using Sonnet 5.5 on Claude.ai, available on web, iOS, and Android. "
    "Sonnet 5.5 is available on the Claude Platform natively, and in Amazon "
    "Web Services, Google Cloud, and Microsoft Foundry."
)

DATED_BOFA_TEXT = (
    "September 28, 2026 at 10:00 AM Eastern. NEW YORK, New York -- Bank of "
    "America today announced the launch of Ask GPS Intelligence Hub, a "
    "suite of tools designed to help employees deliver deeper client insights."
)

DATED_COREWEAVE_TEXT = (
    "Building on nearly a decade of co-engineering, CoreWeave has built "
    "NVIDIA compute, networking and software into a cloud purpose-built for "
    "AI. At CoreWeave Fully Connected, running this week in San Francisco, "
    "CoreWeave announced availability of NVIDIA Vera Rubin NVL72 systems."
)

COHERE_AVAILABILITY_TEXT = (
    "Today, we're releasing Embed 5, a new family of embeddings models. "
    "Embed 5 is generally available today on the Cohere API and Model "
    "Vault, Microsoft Foundry, and Amazon SageMaker."
)


def _assessment(**overrides) -> AgentAssessment:
    kwargs = dict(
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="x"),
        category=TopicCategory.ENTERPRISE_AI,
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
    )
    kwargs.update(overrides)
    return AgentAssessment(**kwargs)


class _FakeEvidence:
    def __init__(self, content):
        self.content = content


# ---------------------------------------------------------------------------
# DATE (tests 1-4)
# ---------------------------------------------------------------------------


def test_1_evergreen_page_forces_is_dated_development_false():
    assessment = _assessment(is_dated_development=True)  # the LLM wrongly says true
    resolved, rationale = _resolve_dated_development(assessment, _FakeEvidence(EVERGREEN_TEXT))

    assert resolved is False
    assert "override" in rationale.lower()


def test_2_dated_announcement_preserves_is_dated_development_true():
    assessment = _assessment(is_dated_development=True)
    resolved, _ = _resolve_dated_development(assessment, _FakeEvidence(DATED_BOFA_TEXT))

    assert resolved is True


def test_3_llms_own_false_is_never_promoted_to_true():
    assessment = _assessment(is_dated_development=False)
    resolved, _ = _resolve_dated_development(assessment, _FakeEvidence(DATED_BOFA_TEXT))  # strong signal present

    assert resolved is False  # the guardrail only narrows true->false, never the reverse


def test_4_a_bare_date_string_with_no_announcement_language_is_not_dated():
    assessment = _assessment(is_dated_development=True)
    text = "Updated September 2026. This product provides enterprise search and retrieval capabilities."
    resolved, _ = _resolve_dated_development(assessment, _FakeEvidence(text))

    assert resolved is False


# 2026-10-15 manager-reported coverage-gap fix: narrative financial
# journalism (CNBC, Yahoo Finance, TradingView, Reuters, ...) reports a
# real, current development without ever using "today announced"/"has
# launched"-style language -- before this fix, that silently dropped the
# item's `event_date` to None, excluding it from every date-bounded digest
# even though it was fully analyzed and judged relevant. Real production
# case: Yahoo Finance's "AI Agents to Soon Trade Bonds, HSBC's Investment
# Bank Head Says" (run 6627f19b41c3382b) -- fully analyzed, judged
# relevant, but excluded from the digest for exactly this reason.
NARRATIVE_MARKET_TEXT = (
    "AI agents will soon be trading bonds, according to HSBC's investment bank head. Speaking at a "
    "conference, the executive said electronic and algorithmic trading already dominates parts of the "
    "fixed income market, and AI agents are the natural next step for bond trading desks."
)


def _narrative_article(precision):
    from research_assistant.evidence.service import NormalizationService
    from research_assistant.models.enums import SourcePriority, SourceType
    from research_assistant.retrieval.models import SourceCategory, SourceDefinition

    source = SourceDefinition(
        name="Yahoo Finance", domain="finance.yahoo.com", urls=["https://finance.yahoo.com/topic/artificial-intelligence/"],
        source_type=SourceType.SECONDARY, priority=SourcePriority.HIGH, category=SourceCategory.SECONDARY,
    )
    extracted = ExtractedArticle(
        url="https://finance.yahoo.com/markets/stocks/articles/ai-agents-soon-trade-bonds-095256902.html",
        title="AI Agents to Soon Trade Bonds, HSBC's Investment Bank Head Says",
        text=NARRATIVE_MARKET_TEXT,
        published_at=datetime(2026, 10, 15, 9, 52, 56, tzinfo=timezone.utc),
    )
    article = NormalizationService().normalize(extracted, source=source)
    return article.model_copy(update={"publication_date_precision": precision})


def test_precisely_dated_narrative_article_with_no_announcement_language_is_still_trusted():
    from research_assistant.models.article import DatePrecision

    assessment = _assessment(is_dated_development=True)
    article = _narrative_article(DatePrecision.EXACT)

    resolved, rationale = _resolve_dated_development(assessment, _FakeEvidence(NARRATIVE_MARKET_TEXT), article)

    assert resolved is True
    assert "publication date is precisely known" in rationale


def test_date_only_precision_narrative_article_is_also_trusted():
    from research_assistant.models.article import DatePrecision

    assessment = _assessment(is_dated_development=True)
    article = _narrative_article(DatePrecision.DATE_ONLY)

    resolved, _ = _resolve_dated_development(assessment, _FakeEvidence(NARRATIVE_MARKET_TEXT), article)

    assert resolved is True


def test_month_precision_narrative_article_is_not_trusted_by_the_fallback():
    """The fallback must stay as strict as the existing LINK-grounded rule
    (agents.link_grounded) it mirrors -- MONTH/UNKNOWN precision is a
    storage placeholder, not a confirmed date, so it must never promote
    is_dated_development even when the text itself has no announcement
    language."""
    from research_assistant.models.article import DatePrecision

    assessment = _assessment(is_dated_development=True)
    article = _narrative_article(DatePrecision.MONTH)

    resolved, rationale = _resolve_dated_development(assessment, _FakeEvidence(NARRATIVE_MARKET_TEXT), article)

    assert resolved is False
    assert "override" in rationale.lower()


def test_without_an_article_argument_behavior_is_unchanged():
    """Every pre-existing direct call to this function (text-pattern
    behavior only, no `article`) must be completely unaffected by this
    fix -- `article` defaults to None, which never engages the new
    fallback path."""
    assessment = _assessment(is_dated_development=True)
    resolved, _ = _resolve_dated_development(assessment, _FakeEvidence(NARRATIVE_MARKET_TEXT))

    assert resolved is False


def test_contains_dated_announcement_language_directly():
    assert _contains_dated_announcement_language(DATED_BOFA_TEXT) is True
    assert _contains_dated_announcement_language(DATED_COREWEAVE_TEXT) is True
    assert _contains_dated_announcement_language(COHERE_AVAILABILITY_TEXT) is True
    assert _contains_dated_announcement_language(EVERGREEN_TEXT) is False
    assert _contains_dated_announcement_language("") is False


# CURRENT run 7aab105954d38497 forensic finding: the real Bits In Glass PR
# Newswire release body text ("has achieved Select partner status") was a
# genuine, same-family verb (present-perfect, reporting a just-completed
# specific action) this list did not yet cover -- Qwen itself correctly
# judged `is_dated_development=true`, but the deterministic guardrail
# narrowed it back to False purely because this verb was missing, costing
# the event its date and dropping it from the CURRENT digest.
BITS_IN_GLASS_TEXT = (
    "DENVER, Oct. 8, 2026 /PRNewswire/ -- Bits In Glass (BIG) has achieved Select partner status in the "
    "Claude Partner Network Services Track. The Services Track recognizes partners for building a "
    "certified delivery team, delivering Claude use cases for clients, and proving the work through "
    "public customer references."
)

# IQuest-Q1's real release text, from the SAME run -- genuinely evergreen
# reception/technical-showcase commentary ("has drawn ... since launch"),
# never an announcement of the launch itself. Must stay undated: this is
# the negative control proving the "achieved" fix above is narrow, not a
# blanket relaxation of the guardrail.
IQUEST_Q1_TEXT = (
    "BEIJING, Oct. 8, 2026 /PRNewswire/ -- IQuest-Q1 has drawn early praise from developers and industry "
    "watchers since launch, particularly for its performance on coding, software engineering, interactive "
    "application generation, and long-horizon agentic workloads."
)


def test_has_achieved_is_recognized_as_dated_announcement_language():
    assert _contains_dated_announcement_language(BITS_IN_GLASS_TEXT) is True


def test_bits_in_glass_dated_development_is_preserved_through_the_guardrail():
    assessment = _assessment(is_dated_development=True)  # Qwen's real, correct judgment on this article
    resolved, _ = _resolve_dated_development(assessment, _FakeEvidence(BITS_IN_GLASS_TEXT))

    assert resolved is True


def test_iquest_q1_reception_commentary_is_not_dated_announcement_language():
    """Negative control: evergreen reception/review commentary about a
    model that "has drawn ... since launch" is not itself an announcement
    of that launch -- the "achieved" fix must not widen the guardrail to
    catch unrelated present-perfect phrasing."""
    assert _contains_dated_announcement_language(IQUEST_Q1_TEXT) is False


# Forensic finding, 2026-10-12: the real PR Newswire KONST Series B release
# ("KONST ... announced that it has raised $30 million Series B
# funding...") uses the reported-speech construction "announced that",
# which none of the existing collocations ("has announced", "today
# announced", "is announcing", "announced the launch/availability/
# release") matched -- Qwen correctly judged `is_dated_development=true`,
# but the deterministic guardrail silently narrowed it back to False,
# discarding a real, already-extracted publication date (exact same class
# of gap as the "today released"/"has achieved" fixes above).
KONST_TEXT = (
    "TAIPEI, Oct. 9, 2026 /PRNewswire/ -- KONST, a Taiwan-based next-generation AI compute "
    "operator, announced that it has raised $30 million Series B funding led by ADATA "
    "Technology (TPEx: 3260), with strategic participation from M Mobility, Pegatron "
    "Venture Capital, and several other corporate investors."
)

# Negative control, same forensic investigation: the real PR Newswire
# PUFsecurity "Trusted AI Agents Must Be Anchored in Silicon" release is a
# technical-perspective/explainer piece ("the company explains that...",
# "PUFsecurity sees...") with no explicit statement that a specific dated
# action occurred -- it must remain correctly undated; the "announced that"
# fix must not widen the guardrail to catch unrelated explanatory prose.
PUFSECURITY_TEXT = (
    "SAN JOSE, Calif., Oct. 9, 2026 /PRNewswire/ -- The first wave of generative AI answered "
    "questions and produced content, and a person still decided what to do with it. "
    "PUFsecurity, a subsidiary of eMemory Technology, believes this shift changes where trust "
    "has to come from. In a new technical perspective, \"Building Trusted AI Agents from the "
    "Silicon Root of Trust,\" the company explains that the security of these agents "
    "ultimately rests on hardware Roots of Trust in the chips that run them."
)


def test_announced_that_is_recognized_as_dated_announcement_language():
    assert _contains_dated_announcement_language(KONST_TEXT) is True


def test_konst_dated_development_is_preserved_through_the_guardrail():
    assessment = _assessment(is_dated_development=True)  # Qwen's real, correct judgment on this article
    resolved, _ = _resolve_dated_development(assessment, _FakeEvidence(KONST_TEXT))

    assert resolved is True


def test_pufsecurity_explainer_commentary_is_not_dated_announcement_language():
    """Negative control: an explanatory technical-perspective piece with no
    announcement verb at all must remain undated -- the "announced that"
    fix must not widen the guardrail beyond this one reported-speech verb."""
    assert _contains_dated_announcement_language(PUFSECURITY_TEXT) is False


# Forensic finding, 2026-10-12 (CURRENT run `b9eb35cb5362b752`): the real
# PR Newswire SWI Group release ("SWI Group ... today announces an €800
# million capital increase with institutional investors...") uses the
# present-tense third-person-singular form of the verb -- a generic tense
# variant, not a topic-specific phrase about capital increases/financing.
# Qwen correctly judged `is_dated_development=true`, but the deterministic
# guardrail silently narrowed it back to False because only past-tense
# "announced" and present-continuous "is announcing" were covered, not
# present-tense "announces" -- the exact same class of gap as the "today
# released"/"has achieved"/"announced that" fixes above.
_SWI_PUBLISHED_AT = datetime(2026, 10, 12, 4, 0, tzinfo=timezone.utc)

SWI_TEXT = (
    "SINGAPORE and AMSTERDAM, Oct. 12, 2026 /PRNewswire/ -- Euronext Amsterdam-listed SWI Group "
    "(\"SWI\"; SWICH) today announces an €800 million capital increase with institutional investors. "
    "The capital increase is realized through the issuance of new ordinary shares by SWICH in several "
    "series, increasing the total share capital to 558,399,385 ordinary shares."
)


def test_today_announces_is_recognized_as_dated_announcement_language():
    assert _contains_dated_announcement_language("The firm today announces a new product line.") is True


def test_announces_today_is_recognized_as_dated_announcement_language():
    assert _contains_dated_announcement_language("The firm announces today a new product line.") is True


def test_swi_group_dated_development_is_preserved_through_the_guardrail():
    """The real SWI sentence is recognized, and the guardrail preserves
    (never narrows) the model's correct is_dated_development=true verdict."""
    assert _contains_dated_announcement_language(SWI_TEXT) is True

    assessment = _assessment(is_dated_development=True)  # Qwen's real, correct judgment on this article
    resolved, _ = _resolve_dated_development(assessment, _FakeEvidence(SWI_TEXT))

    assert resolved is True


def test_existing_announcement_patterns_still_work_after_the_present_tense_addition():
    """Regression guard: adding present-tense 'announces' coverage must
    never disturb any previously-supported pattern."""
    assert _contains_dated_announcement_language(DATED_BOFA_TEXT) is True  # "today announced"
    assert _contains_dated_announcement_language(DATED_COREWEAVE_TEXT) is True  # "announced availability of"
    assert _contains_dated_announcement_language(COHERE_AVAILABILITY_TEXT) is True  # "is releasing"
    assert _contains_dated_announcement_language(BITS_IN_GLASS_TEXT) is True  # "has achieved"
    assert _contains_dated_announcement_language(KONST_TEXT) is True  # "announced that"


def test_unrelated_text_is_not_accidentally_accepted_by_the_present_tense_fix():
    """Negative control: evergreen/non-announcement text, and text merely
    containing the bare word 'announces' without a 'today'/'today ... '
    collocation, must still be rejected -- the fix is narrowly scoped to
    the exact 'today announces'/'announces today' construction, never a
    bare verb match."""
    assert _contains_dated_announcement_language(EVERGREEN_TEXT) is False
    assert _contains_dated_announcement_language(PUFSECURITY_TEXT) is False
    assert _contains_dated_announcement_language(
        "The company announces its quarterly results every year in March, unrelated to today."
    ) is False


def test_deterministic_guardrail_still_rejects_unsupported_model_claims_after_the_fix():
    """The guardrail still only ever narrows true->false, never promotes
    false->true, and still rejects a model's True claim when no
    announcement language of any kind -- including the new present-tense
    forms -- is actually present."""
    assessment_true = _assessment(is_dated_development=True)
    resolved, rationale = _resolve_dated_development(assessment_true, _FakeEvidence(EVERGREEN_TEXT))
    assert resolved is False
    assert "override" in rationale.lower()

    assessment_false = _assessment(is_dated_development=False)
    resolved_false, _ = _resolve_dated_development(assessment_false, _FakeEvidence(SWI_TEXT))
    assert resolved_false is False  # strong signal present, but the model's own False is never promoted


# ---------------------------------------------------------------------------
# `announces the <noun>` / `unveiled at` (confirmed HR Path / ASUS forensic
# findings, CURRENT run 2872655dac13437e, 2026-10-13): two more real
# press-release constructions neither "today announces"/"today announced"
# nor "today unveiled"/"has unveiled" covered.
# ---------------------------------------------------------------------------

HR_PATH_TEXT = (
    "PARIS, Oct. 12, 2026 /PRNewswire/ -- HR Path, a global leader in HR consulting and HRIS "
    "solutions, announces the acquisition of SnapAI, a start-up specialising in Artificial "
    "Intelligence and regulatory compliance. This acquisition strengthens HR Path's expertise in a "
    "strategic field for the future of Human Resources."
)

ASUS_TEXT = (
    "SAN JOSE, Calif., Oct. 12, 2026 /PRNewswire/ -- ASUS is expanding its AI infrastructure "
    "portfolio with the launch of the XA P2N-E2, a 2U NVIDIA MGX architecture-based server powered "
    "by NVIDIA Vera CPUs and designed for agentic AI and reinforcement learning workloads. "
    "Unveiled at OCP Global Summit 2026, this new ASUS server powered by the Vera CPU targets the "
    "execution layer of AI applications."
)


def test_announces_the_acquisition_is_recognized_as_dated_announcement_language():
    assert _contains_dated_announcement_language("The firm announces the acquisition of a rival.") is True


def test_hr_path_dated_development_is_preserved_through_the_guardrail():
    assert _contains_dated_announcement_language(HR_PATH_TEXT) is True

    assessment = _assessment(is_dated_development=True)
    resolved, _ = _resolve_dated_development(assessment, _FakeEvidence(HR_PATH_TEXT))

    assert resolved is True


def test_unveiled_at_is_recognized_as_dated_announcement_language():
    assert _contains_dated_announcement_language("Unveiled at CES 2026, the new device impressed reviewers.") is True


def test_asus_dated_development_is_preserved_through_the_guardrail():
    assert _contains_dated_announcement_language(ASUS_TEXT) is True

    assessment = _assessment(is_dated_development=True)
    resolved, _ = _resolve_dated_development(assessment, _FakeEvidence(ASUS_TEXT))

    assert resolved is True


def test_existing_announcement_patterns_still_work_after_the_two_newest_additions():
    """Regression guard: adding 'announces the <noun>'/'unveiled at'
    coverage must never disturb any previously-supported pattern."""
    assert _contains_dated_announcement_language(DATED_BOFA_TEXT) is True
    assert _contains_dated_announcement_language(DATED_COREWEAVE_TEXT) is True
    assert _contains_dated_announcement_language(BITS_IN_GLASS_TEXT) is True
    assert _contains_dated_announcement_language(KONST_TEXT) is True
    assert _contains_dated_announcement_language(SWI_TEXT) is True


def test_unrelated_text_is_not_accidentally_accepted_by_the_two_newest_fixes():
    """Negative controls: a bare 'announces'/'unveiled' with no qualifying
    collocation, and genuinely evergreen text, must still be rejected."""
    assert _contains_dated_announcement_language(EVERGREEN_TEXT) is False
    assert _contains_dated_announcement_language(
        "The company announces its quarterly results every year in March."
    ) is False
    assert _contains_dated_announcement_language(
        "The device was unveiled years ago and remains a popular reference design."
    ) is False  # "unveiled ... ago", never "unveiled at"


def test_deterministic_guardrail_still_rejects_unsupported_model_claims_after_the_two_newest_fixes():
    assessment_true = _assessment(is_dated_development=True)
    resolved, rationale = _resolve_dated_development(assessment_true, _FakeEvidence(EVERGREEN_TEXT))
    assert resolved is False
    assert "override" in rationale.lower()


# ---------------------------------------------------------------------------
# `today introduced` / `has introduced` (confirmed ActivTrak / IFS Softeon
# forensic finding, CURRENT run `7cd62698e734a392`, 2026-10-13): two more
# real press-release constructions using "introduced" -- a generic
# announcement verb synonymous with launched/unveiled/released, never
# covered in any tense before.
# ---------------------------------------------------------------------------

ACTIVTRAK_TEXT = (
    "AUSTIN, Texas, Oct. 13, 2026 /PRNewswire/ -- ActivTrak today introduced ActivTrak MCP, a new "
    "capability that brings work intelligence directly into the AI platforms, applications and "
    "workflows organizations already use."
)

IFS_SOFTEON_TEXT = (
    "ORLANDO, Fla., 13 October 2026 /PRNewswire/ -- IFS, the leading provider of Industrial AI "
    "software, today introduced Fulfillment Flow from IFS Softeon, a new AI capability designed to "
    "bring context-aware intelligence directly into warehouse fulfillment operations."
)


def test_today_introduced_is_recognized_as_dated_announcement_language():
    assert _contains_dated_announcement_language("The firm today introduced a new platform.") is True


def test_has_introduced_is_recognized_as_dated_announcement_language():
    assert _contains_dated_announcement_language("The firm has introduced a new platform.") is True


def test_activtrak_dated_development_is_preserved_through_the_guardrail():
    assert _contains_dated_announcement_language(ACTIVTRAK_TEXT) is True

    assessment = _assessment(is_dated_development=True)
    resolved, _ = _resolve_dated_development(assessment, _FakeEvidence(ACTIVTRAK_TEXT))

    assert resolved is True


def test_ifs_softeon_dated_development_is_preserved_through_the_guardrail():
    assert _contains_dated_announcement_language(IFS_SOFTEON_TEXT) is True

    assessment = _assessment(is_dated_development=True)
    resolved, _ = _resolve_dated_development(assessment, _FakeEvidence(IFS_SOFTEON_TEXT))

    assert resolved is True


def test_existing_announcement_patterns_still_work_after_the_introduced_fix():
    assert _contains_dated_announcement_language(DATED_BOFA_TEXT) is True
    assert _contains_dated_announcement_language(KONST_TEXT) is True
    assert _contains_dated_announcement_language(SWI_TEXT) is True
    assert _contains_dated_announcement_language(HR_PATH_TEXT) is True
    assert _contains_dated_announcement_language(ASUS_TEXT) is True


def test_unrelated_text_is_not_accidentally_accepted_by_the_introduced_fix():
    """Negative control: a bare 'introduced' with no qualifying
    collocation, and genuinely evergreen text, must still be rejected."""
    assert _contains_dated_announcement_language(EVERGREEN_TEXT) is False
    assert _contains_dated_announcement_language(
        "The product, introduced several years ago, remains a market leader."
    ) is False


def test_deterministic_guardrail_still_rejects_unsupported_model_claims_after_the_introduced_fix():
    assessment_true = _assessment(is_dated_development=True)
    resolved, rationale = _resolve_dated_development(assessment_true, _FakeEvidence(EVERGREEN_TEXT))
    assert resolved is False
    assert "override" in rationale.lower()


# ---------------------------------------------------------------------------
# Generic verb-tense generator (2026-10-13 refactor): rather than keep
# hand-adding one more tense variant each time a new verb/collocation gap
# is found, `_ANNOUNCEMENT_VERB_LEMMAS` + the conjugation helpers generate
# every today/has/is-marked AND bare (present and past) collocation for
# announce/launch/unveil/introduce at once. These tests exercise the
# generator and its two local false-positive guards directly -- the real
# ugee maker (bare present) and CoreWeave (bare past) forensic findings
# are what exposed the two guards' exact shape.
# ---------------------------------------------------------------------------

UGEE_MAKER_TEXT = (
    "HONG KONG, Oct. 12, 2026 /PRNewswire/ -- The consumer 3D printing industry has developed rapidly. "
    "ugee maker introduces a third, easy-to-use, family-friendly option designed for parents, kids, "
    "and beginners. This community provides a collaborative space for shared creativity."
)


def test_bare_present_tense_introduces_is_recognized_without_any_temporal_marker():
    assert _contains_dated_announcement_language(UGEE_MAKER_TEXT) is True


def test_bare_past_tense_announced_is_recognized_without_any_temporal_marker():
    assert _contains_dated_announcement_language(DATED_COREWEAVE_TEXT) is True


def test_every_lemma_gets_full_tense_coverage_automatically():
    """The whole point of the generator: a lemma covered in one tense is
    automatically covered in every other -- no verb can ever again be
    'half-covered' the way 'introduced' briefly was."""
    for lemma, forms in {
        "announce": ("announced", "announces"),
        "launch": ("launched", "launches"),
        "unveil": ("unveiled", "unveils"),
        "introduce": ("introduced", "introduces"),
    }.items():
        past, s_form = forms
        assert _contains_dated_announcement_language(f"The firm today {past} a new product.") is True
        assert _contains_dated_announcement_language(f"The firm {past} today a new product.") is True
        assert _contains_dated_announcement_language(f"The firm has {past} a new product.") is True
        assert _contains_dated_announcement_language(f"The firm today {s_form} a new product.") is True
        assert _contains_dated_announcement_language(f"The firm {s_form} today a new product.") is True


def test_bare_verb_guard_rejects_nearby_recurring_cadence_language():
    """`_RECURRING_CADENCE_RE`'s local-window guard: a cadence word
    actually near the verb still correctly rejects it."""
    assert _contains_dated_announcement_language(
        "The company announces its quarterly results every year in March."
    ) is False


def test_bare_verb_guard_is_not_fooled_by_an_unrelated_distant_cadence_word():
    """Confirmed forensic finding (ugee maker, 2026-10-13): a recurring-
    cadence word FAR AWAY from the actual announcement verb, describing
    something unrelated, must never veto a genuine, nearby one-time
    announcement -- the guard is local, not document-wide."""
    text = (
        "ugee maker introduces a third, easy-to-use option for families. " + ("filler text " * 30) +
        "The platform also offers a weekly newsletter for parents."
    )
    assert _contains_dated_announcement_language(text) is True


def test_bare_past_verb_guard_rejects_historical_distance_language():
    """Confirmed forensic finding: bare past tense ('unveiled'/'introduced')
    is also common EVERGREEN/historical framing -- 'years ago' must veto
    it, distinct from the recurring-cadence guard."""
    assert _contains_dated_announcement_language(
        "The device was unveiled years ago and remains a popular reference design."
    ) is False
    assert _contains_dated_announcement_language(
        "The product, introduced several years ago, remains a market leader."
    ) is False


def test_bare_past_verb_guard_is_not_fooled_by_a_distant_historical_mention():
    text = (
        "The company announced a major new partnership today. " + ("filler text " * 30) +
        "Its flagship product was introduced years ago and remains popular."
    )
    assert _contains_dated_announcement_language(text) is True


def test_bare_verb_guard_rejects_a_future_intention_announcement():
    """Confirmed real-test-fixture false positive (`test_irrelevant_
    consulting_article_is_rejected`, 2026-10-13): 'announced IT WILL host
    a panel' reports a future intention, not a development that has
    actually happened -- bare 'announced' with no object-noun
    restriction now matches it where the old hand-curated noun list
    (launch/availability/release) never did, so this guard is required."""
    assert _contains_dated_announcement_language(
        "McKinsey announced it will host a panel discussion on AI adoption trends at an upcoming "
        "industry conference, with no report, finding, or thesis published alongside the announcement."
    ) is False
    assert _contains_dated_announcement_language(
        "The startup announced it is going to launch a beta program soon."
    ) is False
    # NOTE: "announced that it plans to..." is NOT covered by this guard --
    # the pre-existing, unconditional `announced that` pattern (confirmed
    # KONST forensic finding) fires first and is intentionally unchanged;
    # that pattern's own interaction with future-intention clauses is a
    # separate, pre-existing design choice, out of scope for this fix.


def test_future_intention_guard_does_not_reject_a_genuine_completed_announcement():
    """Regression guard: the future-intention veto must stay narrow --
    it never rejects a real, already-happened announcement just because
    unrelated future-tense language appears elsewhere."""
    text = (
        "The firm announced the acquisition of a rival today. " + ("filler text " * 30) +
        "The combined company will continue to operate under the existing brand."
    )
    assert _contains_dated_announcement_language(text) is True


def test_achieve_and_release_remain_excluded_from_generic_bare_verb_acceptance():
    """Deliberate, reasoned exception: 'achieves'/'releases' bare present
    tense have plausible EVERGREEN uses with no announcement sense at all
    -- they are NOT in `_ANNOUNCEMENT_VERB_LEMMAS` and must stay rejected
    bare, still requiring their existing today/has/is-marked forms."""
    assert _contains_dated_announcement_language("The model achieves 95% accuracy on the benchmark.") is False
    assert _contains_dated_announcement_language("The fund releases quarterly performance reports.") is False
    # Their existing, hand-written, explicitly-marked forms still work.
    assert _contains_dated_announcement_language("The model today achieved a new benchmark record.") is True
    assert _contains_dated_announcement_language("The fund has released its latest report.") is True


# ---------------------------------------------------------------------------
# PARTNERSHIP (tests 5-10)
# ---------------------------------------------------------------------------


def test_5_no_structured_partnerships_strips_unsupported_partnership_sentence():
    text = "The company expanded its platform. This partnership demonstrates deep technical alignment."
    result = _sanitize_partnership_claims(
        text, validated_partners=[], known_companies=["Bank of America"], field="opening_context",
    )

    assert "partnership" not in result.lower()
    assert "expanded its platform" in result


def test_6_available_through_aws_is_never_a_validated_partnership():
    validated = _validate_partnerships_against_evidence(["AWS"], COHERE_AVAILABILITY_TEXT.replace("Amazon SageMaker", "AWS"))
    assert validated == []


def test_7_available_on_amazon_sagemaker_is_never_a_validated_partnership():
    validated = _validate_partnerships_against_evidence(["Amazon SageMaker"], COHERE_AVAILABILITY_TEXT)
    assert validated == []
    validated_msft = _validate_partnerships_against_evidence(["Microsoft Foundry"], COHERE_AVAILABILITY_TEXT)
    assert validated_msft == []


def test_8_named_unsupported_partner_cannot_survive_in_the_report():
    # `known_companies` here mirrors what `_aggregate_facts`' `other_companies`
    # would contain -- entity-derived companies ONLY, never the primary
    # monitored company (NVIDIA) itself.
    text = "NVIDIA announced its partnership with CoreWeave and FrontierCode."
    result = _sanitize_partnership_claims(
        text, validated_partners=["CoreWeave"],
        known_companies=["CoreWeave", "FrontierCode"], field="details_narrative",
    )

    # FrontierCode is not validated, so the whole sentence (which also
    # names it) is dropped rather than left half-true.
    assert "FrontierCode" not in result
    assert result == "The evidence establishes limited detail beyond the structured facts above."


def test_9_a_validated_partnership_fact_survives_the_report_and_evidence_guardrails():
    validated = _validate_partnerships_against_evidence(["CoreWeave"], DATED_COREWEAVE_TEXT)
    assert validated == ["CoreWeave"]

    text = "NVIDIA's close partnership with CoreWeave demonstrates deep technical alignment."
    result = _sanitize_partnership_claims(
        text, validated_partners=["CoreWeave"], known_companies=["CoreWeave"], field="strategic_implications",
    )
    assert "CoreWeave" in result
    assert "partnership" in result.lower()


def test_plural_partners_is_also_caught_real_e2e_regression():
    """Regression for a gap found during the Phase 13C live E2E run: the
    plural possessive "its partners'" slipped past an earlier, narrower
    regex (which only matched partner/partnered/partnership/partnering)."""
    text = "Given NVIDIA's and its partners' involvement in AI technologies, this could impact the landscape."
    result = _sanitize_partnership_claims(
        text, validated_partners=[], known_companies=[], field="strategic_implications",
    )
    assert "partners" not in result.lower()


def test_10_qualified_non_partnership_language_is_left_untouched():
    text = "This could enable new retrieval workflows for enterprise teams evaluating RAG infrastructure."
    result = _sanitize_partnership_claims(
        text, validated_partners=[], known_companies=["Cohere"], field="strategic_implications",
    )
    assert result == text


def test_a_long_corporate_suffix_subject_list_does_not_falsely_split_before_the_verb():
    """Real, confirmed regression (Gemtek/Qualcomm/gFiber event,
    2026-10-15): a naive `(?<=[.!?])\\s+` sentence split treats the
    period in "...and gFiber Inc. announced..." as a sentence boundary
    (the commas after "Co."/"Ltd." happen to avoid a false split there,
    but nothing stopped the final "Inc." right before the verb). The
    subject-list fragment then survives as its own "sentence" (no
    partnership-language match), while the real verb clause becomes a
    separate "sentence" that DOES match the partnership guardrail and
    gets dropped -- leaving a dangling, verb-less fragment as the ENTIRE
    persisted report field. With no validated partners, the guardrail is
    correct to drop the partnership-framed clause; what must never happen
    is the leftover subject list being mistaken for a complete field."""
    text = (
        "On October 13, 2026, Gemtek Technology Co., Ltd., Qualcomm Technologies, Inc., "
        "and gFiber Inc. announced a collaboration to bring Edge AI capabilities to "
        "Wi-Fi 8 broadband gateways."
    )
    result = _sanitize_partnership_claims(
        text, validated_partners=[],
        known_companies=["Gemtek Technology Co., Ltd.", "Qualcomm Technologies, Inc.", "gFiber Inc."],
        field="opening_context",
    )

    # The whole sentence is one unit (the comma-studded corporate names
    # never caused a false split) and it IS partnership-framed with no
    # validated partner, so dropping it entirely -- down to the neutral
    # fallback -- is correct. What the bug produced instead was a
    # dangling, verb-less fragment ending at "...gFiber Inc."
    assert not result.rstrip().endswith("gFiber Inc.")
    assert result == "The evidence establishes limited detail beyond the structured facts above."


# ---------------------------------------------------------------------------
# Phase 13F REPORT QUALITY: inflated, unsupported editorializing phrases are
# removed surgically (phrase-level, not sentence-drop), preserving the
# real, evidence-grounded fact alongside the stripped clause.
# ---------------------------------------------------------------------------


def test_unsupported_inflated_phrase_is_removed_but_the_factual_clause_survives():
    text = "NVIDIA, a key player in AI infrastructure, today announced a new GPU platform for enterprise customers."
    result = _strip_inflated_language(text)

    assert "key player" not in result.lower()
    assert "today announced a new GPU platform for enterprise customers" in result


def test_bare_inflated_adjective_is_removed_without_dropping_the_sentence():
    text = "The firm unveiled a groundbreaking agentic AI platform for capital markets research."
    result = _strip_inflated_language(text)

    assert "groundbreaking" not in result.lower()
    assert "agentic AI platform for capital markets research" in result


def test_evidence_grounded_statement_with_no_inflated_language_is_preserved_verbatim():
    text = "JPMorgan Chase today launched a generative AI copilot for financial advisors."
    assert _strip_inflated_language(text) == text


def test_multiple_inflated_phrases_in_one_field_are_all_removed():
    text = "This is a major market opportunity. The company is an industry leader in enterprise AI deployment."
    result = _strip_inflated_language(text)

    assert "major market opportunity" not in result.lower()
    assert "industry leader" not in result.lower()
    assert "enterprise AI deployment" in result


def test_inflated_phrase_followed_by_of_does_not_leave_a_dangling_article():
    """Confirmed production regression (2026-10-15, real NVIDIA/France
    event): the clause-removal path only matched a phrase followed by
    "in" ("a leading provider IN AI"), so the very common "a leading
    provider OF AI..." phrasing fell through to the bare-word removal
    and left a grammatically broken dangling article: "NVIDIA, a of AI
    and high-performance computing solutions, announced...". The whole
    appositive clause must be removed together, regardless of which
    word (or none) connects the phrase to its object."""
    text = (
        "NVIDIA, a leading provider of AI and high-performance computing solutions, "
        "announced a significant development related to the CANUT framework agreement."
    )
    result = _strip_inflated_language(text)

    assert "leading provider" not in result.lower()
    assert ", a of " not in result
    assert " a of " not in result
    assert result == "NVIDIA announced a significant development related to the CANUT framework agreement."


# ---------------------------------------------------------------------------
# Full analyze() integration (not just the bare functions)
# ---------------------------------------------------------------------------


def test_analyze_applies_the_date_guardrail_through_the_real_pipeline(config, peer_source):
    assessment = _assessment(is_dated_development=True, relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="x"))
    llm = FakeLLMClient([assessment])
    agent = PeerAgent(llm, config)
    extracted = ExtractedArticle(url="https://example.com/a", title="Claude Sonnet", text=EVERGREEN_TEXT + " generative AI")
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.is_dated_development is False  # overridden despite the LLM saying true


def test_swi_group_text_through_the_real_pipeline_preserves_is_dated_development_true(config, peer_source):
    """Full real pipeline (agent.analyze -> ResearchItem), proving the fix
    end to end, not just at the bare guardrail function: the model's
    is_dated_development=true verdict on the real SWI text survives
    through the production agent code path."""
    assessment = _assessment(is_dated_development=True, relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="x"))
    llm = FakeLLMClient([assessment])
    agent = PeerAgent(llm, config)
    extracted = ExtractedArticle(
        url="https://example.com/swi", title="SWI Group announces capital increase to expand transatlantic AI infrastructure",
        text=SWI_TEXT, published_at=_SWI_PUBLISHED_AT,
    )
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.is_dated_development is True  # no longer incorrectly narrowed to False
    assert item.publication_date is not None


def test_swi_group_event_date_propagates_through_the_real_event_builder(config, peer_source):
    """Proves the corrected classification actually reaches `Event.event_date`
    via the UNCHANGED, existing `events.builder._select_event_date` logic
    -- never a fabricated or substituted date, exactly the existing
    publication_date-of-a-dated-development rule."""
    from research_assistant.events.builder import build_event_draft
    from research_assistant.events.clustering import Cluster
    from research_assistant.events.models import ClusterMetadata

    assessment = _assessment(is_dated_development=True, relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="x"))
    llm = FakeLLMClient([assessment])
    agent = PeerAgent(llm, config)
    extracted = ExtractedArticle(
        url="https://example.com/swi2", title="SWI Group announces capital increase to expand transatlantic AI infrastructure",
        text=SWI_TEXT, published_at=_SWI_PUBLISHED_AT,
    )
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)
    item = agent.analyze(article, evidence)
    assert item is not None and item.is_dated_development is True

    draft = build_event_draft(Cluster(items=[item], metadata=ClusterMetadata(member_count=1)))

    assert draft.event_date == item.publication_date  # real propagation, not a fabricated date
    assert draft.event_date == _SWI_PUBLISHED_AT


def test_analyze_applies_the_partnership_guardrail_through_the_real_pipeline(config, peer_source):
    assessment = _assessment(
        is_dated_development=True,
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="x"),
        partnerships=["AWS"],
    )
    llm = FakeLLMClient([assessment])
    agent = PeerAgent(llm, config)
    text = DATED_BOFA_TEXT + " The new tool is available through AWS. generative AI"
    extracted = ExtractedArticle(url="https://example.com/b", title="Bank announcement", text=text)
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.partnerships == []  # "available through AWS" never becomes a validated partnership


# ---------------------------------------------------------------------------
# Part 4: typed entities feed report facts without becoming partnerships
# ---------------------------------------------------------------------------


def test_aggregate_facts_folds_typed_entities_without_creating_partnership_claims():
    item = make_item(
        "a1", company="Anthropic", title="t",
        entities=[
            TypedEntity(name="Claude.ai", type=EntityType.PRODUCT),
            TypedEntity(name="Claude Platform", type=EntityType.PLATFORM),
            TypedEntity(name="Amazon Web Services", type=EntityType.COMPANY),
            TypedEntity(name="Agentic AI", type=EntityType.TECHNOLOGY),
        ],
        partnerships=[],
    )

    facts = _aggregate_facts([item])

    assert "Claude.ai" in facts["products"]
    assert "Claude Platform" in facts["products"]
    assert "Agentic AI" in facts["technologies"]
    assert "Amazon Web Services" in facts["companies"]
    assert "Amazon Web Services" in facts["other_companies"]
    assert "Anthropic" not in facts["other_companies"]  # the primary company is never "an other company"
    assert facts["partnerships"] == []  # entities never leak into partnerships


def test_inflated_phrase_with_no_trailing_comma_does_not_leave_a_dangling_article_before_a_preposition():
    """Real, confirmed regression #2 (2026-10-15, NVIDIA/France report
    regeneration test run): the comma-bounded clause-removal alternative
    only fires on a comma-delimited appositive. A phrase at the end of a
    clause with NO trailing comma ("...which is a leading player in the
    AGENTIC computing space.") falls to the bare-word alternative, which
    strips only the phrase and leaves the exact same class of defect as
    the first dangling-article bug, minus the comma: "...which is a in
    the AGENTIC computing space."."""
    text = "This is especially notable, as the company is a leading player in the AGENTIC computing space."
    result = _strip_inflated_language(text)

    assert " a in " not in result
    assert "is in the AGENTIC computing space" in result


def test_dangling_article_fix_never_touches_a_hyphenated_compound_like_for_profit():
    text = "This is a for-profit company exploring AI tools."
    assert _strip_inflated_language(text) == text


# ---------------------------------------------------------------------------
# Confirmed production regression #3 (2026-10-15, real NVIDIA/France event,
# event_id 9fe7478a8ff77eb5, zero validated partnerships): each of 4
# key_insights independently mentioned a different unvalidated company, so
# each independently tripped the partnership guardrail and each collapsed
# to the IDENTICAL literal fallback string -- "The evidence establishes
# limited detail beyond the structured facts above." appeared 3 times in
# the rendered newsletter. Separately, strategic_implications/
# nomura_opportunity (each effectively one sentence, inevitably using
# collaboration language for an article framed around a multi-company
# agreement) collapsed to the bare, context-free generic hedge even
# though real, validated, non-partnership facts (products/technologies/
# geography) existed and could ground a more specific fallback.
# ---------------------------------------------------------------------------

from research_assistant.agents.report import (  # noqa: E402
    _fallback_with_facts,
    _grounded_fallback_clause,
    _sanitize_key_insights,
    _sanitize_narrative,
)
from research_assistant.agents.schemas import ReportNarrative  # noqa: E402

_NVIDIA_FRANCE_FACTS = {
    "companies": ["NVIDIA"],
    "other_companies": [
        "Alice&Bob", "Altair-1", "Bull", "Dassault Systèmes", "Docaposte", "Doctolib",
        "Entalpic", "Fondation Inria", "Genesis AI", "Gradium", "LINAGORA", "Loft Orbital",
        "Marlan Space", "Mistral", "Orbitworks", "Pigment", "SBS", "Scintil Photonics",
    ],
    "people": [],
    "products": ["NVIDIA Jetson AGX Orin"],
    "technologies": ["open-source models"],
    "partnerships": [],  # nothing validated -- the real-world case
    "geography": ["France"],
    "deal_value": None,
}


def test_key_insights_with_multiple_independently_unsupported_items_never_duplicate_the_fallback_line():
    narrative_insights = [
        "NVIDIA expanded its AGX computing capabilities with the release of the NVIDIA Jetson AGX Orin and open-source models.",
        "NVIDIA's collaboration with Mistral strengthens its position in the French AI ecosystem.",
        "The partnership with Dassault Systèmes signals deeper integration into industrial AI workflows.",
        "Working with Bull and LINAGORA, NVIDIA is expanding its collaborative footprint in sovereign AI infrastructure.",
    ]

    result = _sanitize_key_insights(
        narrative_insights,
        validated_partners=_NVIDIA_FRANCE_FACTS["partnerships"],
        known_companies=_NVIDIA_FRANCE_FACTS["other_companies"],
        facts=_NVIDIA_FRANCE_FACTS,
    )

    # The one genuinely grounded, non-partnership insight survives verbatim.
    assert any("Jetson AGX Orin" in item for item in result)
    # No fallback text is duplicated -- each distinct string appears once.
    assert len(result) == len(set(result))
    # The three partnership-framed insights were dropped, not replaced
    # one-for-one with a repeated fallback line.
    assert result.count("The evidence establishes limited detail beyond the structured facts above.") == 0


def test_key_insights_falls_back_to_exactly_one_fact_grounded_line_when_every_item_is_unsupported():
    narrative_insights = [
        "NVIDIA's partnership with Mistral deepens its French ecosystem ties.",
        "Collaboration with Bull signals a new phase of sovereign AI cooperation.",
    ]

    result = _sanitize_key_insights(
        narrative_insights,
        validated_partners=[], known_companies=_NVIDIA_FRANCE_FACTS["other_companies"],
        facts=_NVIDIA_FRANCE_FACTS,
    )

    assert len(result) == 1
    assert "Jetson AGX Orin" in result[0] or "open-source models" in result[0]
    assert result[0] != "The evidence establishes limited detail beyond the structured facts above."


def test_grounded_fallback_clause_never_cites_companies_or_partnerships():
    """The primary company's own name may legitimately appear as part of a
    product name it ships (e.g. "NVIDIA Jetson AGX Orin") -- the guarantee
    this clause exists for is that no OTHER (unvalidated) company name is
    ever cited, since that is the exact relationship-implication risk the
    partnership guardrail exists to avoid."""
    clause = _grounded_fallback_clause(_NVIDIA_FRANCE_FACTS)

    assert clause is not None
    for other_company in _NVIDIA_FRANCE_FACTS["other_companies"]:
        assert other_company not in clause
    assert "Jetson AGX Orin" in clause or "France" in clause or "open-source models" in clause


def test_grounded_fallback_clause_is_none_when_facts_has_nothing_citable():
    assert _grounded_fallback_clause({}) is None
    assert _grounded_fallback_clause(
        {"products": [], "technologies": [], "geography": [], "deal_value": None}
    ) is None


def test_fallback_with_facts_prefers_fact_grounded_text_over_the_bare_generic_hedge():
    strategic = _fallback_with_facts("strategic_implications", _NVIDIA_FRANCE_FACTS)
    opportunity = _fallback_with_facts("nomura_opportunity", _NVIDIA_FRANCE_FACTS)

    assert strategic != "The evidence does not support a specific strategic interpretation beyond the stated facts."
    assert opportunity != "This development may warrant monitoring as more detail becomes available."
    assert "Jetson AGX Orin" in strategic or "France" in strategic or "open-source models" in strategic


def test_fallback_with_facts_uses_the_generic_hedge_when_no_facts_exist_tier_b_style():
    assert _fallback_with_facts("strategic_implications", {}) == (
        "The evidence does not support a specific strategic interpretation beyond the stated facts."
    )


def test_sanitize_narrative_end_to_end_reproduces_the_real_nvidia_fix():
    """Full reproduction of the real 2026-10-15 regression using the
    actual persisted NVIDIA/France facts shape, through the real
    `_sanitize_narrative` entry point (not just the inner helpers)."""
    narrative = ReportNarrative(
        opening_context=(
            "On October 13, 2026, NVIDIA announced the CANUT framework agreement, a collaborative "
            "initiative involving multiple companies in the AI and industrial automation sectors, "
            "including the release of the NVIDIA Jetson AGX Orin and open-source models."
        ),
        details_narrative="NVIDIA's collaboration spans several French AI startups.",
        key_insights=[
            "NVIDIA expanded its AGX computing capabilities with the release of the NVIDIA Jetson AGX Orin and open-source models.",
            "NVIDIA's collaboration with Mistral strengthens its position in the French AI ecosystem.",
            "The partnership with Dassault Systèmes signals deeper integration into industrial AI workflows.",
            "Working with Bull and LINAGORA, NVIDIA is expanding its collaborative footprint in sovereign AI infrastructure.",
        ],
        strategic_implications=(
            "NVIDIA's partnership with multiple French startups may strengthen its ecosystem position in Europe."
        ),
        nomura_opportunity=(
            "This collaboration with French AI startups may be relevant to Nomura's consideration of "
            "European AI investment opportunities."
        ),
    )

    sanitized = _sanitize_narrative(narrative, _NVIDIA_FRANCE_FACTS)

    assert len(sanitized.key_insights) == len(set(sanitized.key_insights))  # no duplicate lines
    assert sanitized.strategic_implications != (
        "The evidence does not support a specific strategic interpretation beyond the stated facts."
    )
    assert sanitized.nomura_opportunity != "This development may warrant monitoring as more detail becomes available."


# ---------------------------------------------------------------------------
# Real regression (2026-10-19, HKU/GCSA Centre for AI and Cybersecurity
# Research): `opening_context` described the announcement with
# partnership-framed language ("jointly establish..."), which the
# partnership guardrail dropped outright (this item had no validated
# `partnerships_json` entry) -- falling all the way back to the bare,
# uninformative fact clause ("The confirmed facts include Centre for AI
# and Cybersecurity Research, Artificial Intelligence (AI), Cybersecurity,
# Hong Kong..."). Meanwhile `details_narrative` described the EXACT SAME
# announcement with non-partnership-framed phrasing ("...jointly announced
# the establishment of...") and survived with real, substantive content --
# but the UI only ever surfaced `opening_context`'s empty hedge, discarding
# the real content sitting right next to it in the same Report.
# `_reuse_sibling_on_fallback` fixes this: when a field falls back to its
# own generic hedge, an independently-survived sibling's real text is
# surfaced instead.
# ---------------------------------------------------------------------------

_HKU_FACTS = {
    "companies": ["Global Cybersecurity Alliance (GCSA)", "PR Newswire", "University of Hong Kong (HKU)"],
    "other_companies": ["Global Cybersecurity Alliance (GCSA)", "University of Hong Kong (HKU)"],
    "people": ["Professor Ho Chen"],
    "products": ["Centre for AI and Cybersecurity Research"],
    "technologies": ["Artificial Intelligence (AI)", "Cybersecurity"],
    "partnerships": [],  # nothing validated -- the real-world case
    "geography": ["Hong Kong"],
    "deal_value": None,
}


def test_sanitize_narrative_reuses_details_narrative_when_opening_context_falls_back():
    narrative = ReportNarrative(
        opening_context=(
            "HKU and GCSA have partnered to jointly establish the Centre for AI and Cybersecurity Research."
        ),
        details_narrative=(
            "HKU and GCSA jointly announced the establishment of the Centre for AI and Cybersecurity "
            "Research, a center designed to support both academic research and practical industry "
            "applications in AI and cybersecurity."
        ),
        key_insights=[
            "The Centre for AI and Cybersecurity Research marks the first joint effort between HKU and "
            "GCSA in the field of AI and cybersecurity.",
            "The establishment of the Centre represents a significant step towards leveraging AI for "
            "enhancing cybersecurity, potentially influencing broader industry practices and academic "
            "research agendas.",
        ],
        strategic_implications=(
            "This partnership between HKU and GCSA may strengthen cross-institutional collaboration in "
            "AI-driven cybersecurity research."
        ),
        nomura_opportunity=(
            "This development may be relevant to Nomura's consideration of potential opportunities in "
            "cross-disciplinary research and the intersection of AI and cybersecurity."
        ),
    )

    sanitized = _sanitize_narrative(narrative, _HKU_FACTS)

    # Both opening_context and strategic_implications, left to their own
    # sanitization, would fall all the way back to the bare fact clause --
    # confirming the regression is real before checking the fix.
    bare_fallback = _fallback_with_facts("opening_context", _HKU_FACTS)
    assert sanitized.opening_context != bare_fallback
    assert "jointly announced the establishment" in sanitized.opening_context
    assert sanitized.strategic_implications != _fallback_with_facts("strategic_implications", _HKU_FACTS)
    assert "joint effort between HKU and GCSA" in sanitized.strategic_implications


# ---------------------------------------------------------------------------
# Real regression (2026-10-15, follow-up): a fresh real regeneration of the
# same NVIDIA/France event showed "The evidence establishes limited detail
# beyond the structured facts above." as ONE item in a 4-item key_insights
# list alongside 3 genuine, specific insights. `_sanitize_key_insights`
# cannot inject this text unless the whole list is empty (confirmed:
# `_sanitize_insight_item` never returns this string, only "" or the
# sentence-stripped original) -- so this can only be the raw LLM output
# itself echoing the prompt's own "say plainly that little beyond the
# stated facts is known" instruction verbatim, even though 3 other genuine
# insights were ALSO produced in the same response. The sanitizer correctly
# leaves this sentence untouched (it contains no partnership language, so
# nothing strips it) -- this is a PROMPT problem, not a sanitizer bug: the
# prompt now explicitly says this item is only for when fewer than two real
# insights can be produced, never alongside two or more genuine ones.
# This test documents that the sanitizer, even given this exact literal
# input, never ADDS a second copy of it when real insights are present --
# confirming the sanitizer side of the fix held, while the prompt copy
# above was tightened to stop the model emitting it unnecessarily.
# ---------------------------------------------------------------------------


def test_sanitizer_never_duplicates_a_generic_filler_item_the_model_wrote_itself():
    narrative_insights = [
        "The evidence establishes limited detail beyond the structured facts above.",
        "NVIDIA has launched new AI technologies, such as NVIDIA Jetson AGX Orin and NVIDIA Vera Rubin Ultra NVL72, which are part of the CANUT framework.",
        "The CANUT framework, along with the new AI technologies, will likely influence the development and deployment of AI solutions in various sectors in Europe and France.",
        "This announcement highlights NVIDIA's role in fostering a vibrant ecosystem of AI technology and its commitment to expanding its reach across different regions and companies.",
    ]

    result = _sanitize_key_insights(
        narrative_insights,
        validated_partners=[], known_companies=_NVIDIA_FRANCE_FACTS["other_companies"],
        facts=_NVIDIA_FRANCE_FACTS,
    )

    # The sanitizer passes through every item unchanged (none contain
    # partnership language) -- it never adds a SECOND copy of the filler
    # line on top of what the model already wrote once.
    assert result.count("The evidence establishes limited detail beyond the structured facts above.") == 1
    assert len(result) == 4


# ---------------------------------------------------------------------------
# Tier B grounded-fallback (2026-10-15): Tier B has no structured `facts`
# dict (no ResearchItem/Event was ever validated for it), so
# `_fallback_with_facts` always returned the bare generic hedge for
# `strategic_implications`/`nomura_opportunity` even when the article's own
# `opening_context` (already sanitized, grounded only in the article text)
# was substantive. `_tier_b_fallback_with_context` fixes this by citing the
# already-sanitized opening_context instead, the Tier-B-appropriate
# equivalent of Tier A's fact-grounded fallback.
# ---------------------------------------------------------------------------

from research_assistant.agents.report import (  # noqa: E402
    _sanitize_tier_b_narrative,
    _tier_b_fallback_with_context,
)
from research_assistant.agents.schemas import TierBReportNarrative  # noqa: E402


def test_tier_b_fallback_cites_the_sanitized_opening_context_not_a_bare_hedge():
    fallback = _tier_b_fallback_with_context(
        "strategic_implications",
        "d-Matrix announced Squadron compute clusters for AI inference at data center scale.",
    )
    assert fallback != "The evidence does not support a specific strategic interpretation beyond the stated facts."
    assert "described above" in fallback


def test_tier_b_fallback_uses_the_bare_generic_hedge_when_opening_context_itself_is_empty():
    assert _tier_b_fallback_with_context("strategic_implications", "") == (
        "The evidence does not support a specific strategic interpretation beyond the stated facts."
    )


def test_tier_b_fallback_uses_the_bare_generic_hedge_when_opening_context_is_itself_the_fallback():
    bare = "The evidence establishes limited detail beyond the structured facts above."
    assert _tier_b_fallback_with_context("nomura_opportunity", bare) == (
        "This development may warrant monitoring as more detail becomes available."
    )


def test_sanitize_tier_b_narrative_end_to_end_uses_context_grounded_fallback():
    narrative = TierBReportNarrative(
        opening_context="d-Matrix announced Squadron compute clusters for AI inference at data center scale.",
        key_insights=[
            "Squadron clusters run entirely on Corsair XPUs, a memory-centric compute architecture.",
            "The system is now generally available and deployed in d-Matrix's Arizona data center.",
        ],
        strategic_implications=(
            "d-Matrix's partnership with its data center customers signals a new phase of enterprise collaboration."
        ),
        nomura_opportunity=(
            "This collaboration with enterprise customers may be relevant to Nomura's consideration of AI "
            "infrastructure investment opportunities."
        ),
    )

    sanitized = _sanitize_tier_b_narrative(narrative)

    # Both fields got dropped by the partnership guardrail (Tier B has no
    # validated partners at all) and fell back -- but to the context-
    # grounded text, not the bare hedge, since opening_context survived.
    assert sanitized.strategic_implications != (
        "The evidence does not support a specific strategic interpretation beyond the stated facts."
    )
    assert sanitized.nomura_opportunity != "This development may warrant monitoring as more detail becomes available."
    assert "described above" in sanitized.strategic_implications
    assert "described above" in sanitized.nomura_opportunity
