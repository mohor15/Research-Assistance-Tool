"""CURRENT's bounded LLM funnel (RETRIEVAL -> deterministic gate -> batched
triage -> bounded deep analysis -> existing EventBuilder/Report/Digest).
No live Ollama/network anywhere -- FakeLLMClient/FakeTriageLLMClient only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.agents.ai_product import AIProductAgent
from research_assistant.agents.financial_news import FinancialNewsAgent
from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.report import ReportAgent
from research_assistant.agents.triage import TriageCandidate
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.evidence.conversion import article_to_evidence
from research_assistant.evidence.normalizer import normalize_article
from research_assistant.llm.errors import LLMUnavailableError
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, SourcePriority, SourceType
from research_assistant.pipeline import PipelineServices, PipelineState, build_graph
from research_assistant.pipeline.nodes import (
    _build_current_candidate_pool,
    _passes_current_deterministic_gate,
    _run_current_deep_analysis,
    _run_current_triage,
)
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import (
    ExtractedArticle,
    ExtractionMethod,
    ExtractionStatus,
    SourceCategory,
    SourceDefinition,
)

from tests.events.conftest import make_item
from tests.pipeline.conftest import (
    D,
    FakeRetrievalService,
    FakeTriageLLMClient,
    jpmorgan_source,
    real_normalization_service,
)
from tests.pipeline.test_graph import FakeLLMClient, _relevant_assessment
from tests.reporting.conftest import FakeLLMClient as ReportingFakeLLMClient
from tests.reporting.conftest import make_narrative


def _article_and_evidence(*, url, title, text, published_at=D - timedelta(hours=1)):
    extracted = ExtractedArticle(url=url, title=title, text=text, published_at=published_at)
    article = normalize_article(extracted, source=jpmorgan_source())
    evidence = article_to_evidence(article)
    return article, evidence


def _ai_providers_source(name: str) -> SourceDefinition:
    return SourceDefinition(
        name=name, domain=f"{name.lower().replace(' ', '')}.example.com",
        urls=[f"https://{name.lower().replace(' ', '')}.example.com/ai"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.AI_PROVIDER,
    )


def _peer_source(name: str) -> SourceDefinition:
    return SourceDefinition(
        name=name, domain=f"{name.lower().replace(' ', '')}.example.com",
        urls=[f"https://{name.lower().replace(' ', '')}.example.com/newsroom"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )


def _article(name: str, i: int, *, title_suffix: str = "Bank announces major AI partnership") -> ExtractedArticle:
    return ExtractedArticle(
        url=f"https://{name.lower().replace(' ', '')}.example.com/story-{i}",
        title=f"{name} story {i}: {title_suffix}",
        text=(
            f"{name} reported today (story {i}) that it announced a new enterprise AI "
            f"partnership and deployed a generative AI platform, in development number {i}. " * 3
        ),
        published_at=D - timedelta(hours=1),
    )


# ---------------------------------------------------------------------------
# 1/2. Deterministic gate: structural-only, never keyword/relevance-based
# ---------------------------------------------------------------------------


def test_deterministic_gate_rejects_empty_title_and_thin_hub_pages(config):
    empty_article, ev1 = _article_and_evidence(
        url="https://jpmorganchase.com/a1", title="", text="substantive body " * 100,
    )
    assert _passes_current_deterministic_gate(empty_article, ev1) is False

    # A pipe title combined with a THIN body is a real hub/nav shell (the
    # real NVIDIA "News Archive" hub page extracted to only 349 chars).
    thin_hub_article, ev2 = _article_and_evidence(
        url="https://jpmorganchase.com/a2", title="JPMorgan Newsroom | Press Releases", text="too short",
    )
    assert _passes_current_deterministic_gate(thin_hub_article, ev2) is False

    thin_article, ev3 = _article_and_evidence(
        url="https://jpmorganchase.com/a3", title="A real headline about something", text="too short",
    )
    assert _passes_current_deterministic_gate(thin_article, ev3) is False

    genuine_article, ev4 = _article_and_evidence(
        url="https://jpmorganchase.com/a4", title="A real headline about something specific",
        text="substantive body " * 100,
    )
    assert _passes_current_deterministic_gate(genuine_article, ev4) is True


def test_deterministic_gate_does_not_hard_reject_a_substantive_pipe_titled_article(config):
    """Confirmed regression (2026-10-14): a real NatWest press release
    extracted with title "...with NVIDIA Nemotron | NatWest Group" (the
    standard "Headline | Site Name" <title> pattern most news sites use)
    and a genuine 6,127-character body was being hard-excluded by the
    gate on the pipe alone, with zero diagnostic trail. A pipe title must
    only demote (via `_candidate_rank_key`), never hard-exclude, once the
    body is actually substantive."""
    article, evidence = _article_and_evidence(
        url="https://natwestgroup.example.com/a7",
        title="Bank improves AI speech recognition for customers | Bank Group",
        text="A real, substantive press release about a genuine AI deployment. " * 20,
    )
    assert _passes_current_deterministic_gate(article, evidence) is True


def test_deterministic_gate_does_not_check_keywords_or_decide_relevance(config):
    """A candidate with NO AI/financial-services keywords anywhere, but a
    substantive non-hub body, must still PASS the gate -- relevance is
    triage's job, not the gate's."""
    body = "This article discusses gardening, weather patterns, and local sports results. " * 10
    article, evidence = _article_and_evidence(
        url="https://jpmorganchase.com/a5", title="A completely unrelated local news story", text=body,
    )
    assert _passes_current_deterministic_gate(article, evidence) is True


# Forensic regression (CURRENT run `31fa192e4d7a7175`, 2026-10-12): a real
# Financial Times candidate's extracted body was nothing but subscription-
# paywall marketing copy -- no article prose was ever retrieved -- yet it
# cleared the substantive-body-length check and consumed a full triage
# slot plus a full deep-analysis call for content that was never the real
# article. Fixture text is a trimmed, verbatim copy of the real persisted
# `evidence.content` for that candidate.
_REAL_FT_PAYWALL_BODY = (
    "The huge lurking leverage of AI hyperscalers\n"
    "Subscribe to unlock this article\n"
    "Save 50% on Standard Digital\n"
    "was undefined now undefined for your first year. Save now on essential digital access to trusted "
    "FT journalism on any device. Savings based on annual price.\n"
    "Explore more offers.\n"
    "Trial\n"
    "undefined for 4 weeks\n"
    "Then undefined per month. Complete digital access with exclusive insights and industry deep dives "
    "on any device. Cancel anytime during your trial.\n"
    "FT Digital Edition\n"
    "Save 50%\n"
    "undefined for your first 6 months\n"
    "Then undefined per 3 months. Save now on FT Digital Edition: all the content editors have selected "
    "for the print edition, accessible on any device.\n"
    "Premium Digital\n"
    "Complete coverage\n"
    "undefined per month\n"
    "Complete digital access with exclusive insights and industry deep dives on any device.\n"
    "Check whether you already have free access via your university or organisation.\n"
    "Terms & Conditions apply\n"
    "Explore our full range of subscriptions.\n"
    "For individuals\n"
    "Discover all the plans currently available in your country\n"
    "For multiple readers\n"
    "Digital access for organisations. Includes exclusive features and content.\n"
)


def test_deterministic_gate_rejects_paywall_boilerplate_that_was_never_the_real_article(config):
    article, evidence = _article_and_evidence(
        url="https://ft.example.com/content/17a207f4-1563-48af-91e3-14bac89ab4b8",
        title="The huge lurking leverage of AI hyperscalers", text=_REAL_FT_PAYWALL_BODY,
    )
    assert _passes_current_deterministic_gate(article, evidence) is False


def test_deterministic_gate_does_not_reject_a_real_article_with_one_incidental_subscribe_mention():
    """A single, incidental 'subscribe to our newsletter'-style footer link
    on an otherwise real, substantive article must never alone reject real
    content -- the paywall check requires at least two independent
    boilerplate-phrase matches, never a single mention."""
    body = (
        "A bank today announced a new generative AI platform for its wealth management division, "
        "built in partnership with a leading AI infrastructure provider. " * 6
        + "Subscribe to our newsletter for more updates."
    )
    article, evidence = _article_and_evidence(
        url="https://jpmorganchase.com/a6", title="Bank launches new AI wealth management platform", text=body,
    )
    assert _passes_current_deterministic_gate(article, evidence) is True


# ---------------------------------------------------------------------------
# 3/4/5. Triage batching, batch size, max_triage_batches
# ---------------------------------------------------------------------------


def _triage_candidate(candidate_id: str):
    article, evidence = _article_and_evidence(
        url=f"https://jpmorganchase.com/{candidate_id}",
        title=f"JPMorgan Chase story {candidate_id}: AI partnership announced",
        text="JPMorgan Chase today announced a new AI partnership and deployed a GenAI platform. " * 10,
    )
    return TriageCandidate(candidate_id=article.id, article=article, evidence=evidence, primary_agent=AgentName.PEER)


def test_triage_batches_respect_configured_batch_size():
    pool = [_triage_candidate(f"c{i}") for i in range(5)]
    client = FakeTriageLLMClient()
    diagnostics = []

    winners, metrics = _run_current_triage(
        pool, triage_client=client, batch_size=2, max_batches=10, diagnostics=diagnostics,
    )

    assert metrics["triage_batches_attempted"] == 3  # 2 + 2 + 1
    assert len(client.calls) == 3
    assert metrics["triage_candidates"] == 5
    assert metrics["triage_relevant"] == 5
    assert len(winners) == 5


def test_max_triage_batches_bounds_total_candidates_sent():
    pool = [_triage_candidate(f"c{i}") for i in range(20)]
    client = FakeTriageLLMClient()
    diagnostics = []

    winners, metrics = _run_current_triage(
        pool, triage_client=client, batch_size=8, max_batches=3, diagnostics=diagnostics,
    )

    assert metrics["candidates_sent_to_triage"] == 20  # 20 < 8*3=24, all sent
    assert metrics["triage_batches_attempted"] == 3  # ceil(20/8)

    bigger_pool = [_triage_candidate(f"d{i}") for i in range(30)]
    winners2, metrics2 = _run_current_triage(
        bigger_pool, triage_client=client, batch_size=8, max_batches=3, diagnostics=diagnostics,
    )
    assert metrics2["candidates_sent_to_triage"] == 24  # bounded, not 30
    assert metrics2["candidates_truncated_before_triage"] == 6
    assert metrics2["triage_batches_attempted"] == 3  # never more than max_batches


# ---------------------------------------------------------------------------
# 6. Only triage-relevant candidates reach deep analysis
# ---------------------------------------------------------------------------


def test_only_triage_relevant_candidates_proceed_to_deep_analysis():
    pool = [_triage_candidate(f"c{i}") for i in range(4)]
    relevant_ids = {pool[0].candidate_id, pool[2].candidate_id}
    client = FakeTriageLLMClient(relevant_ids=relevant_ids, default_relevant=False)
    diagnostics = []

    winners, metrics = _run_current_triage(
        pool, triage_client=client, batch_size=8, max_batches=3, diagnostics=diagnostics,
    )

    assert {c.candidate_id for c, _ in winners} == relevant_ids
    assert metrics["triage_relevant"] == 2
    assert metrics["triage_irrelevant"] == 2


# ---------------------------------------------------------------------------
# 6b. Per-candidate observability (forensic finding, run 8c605b0881e8edf0):
# every triage/deep-analysis outcome must leave an inspectable trace --
# not just an aggregate count.
# ---------------------------------------------------------------------------


def test_triage_rejection_is_recorded_per_candidate_not_just_aggregated():
    pool = [_triage_candidate(f"c{i}") for i in range(3)]
    relevant_ids = {pool[0].candidate_id}
    client = FakeTriageLLMClient(relevant_ids=relevant_ids, default_relevant=False)
    diagnostics = []

    _run_current_triage(pool, triage_client=client, batch_size=8, max_batches=3, diagnostics=diagnostics)

    irrelevant_messages = [d.message for d in diagnostics if "triage:irrelevant" in d.message]
    relevant_messages = [d.message for d in diagnostics if "triage:relevant" in d.message]
    assert len(irrelevant_messages) == 2  # pool[1], pool[2]
    assert len(relevant_messages) == 1  # pool[0]
    assert any(pool[1].candidate_id in m for m in irrelevant_messages)
    assert any(pool[0].candidate_id in m for m in relevant_messages)
    # Enough context to answer "which article, from where, routed where" --
    # not just an id.
    assert "source=" in irrelevant_messages[0] and "url=" in irrelevant_messages[0]
    assert "agent=" in irrelevant_messages[0]


def test_deep_analysis_irrelevant_verdict_is_recorded_distinctly_from_a_failure():
    """The forensic investigation's exact gap: `deep_analysis_success=0,
    timeouts=0, failures=0` (a valid, non-exceptional IRRELEVANT verdict)
    previously left no trace at all. It must now be visible, and visibly
    distinct from an exception-based failure."""
    candidates = [_triage_candidate("c0")]
    triage_results = [(c, None) for c in candidates]

    class _IrrelevantAgent:
        def analyze(self, article, evidence, *, llm_client=None):
            return None  # the specialist's own relevance.label == IRRELEVANT

    diagnostics = []
    items, metrics, _budget_exhausted, _unprocessed = _run_current_deep_analysis(
        triage_results, agents={AgentName.PEER: _IrrelevantAgent()}, deep_client=object(),
        budget=4, diagnostics=diagnostics,
    )

    assert items == []
    assert metrics["deep_analysis_attempted"] == 1
    assert metrics["deep_analysis_success"] == 0
    assert metrics["deep_analysis_timeouts"] == 0
    assert metrics["deep_analysis_failures"] == 0
    irrelevant_messages = [d.message for d in diagnostics if "deep_analysis:irrelevant" in d.message]
    assert len(irrelevant_messages) == 1
    assert candidates[0].candidate_id in irrelevant_messages[0]
    # Distinct from a failure/timeout diagnostic, which uses a different tag.
    assert not any("deep_analysis:" in d.message and "failed" in d.message for d in diagnostics)


# ---------------------------------------------------------------------------
# 7. max_deep_analysis is a GLOBAL budget, not per-agent
# ---------------------------------------------------------------------------


def test_global_deep_analysis_budget_applies_across_all_agents(config):
    """5 PEER + 5 AI_PRODUCT candidates (10 total, all triage-relevant);
    max_deep_analysis=4 is a single shared budget across both agents, not
    4 per agent (which would allow 8)."""
    config = config.model_copy(deep=True)
    # Real, already-configured peer/ai_provider names (config/sources.yaml)
    # -- routing is an exact source_name membership check against that
    # registry, so a synthetic made-up name would never route at all.
    real_peers = ["JPMorgan Chase", "Morgan Stanley", "Bank of America", "Citi", "Wells Fargo"]
    real_ai_providers = ["OpenAI", "Anthropic", "Google DeepMind", "Microsoft AI", "NVIDIA"]
    peer_sources = [_peer_source(name) for name in real_peers]
    ai_sources = [_ai_providers_source(name) for name in real_ai_providers]
    articles_by_source = {s.name: [_article(s.name, 0)] for s in (peer_sources + ai_sources)}
    retrieval_service = FakeRetrievalService(sources=peer_sources + ai_sources, articles_by_source=articles_by_source)

    peer_agent = PeerAgent(FakeLLMClient([]), config)
    ai_agent = AIProductAgent(FakeLLMClient([]), config)
    deep_client = FakeLLMClient([_relevant_assessment()] * 10)

    services = PipelineServices(
        config=config, retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.PEER: peer_agent, AgentName.AI_PRODUCT: ai_agent},
        event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()] * 10)),
        digest_service=DigestService(config, clock=FixedClock(D)), clock=FixedClock(D),
        triage_llm_client=FakeTriageLLMClient(), deep_analysis_llm_client=deep_client,
    )

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=D - timedelta(hours=24), period_end=D)
    )

    assert result["metrics"]["deep_analysis_budget"] == 4
    assert result["metrics"]["deep_analysis_attempted"] == 4  # global, not 8
    assert result["metrics"]["current_candidates_analyzed"] == 4


# ---------------------------------------------------------------------------
# 8. Duplicate routing never duplicates deep LLM work
# ---------------------------------------------------------------------------


def test_multi_routed_article_appears_once_in_the_candidate_pool(config):
    """An article that independently routes to BOTH Peer and AI_Product
    (e.g. its title names both a configured peer bank and a configured AI
    provider) must contribute exactly ONE candidate to the pool, not two."""
    from research_assistant.agents.router import RoutingResult

    article, evidence = _article_and_evidence(
        url="https://x.example.com/shared", title="Shared story", text="body " * 100,
    )
    routing = RoutingResult()
    routing.by_agent[AgentName.PEER].append(article)
    routing.by_agent[AgentName.AI_PRODUCT].append(article)
    routing.multi_routed_count = 1

    pool = _build_current_candidate_pool(
        routing, {article.id: evidence}, {AgentName.PEER, AgentName.AI_PRODUCT},
    )

    assert len(pool) == 1
    assert pool[0].primary_agent == AgentName.PEER  # first match in ROUTABLE_AGENTS order wins


# ---------------------------------------------------------------------------
# 9/10. Timeouts move forward, never retried/stalled
# ---------------------------------------------------------------------------


class _AlwaysTimesOutTriageClient:
    def __init__(self):
        self.calls = 0

    def generate_structured(self, *, system, prompt, schema):
        self.calls += 1
        raise LLMUnavailableError("Could not reach Ollama: Read timed out (read timeout=60.0)")


def test_triage_batch_timeout_consumes_that_batch_and_moves_forward():
    pool = [_triage_candidate(f"c{i}") for i in range(10)]
    client = _AlwaysTimesOutTriageClient()
    diagnostics = []

    winners, metrics = _run_current_triage(
        pool, triage_client=client, batch_size=5, max_batches=3, diagnostics=diagnostics,
    )

    assert client.calls == 2  # both batches attempted, neither retried
    assert metrics["triage_failures"] == 10
    assert winners == []
    assert any("failed" in d.message for d in diagnostics)


class _TimesOutThenSucceedsAgent:
    """Stand-in ResearchAgent whose `.analyze()` times out on the first
    call and would succeed after, proving a timeout does not retry/stall --
    the funnel must move on to the NEXT candidate, never re-attempt this
    one."""

    def __init__(self):
        self.calls = 0

    def analyze(self, article, evidence, *, llm_client=None):
        self.calls += 1
        raise LLMUnavailableError("Could not reach Ollama at http://localhost:11434: Read timed out (read timeout=90.0)")


def test_deep_analysis_timeout_consumes_the_candidate_and_moves_forward():
    candidates = [_triage_candidate(f"c{i}") for i in range(3)]
    triage_results = [(c, None) for c in candidates]
    agent = _TimesOutThenSucceedsAgent()
    diagnostics = []

    items, metrics, budget_exhausted, unprocessed = _run_current_deep_analysis(
        triage_results, agents={AgentName.PEER: agent}, deep_client=object(), budget=8, diagnostics=diagnostics,
    )

    assert agent.calls == 3  # every candidate attempted exactly once -- no retries
    assert metrics["deep_analysis_timeouts"] == 3
    assert metrics["deep_analysis_failures"] == 0
    assert items == []
    assert budget_exhausted is False
    assert unprocessed == 0


# ---------------------------------------------------------------------------
# 11/12. LLM budget exhaustion: bounded, reported, never a hidden failure
# ---------------------------------------------------------------------------


def test_deep_analysis_budget_exhaustion_reports_unprocessed_and_stops():
    candidates = [_triage_candidate(f"c{i}") for i in range(5)]
    triage_results = [(c, None) for c in candidates]

    class _AlwaysRelevantAgent:
        def analyze(self, article, evidence, *, llm_client=None):
            return make_item(article.id, company="X", title=article.title, publication_date=D)

    agent = _AlwaysRelevantAgent()
    diagnostics = []

    items, metrics, budget_exhausted, unprocessed = _run_current_deep_analysis(
        triage_results, agents={AgentName.PEER: agent}, deep_client=object(), budget=3, diagnostics=diagnostics,
    )

    assert metrics["deep_analysis_attempted"] == 3
    assert len(items) == 3
    assert budget_exhausted is True
    assert unprocessed == 2  # the 2 that never got a chance are explicitly counted, not silently dropped


def test_budget_exhausted_candidates_each_emit_a_budget_skip_diagnostic():
    """Instrumentation task (forensic finding, CURRENT run
    `e818e08adc2e4dd1`): the candidates cut off by the global deep-
    analysis budget must each get their own `budget_skip:deep_analysis`
    diagnostic, distinct from the 3 that were actually attempted and
    succeeded (which get no such diagnostic -- they are not skipped)."""
    candidates = [_triage_candidate(f"c{i}") for i in range(5)]
    triage_results = [(c, None) for c in candidates]

    class _AlwaysRelevantAgent:
        def analyze(self, article, evidence, *, llm_client=None):
            return make_item(article.id, company="X", title=article.title, publication_date=D)

    agent = _AlwaysRelevantAgent()
    diagnostics = []

    _run_current_deep_analysis(
        triage_results, agents={AgentName.PEER: agent}, deep_client=object(), budget=3, diagnostics=diagnostics,
    )

    skip_messages = [d.message for d in diagnostics if d.message.startswith("budget_skip:deep_analysis ")]
    assert len(skip_messages) == 2
    attempted_ids = {c.candidate_id for c in candidates[:3]}
    skipped_ids = {c.candidate_id for c in candidates[3:]}
    for candidate_id in skipped_ids:
        assert any(f"candidate={candidate_id} " in m for m in skip_messages)
    for candidate_id in attempted_ids:
        assert not any(f"candidate={candidate_id} " in m for m in skip_messages)


def test_budget_exhaustion_is_a_warning_not_an_error_diagnostic(config):
    """Budget exhaustion is the bounded funnel working as designed, not a
    pipeline failure -- it must stay a `warning`-severity diagnostic (never
    downgrading the run's status), exactly like the pre-existing top_k
    truncation notice it replaces for CURRENT."""
    config = config.model_copy(deep=True)
    real_peers = ["JPMorgan Chase", "Morgan Stanley", "Bank of America", "Citi", "Wells Fargo",
                  "HSBC", "Lloyds Banking Group", "Deutsche Bank"]
    sources = [_peer_source(name) for name in real_peers]
    # 2 articles per peer (16 total) -- comfortably above max_deep_analysis=4
    # using only the real, already-configured peer registry.
    articles_by_source = {s.name: [_article(s.name, 0), _article(s.name, 1)] for s in sources}
    retrieval_service = FakeRetrievalService(sources=sources, articles_by_source=articles_by_source)
    agent = PeerAgent(FakeLLMClient([]), config)
    services = PipelineServices(
        config=config, retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.PEER: agent}, event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()] * 16)),
        digest_service=DigestService(config, clock=FixedClock(D)), clock=FixedClock(D),
        triage_llm_client=FakeTriageLLMClient(), deep_analysis_llm_client=FakeLLMClient([_relevant_assessment()] * 16),
    )

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=D - timedelta(hours=24), period_end=D)
    )

    assert result["metrics"]["llm_budget_exhausted"]
    budget_diagnostics = [d for d in result["diagnostics"] if "LLM budget" in d.message]
    assert len(budget_diagnostics) == 1
    assert budget_diagnostics[0].severity == "warning"


# ---------------------------------------------------------------------------
# 13/14. LINK-grounded: zero LLM calls, still reaches the digest
# ---------------------------------------------------------------------------


def test_link_grounded_candidates_never_touch_triage_or_deep_analysis_clients(config):
    config = config.model_copy(deep=True)
    source = _peer_source("JPMorgan Chase")
    link_only = [
        ExtractedArticle(
            url="https://jpmorganchase.example.com/l1",
            title="JPMorgan Chase announces major AI partnership with a vendor",
            text=None, published_at=D - timedelta(hours=1), status=ExtractionStatus.FAILED,
            metadata={"reason": "http_error", "content_unavailable_reason": "HTTP_403"},
        ),
    ]

    retrieval_service = FakeRetrievalService(sources=[source], articles_by_source={source.name: link_only})
    agent = PeerAgent(FakeLLMClient([]), config)
    triage_client = FakeTriageLLMClient()
    deep_client = FakeLLMClient([])

    services = PipelineServices(
        config=config, retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.PEER: agent}, event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([])),
        digest_service=DigestService(config, clock=FixedClock(D)), clock=FixedClock(D),
        triage_llm_client=triage_client, deep_analysis_llm_client=deep_client,
    )

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=D - timedelta(hours=24), period_end=D)
    )

    assert triage_client.calls == []  # never invoked -- no gated CONTENT candidates existed
    assert len(result["research_items"]) == 1
    assert result["metrics"]["link_grounded_items"] == 1
    assert result["events"][0].id in result["digest"].events  # still reaches the digest


# ---------------------------------------------------------------------------
# M. Full-scale demonstration: 30 discovered candidates -> gate -> <=24
# triage -> <=8 triage calls -> <=8 deep analysis -> LINK bypasses the LLM
# entirely -> events/reports/digest generated correctly.
# ---------------------------------------------------------------------------


def test_thirty_candidates_end_to_end_through_the_bounded_funnel(config):
    config = config.model_copy(deep=True)
    real_peers = ["JPMorgan Chase", "Morgan Stanley", "Bank of America", "Citi", "Wells Fargo",
                  "HSBC", "Lloyds Banking Group", "Deutsche Bank"]
    sources = [_peer_source(name) for name in real_peers]
    # 8 peers x ~3-4 content-grounded articles each = 29 CONTENT candidates,
    # plus 1 inaccessible-but-substantive article (LINK path) = 30 discovered.
    articles_by_source = {}
    total = 0
    for s in sources:
        count = 4 if s.name != real_peers[-1] else 1  # 7*4 + 1 = 29
        articles_by_source[s.name] = [_article(s.name, i) for i in range(count)]
        total += count
    articles_by_source[real_peers[-1]].append(
        ExtractedArticle(
            url=f"https://{real_peers[-1].lower().replace(' ', '')}.example.com/link-only",
            title=f"{real_peers[-1]} announces major AI partnership with a vendor",
            text=None, published_at=D - timedelta(hours=1), status=ExtractionStatus.FAILED,
            metadata={"reason": "http_error", "content_unavailable_reason": "HTTP_403"},
        )
    )
    total += 1
    assert total == 30

    retrieval_service = FakeRetrievalService(sources=sources, articles_by_source=articles_by_source)
    agent = PeerAgent(FakeLLMClient([]), config)
    triage_client = FakeTriageLLMClient()
    deep_client = FakeLLMClient([_relevant_assessment()] * 29)

    services = PipelineServices(
        config=config, retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.PEER: agent}, event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()] * 29)),
        digest_service=DigestService(config, clock=FixedClock(D)), clock=FixedClock(D),
        triage_llm_client=triage_client, deep_analysis_llm_client=deep_client,
    )

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=D - timedelta(hours=24), period_end=D)
    )
    metrics = result["metrics"]

    assert metrics["candidates_before_llm"] == 29  # the LINK candidate never enters the CONTENT pool
    # Triage bound tripled to 72 (3*24, 2026-10-15 fix) -- 29 candidates all fit under it now,
    # unlike the old 24-candidate bound which would have truncated this pool.
    assert metrics["candidates_sent_to_triage"] == 29
    assert metrics["triage_batches_attempted"] == 10  # ceil(29/3)
    assert len(triage_client.calls) == 10
    assert metrics["deep_analysis_attempted"] == 4  # <= max_deep_analysis, GLOBAL -- unchanged by the triage fix
    assert metrics["total_llm_calls"] == 14  # 10 triage + 4 deep analysis
    assert metrics["link_grounded_items"] == 1
    assert metrics["llm_budget_exhausted"]

    # LINK item reached the digest with zero LLM involvement; CONTENT items
    # reached it through the normal EventBuilder/Report path.
    link_items = [i for i in result["research_items"] if i.grounding_type.value == "link"]
    assert len(link_items) == 1
    assert len(result["events"]) >= 1
    assert len(result["reports"]) == len(result["events"])
    assert result["digest"] is not None


# ---------------------------------------------------------------------------
# Follow-up: restore `_select_with_source_diversity` inside the bounded
# CURRENT triage pool (existing function, reused exactly -- not a new
# diversity algorithm).
# ---------------------------------------------------------------------------


def test_source_diversity_is_applied_within_the_bounded_triage_pool(config):
    """One prolific source (10 equally-ranked candidates) must not be
    allowed to consume the entire bounded triage pool while 3 other
    sources' candidates (1 each, tied in rank, discovered after the
    prolific source) go completely unrepresented -- the same guarantee
    CURRENT's old per-agent top_k always gave, now restored at the merged-
    pool level. A small triage_batch_size*max_triage_batches=4 bound makes
    the effect unambiguous: plain rank-order truncation would take 4/4
    slots from the prolific source alone; source-diversity selection must
    spread across all 4 sources instead."""
    config = config.model_copy(deep=True)
    config.settings.current_llm.triage_batch_size = 2
    config.settings.current_llm.max_triage_batches = 2  # bound = 4

    real_peers = ["JPMorgan Chase", "Morgan Stanley", "Bank of America", "Citi"]
    prolific, others = real_peers[0], real_peers[1:]
    sources = [_peer_source(name) for name in real_peers]
    articles_by_source = {prolific: [_article(prolific, i) for i in range(10)]}
    for name in others:
        articles_by_source[name] = [_article(name, 0)]
    retrieval_service = FakeRetrievalService(sources=sources, articles_by_source=articles_by_source)

    agent = PeerAgent(FakeLLMClient([]), config)
    triage_client = FakeTriageLLMClient()
    deep_client = FakeLLMClient([_relevant_assessment()] * 4)

    services = PipelineServices(
        config=config, retrieval_service=retrieval_service, normalization_service=real_normalization_service(),
        agents={AgentName.PEER: agent}, event_service=EventService(config, clock=FixedClock(D)),
        report_agent=ReportAgent(ReportingFakeLLMClient([make_narrative()] * 4)),
        digest_service=DigestService(config, clock=FixedClock(D)), clock=FixedClock(D),
        triage_llm_client=triage_client, deep_analysis_llm_client=deep_client,
    )

    result = build_graph(services).invoke(
        PipelineState(digest_type=DigestType.CURRENT, period_start=D - timedelta(hours=24), period_end=D)
    )
    metrics = result["metrics"]

    # 1. Diversity applied: all 4 sources represented in the bounded pool,
    # not just the prolific one -- recovered from the triage prompts
    # actually sent (the only place the bounded, diversified pool surfaces).
    sent_titles = "\n".join(triage_client.calls)
    for name in real_peers:
        assert name in sent_titles, f"{name} missing from the bounded triage pool -- diversity not applied"

    # 2/3/4. Bounds unchanged by this follow-up.
    assert metrics["candidates_sent_to_triage"] == 4  # triage_batch_size(2) * max_triage_batches(2)
    assert metrics["deep_analysis_attempted"] <= 4
    assert metrics["triage_batches_attempted"] + metrics["deep_analysis_attempted"] <= 6


def test_global_candidate_bound_is_still_24_by_default(config):
    pool = [_triage_candidate(f"c{i}") for i in range(40)]
    from research_assistant.pipeline.nodes import _apply_source_diversity_to_pool

    diverse = _apply_source_diversity_to_pool(pool, 24)
    assert len(diverse) == 24


def test_max_deep_analysis_bound_and_total_call_ceiling_unchanged(config):
    """Reconfirms the triage performance fix's config values are the ones
    actually in effect. The triage bound was tripled 3*8=24 -> 3*24=72
    (2026-10-15, see config/settings.yaml's own comment) once the
    discovery fixes grew the qualifying pool well past what 24 slots
    could meaningfully sample; `max_deep_analysis=4` (down from 8) remains
    the CPU-only qwen2.5:3b deep-analysis cost-reduction fix's
    conservative budget, deliberately untouched by that change."""
    config = config.model_copy(deep=True)
    assert config.settings.current_llm.triage_batch_size == 3
    assert config.settings.current_llm.max_triage_batches == 24
    assert config.settings.current_llm.max_deep_analysis == 4
    assert config.settings.current_llm.triage_timeout_seconds == 90
    # 140s (raised from 100s, high-recall reliability fix, 2026-10-12): a
    # real-qwen2.5:3b benchmark measured a cold AnalystAgent call at 98.98s
    # -- inside the old 100s timeout by under 1.2% -- and reproduced two
    # genuine timeouts at the 100s cap across five cold specialist calls,
    # matching CURRENT run `beeb562aba597399`'s 3-of-4 deep-analysis
    # timeouts. See config/settings.yaml's `deep_analysis_timeout_seconds`
    # comment for the full measurement.
    assert config.settings.current_llm.deep_analysis_timeout_seconds == 140
