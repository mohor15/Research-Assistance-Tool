"""CURRENT candidate backlog: eligible-but-not-yet-processed candidates are
carried forward across runs instead of being lost to a per-run budget.
Exercises the pure helpers in `pipeline.nodes` directly (never the full
LangGraph/PipelineRunner -- that end-to-end wiring is covered separately
in `tests/pipeline/test_current_end_to_end_regression.py`), with
lightweight fakes standing in for the database/LLM -- no network, no real
Ollama."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from research_assistant.agents.triage import TriageCandidate, TriageItemResult
from research_assistant.models.enums import AgentName, BacklogStatus
from research_assistant.pipeline.nodes import (
    _BACKLOG_AGING_CAP,
    _aging_boosted_rank_key,
    _compute_backlog_updates,
    _resumable_backlog_candidates,
)
from research_assistant.pipeline.state import BacklogUpdate, PipelineDiagnostic, PipelineState

D = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
PERIOD_START = D - timedelta(hours=24)
PERIOD_END = D


class _FakeArticle:
    def __init__(self, article_id="art1", *, title="A substantive headline", publication_date=D,
                 precision="exact", source_name="Source", url="https://example.com/a"):
        self.id = article_id
        self.title = title
        self.publication_date = publication_date
        self.publication_date_precision = precision
        self.source_name = source_name
        self.url = url


class _FakeEvidence:
    def __init__(self, content=None):
        self.content = content or ("Substantive article body text. " * 30)


class _FakeResearchItem:
    def __init__(self, article_id, item_id):
        self.article_id = article_id
        self.id = item_id


class _FakeBacklogRow:
    def __init__(self, *, agent="ai_product", status=BacklogStatus.PENDING_TRIAGE_BUDGET.value,
                 triage_relevant=None, triage_confidence=None, attempt_count=0, defer_count=0,
                 first_seen_run_id="run0", first_seen_at=D - timedelta(days=1)):
        self.agent = agent
        self.status = status
        self.triage_relevant = triage_relevant
        self.triage_confidence = triage_confidence
        self.attempt_count = attempt_count
        self.defer_count = defer_count
        self.first_seen_run_id = first_seen_run_id
        self.first_seen_at = first_seen_at


class _FakeRecord:
    def __init__(self, article, evidence, backlog_row):
        self.article = article
        self.evidence = evidence
        self.backlog = backlog_row


class _FakePersistenceService:
    def __init__(self, records: Optional[List[_FakeRecord]] = None, *, raise_on_read: bool = False):
        self._records = records or []
        self._raise_on_read = raise_on_read
        self.last_limit = None
        self.last_agents = None

    def get_pending_backlog_candidates(self, *, limit, agents=None):
        self.last_limit, self.last_agents = limit, agents
        if self._raise_on_read:
            raise RuntimeError("database unavailable")
        return self._records


class _FakeCurrentLLMSettings:
    backlog_claim_limit = 8
    backlog_max_attempts = 3


class _FakeSettings:
    current_llm = _FakeCurrentLLMSettings()


class _FakeConfig:
    settings = _FakeSettings()


class _FakeServices:
    def __init__(self, persistence_service=None, agents=None):
        self.persistence_service = persistence_service
        self.config = _FakeConfig()
        self.agents = agents or {AgentName.AI_PRODUCT: object()}


def _state(period_start=PERIOD_START, period_end=PERIOD_END):
    return PipelineState(period_start=period_start, period_end=period_end)


# ---------------------------------------------------------------------------
# _aging_boosted_rank_key: bounded fairness
# ---------------------------------------------------------------------------


def test_a_deferred_candidate_outranks_an_otherwise_equal_fresh_one():
    article_a = _FakeArticle("a", publication_date=D)
    article_b = _FakeArticle("b", publication_date=D)
    evidence = _FakeEvidence()
    counts = {"a": 2, "b": 0}

    key_a = _aging_boosted_rank_key(article_a, evidence, [], [], counts)
    key_b = _aging_boosted_rank_key(article_b, evidence, [], [], counts)

    assert key_a > key_b  # same underlying strength, but "a" has been deferred twice already


def test_aging_boost_is_bounded_not_unbounded():
    article = _FakeArticle("a", publication_date=D)
    evidence = _FakeEvidence()

    key_at_cap = _aging_boosted_rank_key(article, evidence, [], [], {"a": _BACKLOG_AGING_CAP})
    key_beyond_cap = _aging_boosted_rank_key(article, evidence, [], [], {"a": _BACKLOG_AGING_CAP + 50})

    assert key_at_cap == key_beyond_cap  # capped, never grows without bound


def test_a_brand_new_candidate_with_no_prior_attempts_gets_no_boost():
    article = _FakeArticle("a", publication_date=D)
    evidence = _FakeEvidence()
    assert _aging_boosted_rank_key(article, evidence, [], [], {})[0] == 0


# ---------------------------------------------------------------------------
# _resumable_backlog_candidates: read-only resume + period revalidation
# ---------------------------------------------------------------------------


def test_never_triaged_backlog_candidate_is_returned_for_normal_triage():
    record = _FakeRecord(
        _FakeArticle("a", publication_date=D - timedelta(hours=1)), _FakeEvidence(), _FakeBacklogRow(triage_relevant=None),
    )
    services = _FakeServices(_FakePersistenceService([record]))
    diagnostics: List[PipelineDiagnostic] = []

    never_triaged, already_triaged, attempt_counts, records_by_id, rejected = _resumable_backlog_candidates(
        services, _state(), diagnostics=diagnostics,
    )

    assert len(never_triaged) == 1
    assert never_triaged[0].candidate_id == "a"
    assert already_triaged == []
    assert attempt_counts["a"] == 0
    assert "a" in records_by_id
    assert rejected == []
    assert any(d.message.startswith("backlog:resumed") for d in diagnostics)


def test_already_triaged_relevant_candidate_bypasses_triage():
    record = _FakeRecord(
        _FakeArticle("a", publication_date=D - timedelta(hours=1)), _FakeEvidence(),
        _FakeBacklogRow(triage_relevant=True, triage_confidence=0.72),
    )
    services = _FakeServices(_FakePersistenceService([record]))

    never_triaged, already_triaged, _counts, _records, _rejected = _resumable_backlog_candidates(
        services, _state(), diagnostics=[],
    )

    assert never_triaged == []
    assert len(already_triaged) == 1
    candidate, result = already_triaged[0]
    assert candidate.candidate_id == "a"
    assert result.relevant is True
    assert result.confidence == 0.72


def test_candidate_no_longer_in_period_is_rejected_on_resume():
    """Revalidation: a candidate deferred under an older period must never
    be smuggled into a newer one merely because it remained pending."""
    stale_article = _FakeArticle("a", publication_date=D - timedelta(days=5))
    record = _FakeRecord(stale_article, _FakeEvidence(), _FakeBacklogRow())
    services = _FakeServices(_FakePersistenceService([record]))
    diagnostics: List[PipelineDiagnostic] = []

    never_triaged, already_triaged, _counts, _records, rejected = _resumable_backlog_candidates(
        services, _state(), diagnostics=diagnostics,
    )

    assert never_triaged == []
    assert already_triaged == []
    assert len(rejected) == 1
    assert rejected[0].status == BacklogStatus.REJECTED_DATE_GATE.value
    assert rejected[0].article_id == "a"
    assert any(d.message.startswith("backlog:rejected_date_gate") for d in diagnostics)


def test_a_backlog_read_failure_never_crashes_the_run():
    services = _FakeServices(_FakePersistenceService(raise_on_read=True))
    diagnostics: List[PipelineDiagnostic] = []

    never_triaged, already_triaged, counts, records, rejected = _resumable_backlog_candidates(
        services, _state(), diagnostics=diagnostics,
    )

    assert (never_triaged, already_triaged, counts, records, rejected) == ([], [], {}, {}, [])
    assert any("backlog:read_failed" in d.message for d in diagnostics)


def test_no_persistence_service_configured_returns_nothing_cleanly():
    services = _FakeServices(persistence_service=None)
    result = _resumable_backlog_candidates(services, _state(), diagnostics=[])
    assert result == ([], [], {}, {}, [])


def test_claim_limit_and_agent_filter_are_passed_through_to_the_read():
    services = _FakeServices(_FakePersistenceService([]), agents={AgentName.AI_PRODUCT: object(), AgentName.STARTUP: object()})
    _resumable_backlog_candidates(services, _state(), diagnostics=[])

    assert services.persistence_service.last_limit == 8  # from _FakeCurrentLLMSettings.backlog_claim_limit
    assert set(services.persistence_service.last_agents) == {"ai_product", "startup"}


# ---------------------------------------------------------------------------
# _compute_backlog_updates: every considered candidate gets exactly one
# explicit, non-guessed disposition.
# ---------------------------------------------------------------------------


def _candidate(article_id, agent=AgentName.AI_PRODUCT):
    return TriageCandidate(
        candidate_id=article_id, article=_FakeArticle(article_id), evidence=_FakeEvidence(), primary_agent=agent,
    )


def test_fresh_success_with_no_prior_backlog_row_needs_no_update():
    updates = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[_candidate("a")], triage_winners=[],
        research_items=[_FakeResearchItem("a", "ri1")], diagnostics=[], records_by_article_id={},
        backlog_max_attempts=3,
    )
    assert updates == []


def test_resumed_success_closes_out_the_backlog_row_as_completed():
    record = _FakeRecord(_FakeArticle("a"), _FakeEvidence(), _FakeBacklogRow(attempt_count=1))
    updates = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[_candidate("a")], triage_winners=[],
        research_items=[_FakeResearchItem("a", "ri1")], diagnostics=[],
        records_by_article_id={"a": record}, backlog_max_attempts=3,
    )
    assert len(updates) == 1
    assert updates[0].status == BacklogStatus.COMPLETED.value
    assert updates[0].research_item_id == "ri1"
    assert updates[0].attempt_count == 2  # prior 1 + this successful attempt


def test_triage_irrelevant_candidate_is_rejected_explicitly():
    diagnostics = [PipelineDiagnostic(
        stage="agent_analysis", message="triage:irrelevant candidate=a agent=ai_product source='X' title='t' url=u confidence=0.1",
        severity="warning",
    )]
    updates = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[_candidate("a")], triage_winners=[], research_items=[],
        diagnostics=diagnostics, records_by_article_id={}, backlog_max_attempts=3,
    )
    assert len(updates) == 1
    assert updates[0].status == BacklogStatus.REJECTED_TRIAGE_IRRELEVANT.value


def test_budget_skip_triage_candidate_becomes_pending_triage_budget():
    diagnostics = [PipelineDiagnostic(
        stage="agent_analysis", message="budget_skip:triage candidate=a agent=ai_product source='X' title='t' url=u reason=triage_pool_bound",
        severity="warning",
    )]
    updates = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[_candidate("a")], triage_winners=[], research_items=[],
        diagnostics=diagnostics, records_by_article_id={}, backlog_max_attempts=3,
    )
    assert len(updates) == 1
    assert updates[0].status == BacklogStatus.PENDING_TRIAGE_BUDGET.value


def test_budget_skip_deep_analysis_selection_persists_the_triage_result():
    candidate = _candidate("a")
    result = TriageItemResult(candidate_id="a", relevant=True, confidence=0.81)
    diagnostics = [PipelineDiagnostic(
        stage="agent_analysis",
        message="budget_skip:deep_analysis_selection candidate=a agent=ai_product source='X' title='t' url=u reason=domain_fair_selection",
        severity="warning",
    )]
    updates = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[], triage_winners=[(candidate, result)], research_items=[],
        diagnostics=diagnostics, records_by_article_id={}, backlog_max_attempts=3,
    )
    assert len(updates) == 1
    assert updates[0].status == BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value
    assert updates[0].triage_relevant is True
    assert updates[0].triage_confidence == 0.81  # never re-triaged next time


def test_deep_analysis_failure_below_cap_is_recoverable_pending_retry():
    diagnostics = [PipelineDiagnostic(
        stage="agent_analysis", message="deep_analysis:ai_product:a: Could not reach Ollama: Read timed out",
        severity="warning",
    )]
    updates = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[_candidate("a")], triage_winners=[], research_items=[],
        diagnostics=diagnostics, records_by_article_id={}, backlog_max_attempts=3,
    )
    assert len(updates) == 1
    assert updates[0].status == BacklogStatus.PENDING_RETRY.value
    assert updates[0].attempt_count == 1


def test_deep_analysis_failure_at_cap_becomes_rejected_retry_exhausted():
    record = _FakeRecord(_FakeArticle("a"), _FakeEvidence(), _FakeBacklogRow(attempt_count=2))  # already failed twice
    diagnostics = [PipelineDiagnostic(
        stage="agent_analysis", message="deep_analysis:ai_product:a: Could not reach Ollama: Read timed out",
        severity="warning",
    )]
    updates = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[_candidate("a")], triage_winners=[], research_items=[],
        diagnostics=diagnostics, records_by_article_id={"a": record}, backlog_max_attempts=3,
    )
    assert len(updates) == 1
    assert updates[0].status == BacklogStatus.REJECTED_RETRY_EXHAUSTED.value  # 2 prior + this = 3 = cap
    assert updates[0].attempt_count == 3


def test_a_timeout_is_still_fully_inspectable_not_deleted():
    """Even a terminal rejected_retry_exhausted row carries a real,
    readable reason -- never a bare status with no explanation."""
    record = _FakeRecord(_FakeArticle("a"), _FakeEvidence(), _FakeBacklogRow(attempt_count=2))
    diagnostics = [PipelineDiagnostic(
        stage="agent_analysis", message="deep_analysis:ai_product:a: timed out", severity="warning",
    )]
    updates = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[_candidate("a")], triage_winners=[], research_items=[],
        diagnostics=diagnostics, records_by_article_id={"a": record}, backlog_max_attempts=3,
    )
    assert "3 time(s)" in updates[0].reason
    assert "backlog_max_attempts=3" in updates[0].reason


def test_a_completed_deep_analysis_irrelevant_verdict_is_its_own_terminal_disposition():
    """Reliability fix (CURRENT run `31fa192e4d7a7175`): a deep-analysis
    call that completed normally and produced a genuine, considered
    non-relevant verdict must never be confused with a timeout/failure
    (pending_retry) or with the generic unresolved fallback -- it is a
    real, final, negative answer."""
    diagnostics = [PipelineDiagnostic(
        stage="agent_analysis",
        message="deep_analysis:irrelevant candidate=a agent=peer source='JPMorgan Chase' title='t' url=u",
        severity="warning",
    )]
    updates = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[_candidate("a")], triage_winners=[], research_items=[],
        diagnostics=diagnostics, records_by_article_id={}, backlog_max_attempts=3,
    )
    assert len(updates) == 1
    assert updates[0].status == BacklogStatus.REJECTED_DEEP_ANALYSIS_IRRELEVANT.value
    assert updates[0].attempt_count == 1  # a genuine attempt really happened


def test_deep_analysis_irrelevant_is_never_mistaken_for_a_timeout_or_budget_skip():
    """Both a genuine timeout AND a genuine completed-irrelevant verdict in
    the SAME run's diagnostics must each land in their own distinct
    disposition, never collapsed together."""
    diagnostics = [
        PipelineDiagnostic(
            stage="agent_analysis",
            message="deep_analysis:irrelevant candidate=irrelevant_one agent=ai_product source='FT' title='t' url=u",
            severity="warning",
        ),
        PipelineDiagnostic(
            stage="agent_analysis",
            message="deep_analysis:analyst:deadbeef01: Could not reach Ollama: Read timed out",
            severity="warning",
        ),
    ]
    updates = _compute_backlog_updates(
        considered_ids={"irrelevant_one", "deadbeef01"},
        gated_pool=[_candidate("irrelevant_one"), _candidate("deadbeef01")],
        triage_winners=[], research_items=[], diagnostics=diagnostics, records_by_article_id={},
        backlog_max_attempts=3,
    )
    by_id = {u.article_id: u for u in updates}
    assert by_id["irrelevant_one"].status == BacklogStatus.REJECTED_DEEP_ANALYSIS_IRRELEVANT.value
    assert by_id["deadbeef01"].status == BacklogStatus.PENDING_RETRY.value


def test_every_considered_candidate_gets_exactly_one_disposition():
    """No eligible candidate disappears without a disposition -- even a
    candidate matching none of the known diagnostic categories (e.g. its
    routed agent was disabled this run, which spends no budget and emits
    no diagnostic) still gets an explicit, non-silent fallback update."""
    updates = _compute_backlog_updates(
        considered_ids={"mystery"}, gated_pool=[_candidate("mystery")], triage_winners=[], research_items=[],
        diagnostics=[], records_by_article_id={}, backlog_max_attempts=3,
    )
    assert len(updates) == 1
    assert updates[0].status == BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value
    assert updates[0].reason  # never empty/blank


# ---------------------------------------------------------------------------
# Reliability patch 1: `defer_count` increments only on a genuine lost-
# opportunity-to-budget deferral, never on discovery/load, an explicit
# rejection, or a real analysis attempt -- and the aging boost is fed by
# `defer_count + attempt_count`, closing the gap where a purely
# budget-deferred candidate never gained any priority across runs.
# ---------------------------------------------------------------------------


def test_repeated_triage_budget_deferral_increments_defer_count_each_run():
    diagnostics = [PipelineDiagnostic(
        stage="agent_analysis", message="budget_skip:triage candidate=a agent=ai_product source='X' title='t' url=u reason=triage_pool_bound",
        severity="warning",
    )]
    # Round 1: brand-new candidate, no prior record.
    updates_1 = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[_candidate("a")], triage_winners=[], research_items=[],
        diagnostics=diagnostics, records_by_article_id={}, backlog_max_attempts=3,
    )
    assert updates_1[0].defer_count == 1
    assert updates_1[0].attempt_count == 0  # never touched by a pure deferral

    # Round 2: the SAME candidate, resumed from the persisted round-1 state.
    record = _FakeRecord(_FakeArticle("a"), _FakeEvidence(), _FakeBacklogRow(defer_count=updates_1[0].defer_count))
    updates_2 = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[_candidate("a")], triage_winners=[], research_items=[],
        diagnostics=diagnostics, records_by_article_id={"a": record}, backlog_max_attempts=3,
    )
    assert updates_2[0].defer_count == 2  # increments again, never resets, never stalls at 1


def test_repeated_deep_analysis_budget_deferral_increments_defer_count_each_run():
    candidate = _candidate("a")
    result = TriageItemResult(candidate_id="a", relevant=True, confidence=0.7)
    diagnostics = [PipelineDiagnostic(
        stage="agent_analysis",
        message="budget_skip:deep_analysis_selection candidate=a agent=ai_product source='X' title='t' url=u reason=domain_fair_selection",
        severity="warning",
    )]
    record = _FakeRecord(_FakeArticle("a"), _FakeEvidence(), _FakeBacklogRow(defer_count=2, triage_relevant=True))
    updates = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[], triage_winners=[(candidate, result)], research_items=[],
        diagnostics=diagnostics, records_by_article_id={"a": record}, backlog_max_attempts=3,
    )
    assert updates[0].defer_count == 3  # prior 2 + this run's deferral
    assert updates[0].attempt_count == 0  # still never a real analysis attempt


def test_defer_count_is_not_incremented_by_an_explicit_rejection_or_real_attempt():
    """Rejections and real analysis attempts/failures must never inflate
    the budget-deferral-specific counter -- `attempt_count` already covers
    real attempts, and an explicit rejection is not a deferral at all."""
    record = _FakeRecord(_FakeArticle("a"), _FakeEvidence(), _FakeBacklogRow(defer_count=1))

    irrelevant_diag = [PipelineDiagnostic(
        stage="agent_analysis", message="triage:irrelevant candidate=a agent=ai_product source='X' title='t' url=u confidence=0.1",
        severity="warning",
    )]
    updates = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[_candidate("a")], triage_winners=[], research_items=[],
        diagnostics=irrelevant_diag, records_by_article_id={"a": record}, backlog_max_attempts=3,
    )
    assert updates[0].defer_count == 1  # unchanged

    failure_diag = [PipelineDiagnostic(
        stage="agent_analysis", message="deep_analysis:ai_product:a: Read timed out", severity="warning",
    )]
    updates = _compute_backlog_updates(
        considered_ids={"a"}, gated_pool=[_candidate("a")], triage_winners=[], research_items=[],
        diagnostics=failure_diag, records_by_article_id={"a": record}, backlog_max_attempts=3,
    )
    assert updates[0].defer_count == 1  # unchanged -- this is a real attempt, tracked by attempt_count instead
    assert updates[0].attempt_count == 1


def test_aging_boost_increases_from_defer_count_alone():
    """A candidate that has ONLY ever been budget-deferred (never actually
    attempted) must still gain rank across repeated deferrals -- the gap
    the pre-live-scan audit found and this patch closes."""
    article = _FakeArticle("a", publication_date=D - timedelta(hours=1))
    evidence = _FakeEvidence()

    never_deferred = _aging_boosted_rank_key(article, evidence, [], [], {"a": 0})
    deferred_twice = _aging_boosted_rank_key(article, evidence, [], [], {"a": 2})
    assert deferred_twice > never_deferred


def test_aging_boost_combines_defer_count_and_attempt_count_bounded_by_the_same_cap():
    article = _FakeArticle("a", publication_date=D - timedelta(hours=1))
    evidence = _FakeEvidence()

    # 2 deferrals + 3 attempts = 5, exactly at the cap.
    at_cap = _aging_boosted_rank_key(article, evidence, [], [], {"a": 5})
    beyond_cap = _aging_boosted_rank_key(article, evidence, [], [], {"a": 50})
    assert at_cap == beyond_cap
    assert at_cap[0] == _BACKLOG_AGING_CAP


def test_fairness_counts_from_resumable_backlog_candidates_combine_both_counters():
    record = _FakeRecord(
        _FakeArticle("a", publication_date=D - timedelta(hours=1)), _FakeEvidence(),
        _FakeBacklogRow(attempt_count=2, defer_count=3),
    )
    services = _FakeServices(_FakePersistenceService([record]))

    _never_triaged, _already_triaged, fairness_counts, _records, _extra = _resumable_backlog_candidates(
        services, _state(), diagnostics=[],
    )

    assert fairness_counts["a"] == 5  # 2 attempts + 3 defers


# ---------------------------------------------------------------------------
# Reliability patch 2: a malformed persisted backlog record (e.g. an
# out-of-range `triage_confidence`) is isolated per-record -- it can never
# abort the whole run, and it is never silently marked completed/
# irrelevant, nor silently left looking like ordinary pending work.
# ---------------------------------------------------------------------------


def test_a_malformed_triage_confidence_is_quarantined_not_crashed():
    malformed_record = _FakeRecord(
        _FakeArticle("bad", publication_date=D - timedelta(hours=1)), _FakeEvidence(),
        _FakeBacklogRow(triage_relevant=True, triage_confidence=150.0),  # out of range, not a 0-100 rescale either
    )
    services = _FakeServices(_FakePersistenceService([malformed_record]))
    diagnostics: List[PipelineDiagnostic] = []

    never_triaged, already_triaged, _fairness, _records, extra_updates = _resumable_backlog_candidates(
        services, _state(), diagnostics=diagnostics,
    )

    assert never_triaged == []
    assert already_triaged == []  # never silently treated as a valid, usable triage winner
    assert len(extra_updates) == 1
    assert extra_updates[0].status == BacklogStatus.QUARANTINED.value
    assert extra_updates[0].article_id == "bad"
    assert extra_updates[0].reason  # a real, readable reason, never blank
    assert any(d.message.startswith("backlog:quarantined") and "bad" in d.message for d in diagnostics)


def test_a_mixed_batch_with_one_bad_row_still_processes_the_good_ones():
    good_record_1 = _FakeRecord(
        _FakeArticle("good1", publication_date=D - timedelta(hours=1)), _FakeEvidence(), _FakeBacklogRow(),
    )
    malformed_record = _FakeRecord(
        _FakeArticle("bad", publication_date=D - timedelta(hours=1)), _FakeEvidence(),
        _FakeBacklogRow(triage_relevant=True, triage_confidence=-3.0),
    )
    good_record_2 = _FakeRecord(
        _FakeArticle("good2", publication_date=D - timedelta(hours=1)), _FakeEvidence(),
        _FakeBacklogRow(triage_relevant=True, triage_confidence=0.6),
    )
    services = _FakeServices(_FakePersistenceService([good_record_1, malformed_record, good_record_2]))

    never_triaged, already_triaged, _fairness, _records, extra_updates = _resumable_backlog_candidates(
        services, _state(), diagnostics=[],
    )

    assert [c.candidate_id for c in never_triaged] == ["good1"]
    assert [c.candidate_id for c, _ in already_triaged] == ["good2"]
    assert [u.article_id for u in extra_updates] == ["bad"]
    assert extra_updates[0].status == BacklogStatus.QUARANTINED.value


def test_a_systemic_read_failure_still_fails_safely_not_silently():
    """Contrast with per-record isolation: a database-wide read failure is
    NOT quietly absorbed as if nothing were wrong -- it is diagnosed
    explicitly and the run proceeds with zero backlog candidates rather
    than guessing at partial data."""
    services = _FakeServices(_FakePersistenceService(raise_on_read=True))
    diagnostics: List[PipelineDiagnostic] = []

    result = _resumable_backlog_candidates(services, _state(), diagnostics=diagnostics)

    assert result == ([], [], {}, {}, [])
    assert any("backlog:read_failed" in d.message for d in diagnostics)
