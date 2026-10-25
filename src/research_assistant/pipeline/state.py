"""The graph's typed state.

Only data flows through this model — services (RetrievalService,
NormalizationService, the research agents, EventService, ReportAgent,
DigestService) are dependency-injected into the graph's nodes via
`PipelineServices` (see `pipeline.nodes`), never stored in state. That is
what "no arbitrary mutable global state in the graph" means here: state is
just the Phase 0–6 data flowing between already-existing services.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from research_assistant.config.schema import AppConfig
from research_assistant.events.models import Event
from research_assistant.models.article import Article
from research_assistant.models.digest import Digest, DigestType
from research_assistant.models.evidence import Evidence
from research_assistant.models.report import Report, ReportFailure
from research_assistant.models.research_item import ResearchItem
from research_assistant.retrieval.models import ExtractedArticle, SourceDefinition


class ExtractedBundle(BaseModel):
    """One (source, extracted-article) pair — kept together because
    normalization needs both."""

    source: SourceDefinition
    extracted: ExtractedArticle


class PipelineDiagnostic(BaseModel):
    """A structured, non-fatal failure record — one source/article/agent-call/
    event failing never silently disappears and never aborts the whole run."""

    stage: str
    message: str
    severity: Literal["warning", "error"] = "error"


class BacklogUpdate(BaseModel):
    """One candidate's backlog disposition computed by this run's
    `agent_analysis` node -- never written to the database directly from
    there (see that node's own comment): collected in `PipelineState` and
    persisted atomically, in the same transaction as everything else this
    run produced, by `make_persistence_node`. `status` is one of the
    explicit, non-guessed values in `storage.repositories.backlog`
    (`PENDING_*`/`COMPLETED`/`REJECTED_*`)."""

    article_id: str
    agent: str
    status: str
    reason: str
    attempt_count: int = 0
    defer_count: int = 0
    """Incremented ONLY when this candidate genuinely lost a processing
    opportunity to a budget limit this run (`pending_triage_budget`/
    `pending_deep_analysis_budget`) -- never for a real analysis attempt
    (that is `attempt_count`'s job), and never merely because the
    candidate was discovered/loaded. Feeds the bounded aging boost in
    `_aging_boosted_rank_key` so repeated pure budget deferrals, not only
    repeated analysis failures, raise a candidate's priority over time."""
    triage_relevant: Optional[bool] = None
    triage_confidence: Optional[float] = None
    research_item_id: Optional[str] = None
    first_seen_run_id: Optional[str] = None
    first_seen_at: Optional[datetime] = None
    """Only set when resuming a PRE-EXISTING backlog row (preserves its
    original first-seen time/run for fairness aging/ordering); both `None`
    for a brand-new candidate deferred for the first time this run (the
    persistence node then stamps THIS run/time as its first-seen run)."""


class PipelineState(BaseModel):
    # Inputs (set before/at invocation)
    digest_type: DigestType = DigestType.MORNING
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    run_id: Optional[str] = None
    """This run's deterministic id (`pipeline.runner.generate_run_id`),
    set once by `PipelineRunner.run()` before the graph is invoked. Used
    to stamp explicit run ownership onto every ResearchItem/Event this
    run produces -- `None` only for callers that invoke the graph
    directly without going through `PipelineRunner` (e.g. most unit
    tests), which is also why `ResearchItem.run_id`/`Event.run_id` default
    to `None`."""

    # Populated as the graph runs
    config: Optional[AppConfig] = None
    sources: List[SourceDefinition] = Field(default_factory=list)
    extracted_bundles: List[ExtractedBundle] = Field(default_factory=list)
    articles: List[Article] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    research_items: List[ResearchItem] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)
    reports: List[Report] = Field(default_factory=list)
    report_failures: List[ReportFailure] = Field(default_factory=list)
    digest: Optional[Digest] = None
    backlog_updates: List[BacklogUpdate] = Field(default_factory=list)
    """CURRENT-only: this run's computed disposition for every candidate
    that did not reach a completed ResearchItem via the normal path --
    deferred (pending), rejected, or (for a resumed backlog candidate that
    succeeded) completed. Empty for Morning/Evening. Persisted atomically
    alongside everything else by `make_persistence_node`."""

    diagnostics: List[PipelineDiagnostic] = Field(default_factory=list)

    metrics: Dict[str, int] = Field(default_factory=dict)
    """Simple structured counters for the discovery->relevant->events->reports
    funnel (Phase 8 Part 11) — e.g. `sources_discovered`, `candidates_discovered`,
    `research_items_relevant`, `events_formed`, `reports_succeeded`. Each node
    adds its own keys; nothing here recomputes anything a service already
    computed, it just counts what already happened."""
