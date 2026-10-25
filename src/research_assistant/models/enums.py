"""Shared enums used by configuration and (in later phases) by evidence/event models.

Kept intentionally minimal for Phase 0: these are identifiers referenced by
config/sources.yaml and config/topics.yaml, not business-logic models.
"""

from __future__ import annotations

from enum import Enum


class SourceType(str, Enum):
    """How a source relates to the underlying facts it publishes."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    ANALYST = "analyst"
    PRESS_DISTRIBUTION = "press_distribution"


class SourcePriority(str, Enum):
    """Crawl/attention priority assigned to a source in the registry."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgentName(str, Enum):
    """The research agents defined by the project (see agents.* in settings.yaml)."""

    PEER = "peer"
    AI_PRODUCT = "ai_product"
    STARTUP = "startup"
    ANALYST = "analyst"
    FINANCIAL_NEWS = "financial_news"
    REPORT = "report"


class EventType(str, Enum):
    """Underlying real-world event categories tracked across all agents."""

    PRODUCT_LAUNCH = "product_launch"
    PARTNERSHIP = "partnership"
    ACQUISITION = "acquisition"
    INVESTMENT = "investment"
    FUNDING = "funding"
    IPO = "IPO"
    STRATEGIC_INITIATIVE = "strategic_initiative"
    RESEARCH = "research"
    MODEL_RELEASE = "model_release"
    PLATFORM_RELEASE = "platform_release"
    INFRASTRUCTURE = "infrastructure"
    REGULATORY_DEVELOPMENT = "regulatory_development"
    ENTERPRISE_DEAL = "enterprise_deal"
    HIRING = "hiring"


class EntityType(str, Enum):
    """Closed vocabulary for a single extracted `entities` name (Phase 13B).

    Distinguishes genuine organizations from products/platforms/models/
    technologies that an agent might otherwise list alongside real
    companies (e.g. "Claude.ai" or "Agentic AI") -- only `COMPANY` (and
    `ORGANIZATION`) are ever surfaced as an event's "related companies"
    (see `events.builder._collect_related_companies`)."""

    COMPANY = "company"
    PRODUCT = "product"
    PLATFORM = "platform"
    MODEL = "model"
    TECHNOLOGY = "technology"
    ORGANIZATION = "organization"
    PERSON = "person"
    LOCATION = "location"


class DigestPeriod(str, Enum):
    """The two scheduled digest runs."""

    MORNING = "morning"
    EVENING = "evening"


class ExecutionMode(str, Enum):
    """Phase 14: how a pipeline run was triggered -- never inferred after
    the fact. `SCHEDULED` is set ONLY by `ResearchScheduler`'s own cron
    callbacks; every other caller (the manual CLI, direct `PipelineRunner`
    use, tests) is `MANUAL`."""

    SCHEDULED = "scheduled"
    MANUAL = "manual"


class GroundingType(str, Enum):
    """Distinguishes two kinds of intelligence a ResearchItem/Event can be
    built from -- never a quality judgment about the development itself,
    only about what evidence backs the system's claims about it.

    CONTENT: the article body was actually retrieved; relevance/summary/
    extracted facts are grounded in that real text (the only kind that
    existed before this field was added -- the default, for backward
    compatibility with every pre-existing ResearchItem/Event).

    LINK: the article body could not be retrieved (robots.txt, anti-bot,
    HTTP error, or browser-extraction failure -- see
    retrieval.classification), but a relevant, dated development was
    still deterministically identified from trusted discovery metadata
    (title, source, publication date) alone. A LINK-grounded item/event
    must never carry a fact that wasn't actually present in that metadata
    -- see `agents.link_grounded` for the one place these are built."""

    CONTENT = "content"
    LINK = "link"


class DigestPersistenceStatus(str, Enum):
    """Phase 14: closes the forensic-diagnosis finding that `digest` being
    optional in `persist_pipeline_result` made "no digest" ambiguous between
    four genuinely different situations. Exactly one of these is recorded on
    every `PipelineRunResult`."""

    PERSISTED = "persisted"  # a digest with >=1 qualifying event was built and persisted
    NO_QUALIFYING_EVENTS = "no_qualifying_events"  # a digest WAS built and persisted, but with zero events
    PERSISTENCE_FAILED = "persistence_failed"  # the digest was built but the persistence transaction raised
    NOT_ATTEMPTED = "not_attempted"  # the graph never reached digest_generation (e.g. an earlier node raised) or no persistence_service is configured


class BacklogStatus(str, Enum):
    """A CURRENT candidate's disposition in `candidate_backlog`. Shared
    between `pipeline.nodes` (which computes it, without importing
    anything from `storage`) and `storage.repositories.backlog` (which
    persists it) -- defined once, here, at the dependency-free model
    layer, exactly like `DigestPersistenceStatus` above.

    PENDING_* are non-terminal: a candidate here MUST be reconsidered by a
    later run. The remaining values are terminal, but every terminal row
    still carries an explicit, inspectable reason -- nothing here is ever
    a silent drop."""

    PENDING_TRIAGE_BUDGET = "pending_triage_budget"
    """Passed every deterministic gate and was routed, but this run's
    bounded triage-pool budget was exhausted before it was ever offered a
    triage slot."""
    PENDING_DEEP_ANALYSIS_BUDGET = "pending_deep_analysis_budget"
    """Triage already judged it relevant (and that result is persisted, so
    a resuming run never re-triages it), but this run's bounded
    deep-analysis budget/domain-fair selection did not reach it."""
    PENDING_RETRY = "pending_retry"
    """A deep-analysis attempt was actually made and failed/timed out --
    recoverable, bounded by `CurrentLLMSettings.backlog_max_attempts`."""
    COMPLETED = "completed"
    """Deep analysis succeeded; `research_item_id` names the result."""
    REJECTED_TRIAGE_IRRELEVANT = "rejected_triage_irrelevant"
    """An explicit triage call judged it irrelevant under the existing
    triage policy -- never applied to a candidate that was merely never
    triaged."""
    REJECTED_DEEP_ANALYSIS_IRRELEVANT = "rejected_deep_analysis_irrelevant"
    """Reliability-fix finding (CURRENT run `31fa192e4d7a7175`, 2026-10-12):
    the specialist's full deep-analysis call COMPLETED normally and
    produced a genuine, considered `relevance.label != RELEVANT` verdict
    (`agents.base.ResearchAgent.analyze` returned `None`) -- this is a
    real, final answer, not a failure, not a timeout, and not a budget
    deferral. It must never be confused with `pending_retry` (which is for
    a call that did NOT complete) or with the generic fallback (which is
    for a candidate this run genuinely never resolved at all). Terminal,
    like `rejected_triage_irrelevant` -- never retried automatically."""
    REJECTED_DATE_GATE = "rejected_date_gate"
    """A previously-pending candidate no longer falls inside the CURRENT
    period it is being reconsidered against (revalidated on every resume,
    never carried into a period it wasn't actually confirmed to belong
    to)."""
    REJECTED_RETRY_EXHAUSTED = "rejected_retry_exhausted"
    """`backlog_max_attempts` recoverable failures/timeouts were reached --
    terminal, but still fully inspectable/recoverable by a human (never
    deleted by this status alone; see `BacklogRepository.purge_expired_
    terminal`). Explicit, supported recovery:
    `BacklogRepository.requeue_exhausted`."""
    QUARANTINED = "quarantined"
    """A persisted backlog row could not be safely reconstructed into a
    candidate this run (e.g. a malformed/out-of-range field) -- isolated
    here so one bad row can never abort the whole run or be silently
    treated as completed/irrelevant. Excluded from automatic `list_pending`
    reconsideration (retrying a structurally malformed row would just fail
    identically every run); recoverable only by explicit, deliberate
    operator action, never automatically."""
