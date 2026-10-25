"""Dashboard-facing query layer — the only module `ui/app.py` should import
for data access (Phase 7 Part 7: "Streamlit -> Query/Repository layer ->
SQLite", never raw SQL scattered through the UI).

Every function here is read-only and delegates to `PersistenceService`'s
repositories; none of it performs retrieval, clustering, scoring, or an
LLM call — those constraints are enforced structurally by this module
never importing `research_assistant.retrieval`, `research_assistant.llm`,
`research_assistant.events.clustering`, or `research_assistant.events.scoring`.
It has no dependency on Streamlit either, so it is fully unit-testable
without installing it.

List views return lightweight `EventSummary` rows (Phase 7 Part 8:
avoid loading full evidence/article bodies for a list); full detail
(`get_event_detail`) is only fetched when a caller asks for one event.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

if TYPE_CHECKING:
    # Type-only: this module's own contract (see the module docstring) is
    # to never import research_assistant.llm/.retrieval at RUNTIME --
    # `reporting.tier_b_newsletter_resolution` does (it is the one place
    # that actually calls the model). A caller resolves Tier B reports
    # there FIRST and simply hands the already-resolved dict in here for
    # rendering -- this module only ever reads it.
    from research_assistant.reporting.tier_b_newsletter_resolution import TierBReportResolution

from research_assistant.config.schema import AppConfig, SourceEntry
from research_assistant.events.models import Event
from research_assistant.models.digest import Digest, DigestType
from research_assistant.models.enums import DigestPersistenceStatus, EventType, GroundingType
from research_assistant.models.report import Report, ReportFailure
from research_assistant.models.research_item import ResearchItem
from research_assistant.storage.persistence_service import PersistenceService


@dataclass
class EventSummary:
    """A lightweight row for list views."""

    id: str
    canonical_title: str
    primary_company: str
    event_type: Optional[EventType]
    event_date: Optional[datetime]
    relevance_score: float
    confidence_score: float


@dataclass
class EventDetail:
    """Everything the "Event Detail" dashboard section (Phase 7 Part 6, item
    5) needs in one call: the Event itself, its linked ResearchItems, and
    its generated Report (or failure), if any."""

    event: Event
    research_items: List[ResearchItem]
    report: Optional[Union[Report, ReportFailure]]


@dataclass
class CompanyCount:
    company: str
    event_count: int


@dataclass
class EventTypeCount:
    event_type: str
    event_count: int


def _summarize(event: Event) -> EventSummary:
    return EventSummary(
        id=event.id,
        canonical_title=event.canonical_title,
        primary_company=event.primary_company,
        event_type=event.event_type,
        event_date=event.event_date,
        relevance_score=event.relevance_score,
        confidence_score=event.confidence_score,
    )


def get_latest_intelligence(service: PersistenceService, *, limit: int = 20) -> List[EventSummary]:
    """Recent, high-relevance-first Events for the dashboard's landing view."""
    conn = service.database.connect()
    try:
        events = service.events.get_recent(conn, limit=limit)
        return [_summarize(e) for e in sorted(events, key=lambda e: -e.relevance_score)]
    finally:
        conn.close()


def get_high_relevance_events(service: PersistenceService, *, min_relevance: float = 0.5, limit: int = 50) -> List[EventSummary]:
    """Sorted by the existing deterministic relevance_score only — no
    dashboard-invented ranking (Phase 7 Part 6, item 4)."""
    conn = service.database.connect()
    try:
        events = service.events.get_by_min_relevance(conn, min_relevance, limit=limit)
        return [_summarize(e) for e in events]
    finally:
        conn.close()


def search_events(
    service: PersistenceService,
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    company: Optional[str] = None,
    event_type: Optional[EventType] = None,
    agent: Optional[str] = None,
    min_relevance: Optional[float] = None,
    min_confidence: Optional[float] = None,
    limit: int = 50,
) -> List[EventSummary]:
    """The dashboard's filter panel (Phase 7 Part 6, item 7) — every
    parameter is optional and AND-ed together."""
    conn = service.database.connect()
    try:
        events = service.events.search(
            conn, since=since, until=until, company=company, event_type=event_type, agent=agent,
            min_relevance=min_relevance, min_confidence=min_confidence, limit=limit,
        )
        return [_summarize(e) for e in events]
    finally:
        conn.close()


def get_market_activity(service: PersistenceService, *, since_days: int = 30) -> dict:
    """Event counts by event type and by company for the "Market Activity"
    section (Phase 7 Part 6, item 3)."""
    conn = service.database.connect()
    try:
        by_company = [CompanyCount(company=row["company"], event_count=row["event_count"]) for row in service.events.count_by_company(conn)]
        by_event_type = [
            EventTypeCount(event_type=row["event_type"], event_count=row["event_count"])
            for row in service.events.count_by_event_type(conn)
        ]
        recent = service.events.get_recent(conn, limit=1000)
        return {
            "by_company": by_company,
            "by_event_type": by_event_type,
            "recent_event_count": len(recent),
        }
    finally:
        conn.close()


def get_event_detail(service: PersistenceService, event_id: str) -> Optional[EventDetail]:
    """Full detail for one event, loaded only when a user opens it (Phase 7
    Part 8) — never for list views."""
    conn = service.database.connect()
    try:
        event = service.events.get(conn, event_id)
        if event is None:
            return None
        research_items = service.research_items.get_many(conn, event.research_items)
        report = service.reports.get(conn, event_id)
        return EventDetail(event=event, research_items=research_items, report=report)
    finally:
        conn.close()


def get_source_evidence(service: PersistenceService, event_id: str):
    """The Evidence content behind one event's research items (Phase 7 Part
    6, item 6) — loaded only for the detail view, never a list."""
    conn = service.database.connect()
    try:
        event = service.events.get(conn, event_id)
        if event is None:
            return []
        results = []
        for article_id in event.articles:
            results.extend(service.evidence.get_by_article_id(conn, article_id))
        return results
    finally:
        conn.close()


def get_latest_digest(service: PersistenceService, digest_type: DigestType) -> Optional[Digest]:
    conn = service.database.connect()
    try:
        recent = service.digests.get_recent(conn, digest_type=digest_type, limit=1)
        return recent[0] if recent else None
    finally:
        conn.close()


# A few seconds' tolerance on period boundaries -- the scheduled job and a
# UI page load compute `compute_digest_period` independently (same pure
# function, same config, same anchor hours), so they match exactly in
# practice; this only guards against sub-second clock jitter, never enough
# to blur a scheduled digest with a manual or historical/validation one.
_OPERATIONAL_PERIOD_TOLERANCE = timedelta(seconds=5)


def get_operational_digest(
    service: PersistenceService, digest_type: DigestType, config: AppConfig, *, now: Optional[datetime] = None,
) -> Optional[Digest]:
    """Phase 13E: the UI must show THE CURRENT OPERATIONAL digest -- the one
    whose period is exactly `compute_digest_period(digest_type, now, config)`
    -- never merely "whatever digest row is newest". A historical/
    validation-style digest (a much wider period, e.g. a multi-day E2E
    validation window) or a manual run's digest (a different, non-anchored
    period_end) will never match this exact period, so neither can ever be
    mistaken for "today's" scheduled digest. Returns None, rather than
    falling back to any other row, when no digest for the exact operational
    period exists yet."""
    from research_assistant.scheduling.period import compute_digest_period

    now = now or datetime.now(timezone.utc)
    period_start, period_end = compute_digest_period(digest_type, now, config)

    conn = service.database.connect()
    try:
        candidates = service.digests.get_recent(conn, digest_type=digest_type, limit=50)
    finally:
        conn.close()

    for digest in candidates:
        if (
            abs(digest.period_start - period_start) <= _OPERATIONAL_PERIOD_TOLERANCE
            and abs(digest.period_end - period_end) <= _OPERATIONAL_PERIOD_TOLERANCE
        ):
            return digest
    return None


def get_recent_digests(service: PersistenceService, *, limit: int = 20) -> List[Digest]:
    conn = service.database.connect()
    try:
        return service.digests.get_recent(conn, limit=limit)
    finally:
        conn.close()


def get_latest_run_summary(service: PersistenceService):
    """The most recent pipeline run record (or `None` if none has ever
    completed) — for the dashboard's "last successful run" status display.
    Returns the real persisted `PipelineRunResult`, never a guess."""
    conn = service.database.connect()
    try:
        recent = service.runs.get_recent(conn, limit=1)
        return recent[0] if recent else None
    finally:
        conn.close()


def get_latest_run_for_digest_type(service: PersistenceService, digest_type: DigestType):
    """Like `get_latest_run_summary`, narrowed to one `digest_type` -- used
    by the CURRENT / Live Scan page so it reports the last CURRENT run
    specifically, never "whatever run happens to be newest overall" (which
    could be an unrelated Morning/Evening run)."""
    conn = service.database.connect()
    try:
        recent = service.runs.get_recent(conn, digest_type=digest_type, limit=1)
        return recent[0] if recent else None
    finally:
        conn.close()


def get_current_run_options(service: PersistenceService, *, limit: int = 20) -> List["RunOption"]:
    """CURRENT runs only, newest first -- the selector for the CURRENT
    Scan Explorer (never Morning/Evening/historical inventory)."""
    conn = service.database.connect()
    try:
        runs = service.runs.get_recent(conn, digest_type=DigestType.CURRENT, limit=limit)
    finally:
        conn.close()
    return [RunOption(run_id=r.run_id, started_at=r.started_at, status=r.status.value) for r in runs]


@dataclass
class RunOption:
    run_id: str
    started_at: datetime
    status: str


@dataclass
class ScanArticleRow:
    """One row of the CURRENT Scan Explorer's "Retrieved Articles" table --
    deliberately lightweight (no article body) for fast listing; full
    detail is only fetched on demand via `get_scan_article_detail`."""

    article_id: str
    source_name: str
    category: str  # "Peer Research" / "AI Product" / "Analyst" / "Financial News" / "Unknown"
    title: Optional[str]
    url: str
    publication_date: Optional[datetime]
    publication_date_precision: str
    analysis_status: str  # "Analyzed" / "Link-grounded" / "Rejected by deep analysis" / "Rejected by triage" / "Content unavailable" / "Not analyzed"
    agent: Optional[str]
    relevance: Optional[str]  # "Relevant" / "Irrelevant" / "Ambiguous" / None
    has_event: bool
    event_title: Optional[str]
    research_item_id: Optional[str]
    event_id: Optional[str]
    evidence_id: Optional[str]
    content_unavailable_reason: Optional[str] = None  # e.g. "ROBOTS_BLOCKED", "HTTP_403" -- only set when evidence_id is None


@dataclass
class ScanArticleDetail:
    """Full detail for one article, fetched only when the user opens it
    (Part 8/15: never loaded for the whole listing up front)."""

    article_id: str
    source_name: str
    title: Optional[str]
    url: str
    canonical_url: str
    publication_date: Optional[datetime]
    publication_date_precision: str
    retrieved_at: datetime
    category: str
    clean_text_available: bool
    evidence_id: Optional[str]
    research_item_id: Optional[str]
    event_id: Optional[str]
    # Only populated when the article was actually analyzed -- existing
    # stored agent output, never regenerated.
    agent: Optional[str] = None
    relevance: Optional[str] = None
    relevance_reason: Optional[str] = None
    is_dated_development: Optional[bool] = None
    development_date_rationale: Optional[str] = None
    content_unavailable_reason: Optional[str] = None
    grounding_type: Optional[str] = None  # "content" / "link" -- see the grounding_type feature


@dataclass
class ScanEventRow:
    event_id: str
    canonical_title: str
    event_type: Optional[str]
    primary_company: str
    related_companies: List[str]
    research_item_ids: List[str]
    article_ids: List[str]
    event_date: Optional[datetime]
    is_dated_development: Optional[bool]
    relevance_score: float
    confidence_score: float
    report_title: Optional[str]
    digest_qualified: bool
    qualification_reason: str  # "Included" / "Missing date" / "Outside current window" / "Not a dated development" / "Other"


@dataclass
class CurrentScanExplorerData:
    run_id: str
    status: str
    started_at: datetime
    finished_at: datetime
    period_start: Optional[datetime]
    period_end: Optional[datetime]
    metrics: dict  # the run's own persisted counts_json -- verbatim, never recomputed
    articles: List[ScanArticleRow]
    events: List[ScanEventRow]
    # Part 11: coverage, computed from the articles actually listed above
    # (internally consistent with them by construction), separate from
    # `metrics` (the run's own authoritative totals, which may legitimately
    # differ -- see the module-level docstring note on this being a
    # best-effort retrieved_at-window reconstruction, not an exact per-run
    # join).
    coverage_evidence: int
    coverage_analyzed: int
    coverage_rejected_by_agent: int  # triage or deep analysis actually ran and judged it not relevant -- never "unanalyzed"
    coverage_not_analyzed: int  # had Evidence, never reached an agent at all (budget/routing) -- never "irrelevant"
    coverage_content_unavailable: int  # discovered, but no Evidence could be created (robots/anti-bot/HTTP/browser failure)
    coverage_relevant: int
    coverage_irrelevant: int
    coverage_ambiguous: int
    reconstruction_note: str
    digest_persistence_status: str = "unavailable"
    """This run's own authoritative `PipelineRunResult.digest_persistence_status`
    (e.g. "no_qualifying_events", "persisted", "persistence_failed",
    "not_attempted") -- verbatim, never recomputed -- so the dashboard can
    tell "the run completed cleanly but nothing qualified" apart from "the
    run never even attempted a digest" or "persistence itself failed".
    Defaults to the honest literal "unavailable" (never fabricated, never
    inferred from zero Events) for any caller that does not supply it."""
    counts_mismatch_warning: Optional[str] = None
    """Set only when this run's own authoritative `metrics['agent_candidates_analyzed']`
    disagrees with the reconstruction's `coverage_analyzed` -- the clearest
    possible symptom of retrieved_at-window misattribution (Part 18): an
    Article row is a single, deterministically-id'd record keyed by
    canonical_url, shared and overwritten by every pipeline run that
    re-fetches that same URL (CURRENT always FRESH-refetches). A later run
    that re-fetches a URL an EARLIER run already turned into a
    ResearchItem/Event overwrites that Article's `retrieved_at`, which can
    make the earlier run's analysis appear -- via this reconstruction -- to
    belong to the later run, even though the later run's own persisted
    counts (verbatim, never recomputed) show it never actually analyzed
    that candidate. When this happens, `metrics` above remains the
    authoritative description of what THIS run did; the reconstructed
    Articles/Events list below may include rows that actually belong to a
    different run."""


_DIAGNOSTIC_CANDIDATE_ID_RE = re.compile(r"candidate=(\S+)")


def _candidate_ids_with_message_prefix(diagnostics, prefix: str) -> Set[str]:
    """Forensic finding (run 7aab105954d38497, IQuest-Q1): the Scan
    Explorer's "Not analyzed" status previously meant two genuinely
    different things -- "never reached an agent at all" and "an agent DID
    run deep analysis on this and judged it irrelevant" -- because the
    only signal checked was whether a `ResearchItem` was produced. A
    candidate deep-analysis correctly rejects is not "unanalyzed"; the
    run's own per-candidate diagnostics (`pipeline.nodes`) already record
    exactly this, as `f"{prefix}candidate={candidate_id} ..."` lines --
    this just reads them back out, never recomputes or re-derives a
    relevance judgment."""
    ids: Set[str] = set()
    for diagnostic in diagnostics:
        if not diagnostic.message.startswith(prefix):
            continue
        match = _DIAGNOSTIC_CANDIDATE_ID_RE.search(diagnostic.message)
        if match:
            ids.add(match.group(1))
    return ids


def _article_category(article: Article, config: AppConfig) -> str:
    """Mirrors `pipeline.nodes._source_category`'s name-membership logic
    (not imported from there -- that module pulls in agents/events/LLM
    dependencies this UI-facing query layer must never depend on) purely
    for DISPLAY grouping. Never affects routing/relevance."""
    name = article.source_name
    if any(s.name == name for s in config.sources.peers):
        return "Peer Research"
    if any(s.name == name for s in config.sources.ai_providers):
        return "AI Product"
    if any(s.name == name for s in config.sources.analysts):
        return "Analyst"
    if any(s.name == name for s in config.sources.secondary_sources):
        return "Financial News"
    return "Unknown"


def get_current_scan_explorer(
    service: PersistenceService, config: AppConfig, run_id: str,
) -> Optional[CurrentScanExplorerData]:
    """Read-only view of one CURRENT run's retrieval diagnostics and actual
    outputs, for the Scan Explorer. No pipeline/LLM/network call is made
    here -- every value comes from already-persisted rows.

    Two deliberately DIFFERENT sources of truth (Phase 17, Section 5):

    - `articles` (the "Retrieved Articles" table) is RAW RETRIEVAL
      DIAGNOSTICS ONLY -- a best-effort reconstruction using each
      article's `retrieved_at` falling inside `[run.started_at,
      run.finished_at]`. `Article` rows are shared/mutable across every
      run that re-fetches the same URL, so this can legitimately include
      stale/duplicate/irrelevant content, or omit/misattribute a row for
      an older selected run if a later run re-fetched the same URL since.
      This is explicitly allowed to be approximate (never used to decide
      CURRENT membership).
    - `events` (and the research-item-derived coverage counters) use
      EXPLICIT RUN OWNERSHIP (`Event.run_id`/`ResearchItem.run_id`,
      stamped by `pipeline.nodes` from `PipelineState.run_id`) -- the
      exact same mechanism and exact same qualifying-event set the user
      CURRENT dashboard reads via this run's `digest_id`. A run persisted
      before this ownership column existed has `run_id=None` on its rows,
      so it will correctly show zero run-owned items here rather than a
      guessed, possibly-wrong reconstruction.
    """
    conn = service.database.connect()
    try:
        run = service.runs.get(conn, run_id)
        if run is None or run.digest_type != DigestType.CURRENT:
            return None

        articles = service.articles.get_retrieved_between(conn, run.started_at, run.finished_at)

        # Explicit run ownership (Section 4/9) -- never Article.retrieved_at.
        run_owned_items = service.research_items.get_by_run_id(conn, run.run_id)
        research_items_by_article = {item.article_id: item for item in run_owned_items}
        events = service.events.get_by_run_id(conn, run.run_id)

        # This run's own candidate-level diagnostics (Part C observability) --
        # distinguishes "an agent ran deep analysis and rejected it" and
        # "triage rejected it before deep analysis" from "never reached an
        # agent at all", none of which a bare ResearchItem lookup can tell
        # apart on its own.
        deep_analysis_rejected_ids = _candidate_ids_with_message_prefix(run.diagnostics, "deep_analysis:irrelevant ")
        triage_rejected_ids = _candidate_ids_with_message_prefix(run.diagnostics, "triage:irrelevant ")

        event_by_article_id = {}
        for event in events:
            for aid in event.articles:
                event_by_article_id.setdefault(aid, event)

        qualified_event_ids: set = set()
        if run.digest_id:
            digest = service.digests.get(conn, run.digest_id)
            if digest is not None:
                qualified_event_ids = set(digest.events)

        analyzed_count = len(run_owned_items)
        relevant_count = sum(1 for item in run_owned_items if item.relevance.label.value == "relevant")
        irrelevant_count = sum(1 for item in run_owned_items if item.relevance.label.value == "irrelevant")
        ambiguous_count = sum(1 for item in run_owned_items if item.relevance.label.value == "ambiguous")
        evidence_count = 0
        rejected_by_agent_count = 0

        article_rows: List[ScanArticleRow] = []
        for article in articles:
            evidence_list = service.evidence.get_by_article_id(conn, article.id)
            evidence = evidence_list[0] if evidence_list else None
            if evidence is not None:
                evidence_count += 1
            item = research_items_by_article.get(article.id)
            analyzed = item is not None
            linked_event = event_by_article_id.get(article.id)
            if analyzed and getattr(item, "grounding_type", None) is not None and item.grounding_type.value == "link":
                status = "Link-grounded"
            elif analyzed:
                status = "Analyzed"
            elif evidence is None:
                status = "Content unavailable"
            elif article.id in deep_analysis_rejected_ids:
                status = "Rejected by deep analysis"
                rejected_by_agent_count += 1
            elif article.id in triage_rejected_ids:
                status = "Rejected by triage"
                rejected_by_agent_count += 1
            else:
                status = "Not analyzed"
            article_rows.append(ScanArticleRow(
                article_id=article.id,
                source_name=article.source_name,
                category=_article_category(article, config),
                title=article.title,
                url=article.url,
                publication_date=article.publication_date,
                publication_date_precision=article.publication_date_precision.value,
                analysis_status=status,
                agent=item.agent.value if analyzed else None,
                relevance=item.relevance.label.value.capitalize() if analyzed else None,
                has_event=linked_event is not None,
                event_title=linked_event.canonical_title if linked_event else None,
                research_item_id=item.id if analyzed else None,
                event_id=linked_event.id if linked_event else None,
                evidence_id=evidence.evidence_id if evidence else None,
                content_unavailable_reason=(
                    article.metadata.get("content_unavailable_reason") if evidence is None else None
                ),
            ))

        event_rows: List[ScanEventRow] = []
        for event in events:
            research_items = service.research_items.get_many(conn, event.research_items)
            is_dated = any(ri.is_dated_development for ri in research_items) if research_items else None
            qualified = event.id in qualified_event_ids
            if qualified:
                reason = "Included"
            elif event.event_date is None:
                if any(ri.publication_date is not None and not ri.is_dated_development for ri in research_items):
                    reason = "Not a dated development"
                else:
                    reason = "Missing date"
            elif run.period_start is not None and run.period_end is not None and not (
                run.period_start <= event.event_date < run.period_end
            ):
                reason = "Outside current window"
            else:
                reason = "Other"
            report = service.reports.get(conn, event.id)
            report_title = getattr(report, "title", None) if report is not None else None
            event_rows.append(ScanEventRow(
                event_id=event.id, canonical_title=event.canonical_title,
                event_type=event.event_type.value if event.event_type else None,
                primary_company=event.primary_company, related_companies=list(event.related_companies),
                research_item_ids=list(event.research_items), article_ids=list(event.articles),
                event_date=event.event_date, is_dated_development=is_dated,
                relevance_score=event.relevance_score, confidence_score=event.confidence_score,
                report_title=report_title, digest_qualified=qualified, qualification_reason=reason,
            ))

        authoritative_analyzed = run.counts.get("agent_candidates_analyzed")
        counts_mismatch_warning = None
        if authoritative_analyzed is not None and authoritative_analyzed != analyzed_count and run_owned_items == []:
            counts_mismatch_warning = (
                f"This run's own persisted metrics say agent_candidates_analyzed={authoritative_analyzed}, but "
                f"zero run-owned ResearchItems were found (run_id={run.run_id}). This run was almost certainly "
                "persisted BEFORE explicit run ownership existed -- its ResearchItems/Events have no run_id and "
                "cannot be distinguished from any other run's output, so none are shown here rather than guessed "
                "from a mutable Article.retrieved_at reconstruction. The summary metrics above (verbatim from "
                "this run's own persisted counts) remain authoritative for what this run did."
            )

        return CurrentScanExplorerData(
            run_id=run.run_id, status=run.status.value, started_at=run.started_at, finished_at=run.finished_at,
            period_start=run.period_start, period_end=run.period_end, metrics=dict(run.counts),
            articles=article_rows, events=event_rows,
            coverage_evidence=evidence_count, coverage_analyzed=analyzed_count,
            coverage_rejected_by_agent=rejected_by_agent_count,
            coverage_not_analyzed=evidence_count - analyzed_count - rejected_by_agent_count,
            coverage_content_unavailable=len(articles) - evidence_count,
            coverage_relevant=relevant_count, coverage_irrelevant=irrelevant_count, coverage_ambiguous=ambiguous_count,
            reconstruction_note=(
                "Retrieved Articles below is RAW RETRIEVAL DIAGNOSTICS ONLY, reconstructed from each article's "
                "retrieved_at timestamp falling inside this run's [started_at, finished_at] -- Article rows are "
                "shared/mutable across every run that re-fetches the same URL, so this list may legitimately "
                "include stale, duplicate, or ultimately-irrelevant content, and can omit/misattribute a row for "
                "an older selected run. It is NEVER used to decide CURRENT membership. Events below and the "
                "analysis coverage counters use EXPLICIT RUN OWNERSHIP (Event.run_id/ResearchItem.run_id) -- the "
                "exact same mechanism and exact same qualifying-event set the user CURRENT dashboard reads."
            ),
            counts_mismatch_warning=counts_mismatch_warning,
            digest_persistence_status=(
                run.digest_persistence_status.value if run.digest_persistence_status is not None else "unavailable"
            ),
        )
    finally:
        conn.close()


def get_run_diagnostic_messages(service: PersistenceService, run_id: str, *, contains: str) -> List[str]:
    """This run's own persisted diagnostic messages matching a substring
    (e.g. the existing top_k-truncation notices) -- Part 12's agent-budget
    transparency reuses these verbatim rather than recomputing anything."""
    conn = service.database.connect()
    try:
        run = service.runs.get(conn, run_id)
    finally:
        conn.close()
    if run is None:
        return []
    return [d.message for d in run.diagnostics if contains in d.message]


def get_scan_article_detail(service: PersistenceService, config: AppConfig, article_id: str) -> Optional[ScanArticleDetail]:
    """Full detail for one article (Part 8) -- only called when the user
    opens a specific row, never for the whole listing."""
    conn = service.database.connect()
    try:
        article = service.articles.get(conn, article_id)
        if article is None:
            return None
        evidence_list = service.evidence.get_by_article_id(conn, article_id)
        evidence = evidence_list[0] if evidence_list else None
        items = service.research_items.get_by_article_id(conn, article_id)
        item = items[0] if items else None
        event_row = conn.execute(
            "SELECT event_id FROM event_articles WHERE article_id = ? LIMIT 1", (article_id,)
        ).fetchone()
        return ScanArticleDetail(
            article_id=article.id, source_name=article.source_name, title=article.title, url=article.url,
            canonical_url=article.canonical_url, publication_date=article.publication_date,
            publication_date_precision=article.publication_date_precision.value, retrieved_at=article.retrieved_at,
            category=_article_category(article, config), clean_text_available=bool(article.clean_text),
            evidence_id=evidence.evidence_id if evidence else None,
            research_item_id=item.id if item else None,
            event_id=event_row["event_id"] if event_row else None,
            agent=item.agent.value if item else None,
            relevance=item.relevance.label.value.capitalize() if item else None,
            relevance_reason=item.relevance.reason if item else None,
            is_dated_development=item.is_dated_development if item else None,
            development_date_rationale=item.development_date_rationale if item else None,
            content_unavailable_reason=(
                article.metadata.get("content_unavailable_reason") if evidence is None else None
            ),
            grounding_type=getattr(item, "grounding_type", None) if item else None,
        )
    finally:
        conn.close()


def get_recent_reports(service: PersistenceService, *, limit: int = 20) -> List[Union[Report, ReportFailure]]:
    conn = service.database.connect()
    try:
        return service.reports.get_recent(conn, limit=limit)
    finally:
        conn.close()


def get_events_for_period(service: PersistenceService, *, days: int, now: datetime, limit: int = 200) -> List[EventSummary]:
    """"What events happened over the last N days (7/30/90)?" (Phase 7 Part
    5). `now` is caller-supplied (e.g. from the same `Clock` used elsewhere)
    rather than read from the wall clock here, keeping this deterministic
    and testable like every other date-relative computation in this project."""
    since = now - timedelta(days=days)
    return search_events(service, since=since, until=now, limit=limit)


# ---------------------------------------------------------------------------
# Entity-centric monitoring pages (Phase 1): Peers/AI Providers/Analysts
# must always show every CONFIGURED entity, even with zero qualifying
# developments -- driven by the source registry, not by which entities
# happen to already have events. See `get_entity_statuses` and the
# `EntityStatus.status` closed vocabulary below.
# ---------------------------------------------------------------------------

STATUS_RECENT_DEVELOPMENTS = "Recent developments found"
STATUS_NO_QUALIFYING_DEVELOPMENTS = "No qualifying developments in selected period"
STATUS_NO_ARTICLES_RETRIEVED = "No articles retrieved"
STATUS_RETRIEVAL_FAILED = "Source retrieval failed"
STATUS_RETRIEVED_NOT_RELEVANT = "Articles retrieved but none passed relevance filtering"
STATUS_NOT_YET_ANALYZED = "Not yet analyzed"


@dataclass
class EntityStatus:
    """One row on an entity-centric monitoring page (Peers/AI
    Providers/Analysts) -- always constructed from the configured source
    registry, never from "which entities happen to have an event". An
    entity with zero qualifying developments is still a real row, with an
    explicit, distinguishable status -- never silently omitted and never
    fabricated."""

    name: str
    official_url: Optional[str]
    status: str
    articles_retrieved: int
    relevant_items: int
    event_count: int
    latest_development: Optional[EventSummary] = None
    latest_timestamp: Optional[datetime] = None
    latest_article_url: Optional[str] = None
    source_failure_reason: Optional[str] = None


def _latest_retrieve_failures(service: PersistenceService) -> dict:
    """Maps entity name -> failure reason, from the most recent pipeline
    run's `stage="retrieve"` diagnostics (see `pipeline.nodes.retrieve`,
    which logs exactly `f"{source.name}: {exc}"`). Returns {} if no run has
    ever completed, or the latest run had no retrieve-stage diagnostics."""
    conn = service.database.connect()
    try:
        recent_runs = service.runs.get_recent(conn, limit=1)
    finally:
        conn.close()
    if not recent_runs:
        return {}
    failures = {}
    for diagnostic in recent_runs[0].diagnostics:
        if diagnostic.stage != "retrieve":
            continue
        name, _, reason = diagnostic.message.partition(": ")
        if reason:
            failures[name] = reason
    return failures


def has_any_pipeline_run_completed(service: PersistenceService) -> bool:
    conn = service.database.connect()
    try:
        return len(service.runs.get_recent(conn, limit=1)) > 0
    finally:
        conn.close()


def _entity_status(
    service: PersistenceService,
    entry: SourceEntry,
    *,
    since: Optional[datetime],
    until: Optional[datetime],
    retrieve_failures: dict,
    any_run_completed: bool,
) -> EntityStatus:
    official_url = entry.urls[0] if entry.urls else None

    if not any_run_completed:
        return EntityStatus(
            name=entry.name, official_url=official_url, status=STATUS_NOT_YET_ANALYZED,
            articles_retrieved=0, relevant_items=0, event_count=0,
        )

    if entry.name in retrieve_failures:
        return EntityStatus(
            name=entry.name, official_url=official_url, status=STATUS_RETRIEVAL_FAILED,
            articles_retrieved=0, relevant_items=0, event_count=0,
            source_failure_reason=retrieve_failures[entry.name],
        )

    conn = service.database.connect()
    try:
        articles = service.articles.get_by_source_name(conn, entry.name, since=since, until=until)
        if not articles:
            return EntityStatus(
                name=entry.name, official_url=official_url, status=STATUS_NO_ARTICLES_RETRIEVED,
                articles_retrieved=0, relevant_items=0, event_count=0,
            )

        research_items = service.research_items.get_by_company(conn, entry.name, since=since, until=until)
        if not research_items:
            return EntityStatus(
                name=entry.name, official_url=official_url, status=STATUS_RETRIEVED_NOT_RELEVANT,
                articles_retrieved=len(articles), relevant_items=0, event_count=0,
            )

        events = [
            e for e in service.events.get_by_company(conn, entry.name, limit=200)
            if e.event_date is None or ((since is None or e.event_date >= since) and (until is None or e.event_date <= until))
        ]
    finally:
        conn.close()

    if not events:
        return EntityStatus(
            name=entry.name, official_url=official_url, status=STATUS_NO_QUALIFYING_DEVELOPMENTS,
            articles_retrieved=len(articles), relevant_items=len(research_items), event_count=0,
        )

    latest_event = max(events, key=lambda e: e.event_date or datetime.min.replace(tzinfo=timezone.utc))
    return EntityStatus(
        name=entry.name, official_url=official_url, status=STATUS_RECENT_DEVELOPMENTS,
        articles_retrieved=len(articles), relevant_items=len(research_items), event_count=len(events),
        latest_development=_summarize(latest_event), latest_timestamp=latest_event.event_date,
        latest_article_url=latest_event.primary_source.url,
    )


@dataclass
class ArticleStatus:
    """One retrieved article for an entity-centric page's expanded article
    list (Phase 13D) -- never just "the latest one". Every field comes
    straight from an existing, already-persisted record (Article /
    ResearchItem / Event); nothing here is computed or guessed for display
    purposes. `status`/`relevance_label`/scores are explicitly `None` when
    that stage genuinely never ran for this article, rather than a
    fabricated placeholder."""

    article_id: str
    title: Optional[str]
    publication_date: Optional[datetime]
    url: str
    status: str
    relevance_label: Optional[str] = None
    event_relevance_score: Optional[float] = None
    event_confidence_score: Optional[float] = None


ARTICLE_STATUS_EVENT_FORMED = "Event formed"
ARTICLE_STATUS_RELEVANT_NO_EVENT = "Relevant — no event yet"
ARTICLE_STATUS_NOT_RELEVANT_OR_UNANALYZED = "Not relevant / not analyzed"


def get_entity_articles(
    service: PersistenceService,
    entity_name: str,
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = 50,
) -> List[ArticleStatus]:
    """The FULL set of retrieved articles for one entity within the window
    (Phase 13D) -- not just whichever one happens to be "latest". Addresses
    the human-in-the-loop requirement that an analyst be able to inspect
    every surviving article, not only the single one the summary card
    highlights. Built entirely from existing repository reads (no new
    retrieval, no new scoring, no fabricated data) -- `get_by_source_name`
    for articles, `get_by_company` for ResearchItems and Events."""
    conn = service.database.connect()
    try:
        articles = service.articles.get_by_source_name(conn, entity_name, since=since, until=until, limit=limit)
        research_items = service.research_items.get_by_company(conn, entity_name, since=since, until=until, limit=limit)
        events = service.events.get_by_company(conn, entity_name, limit=200)
    finally:
        conn.close()

    items_by_article_id = {item.article_id: item for item in research_items}
    event_by_article_id = {}
    for event in events:
        for article_id in event.articles:
            event_by_article_id[article_id] = event

    rows = []
    for article in articles:
        item = items_by_article_id.get(article.id)
        event = event_by_article_id.get(article.id)
        if event is not None:
            status = ARTICLE_STATUS_EVENT_FORMED
        elif item is not None:
            status = ARTICLE_STATUS_RELEVANT_NO_EVENT
        else:
            status = ARTICLE_STATUS_NOT_RELEVANT_OR_UNANALYZED
        rows.append(
            ArticleStatus(
                article_id=article.id, title=article.title, publication_date=article.publication_date,
                url=article.url, status=status,
                relevance_label=item.relevance.label.value if item else None,
                event_relevance_score=event.relevance_score if event else None,
                event_confidence_score=event.confidence_score if event else None,
            )
        )

    rows.sort(key=lambda r: r.publication_date or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return rows


def get_entity_statuses(
    service: PersistenceService,
    entries: List[SourceEntry],
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> List[EntityStatus]:
    """The core Phase 1 fix: one `EntityStatus` per CONFIGURED entity
    (`entries` — e.g. `config.sources.peers`), in registry order, regardless
    of whether that entity has any events. Never fabricates data: an entity
    with no qualifying developments gets an explicit status, not an
    omission."""
    retrieve_failures = _latest_retrieve_failures(service)
    any_run_completed = has_any_pipeline_run_completed(service)
    return [
        _entity_status(
            service, entry, since=since, until=until,
            retrieve_failures=retrieve_failures, any_run_completed=any_run_completed,
        )
        for entry in entries
    ]


def get_peer_entities(service: PersistenceService, config: AppConfig, **kwargs) -> List[EntityStatus]:
    return get_entity_statuses(service, config.sources.peers, **kwargs)


def get_ai_provider_entities(service: PersistenceService, config: AppConfig, **kwargs) -> List[EntityStatus]:
    return get_entity_statuses(service, config.sources.ai_providers, **kwargs)


def get_analyst_entities(service: PersistenceService, config: AppConfig, **kwargs) -> List[EntityStatus]:
    return get_entity_statuses(service, config.sources.analysts, **kwargs)


# ---------------------------------------------------------------------------
# Scheduler operational status (Phase 14): closes the forensic-diagnosis
# finding that the UI could not distinguish "the scheduler is actually
# running" from "there just happens to be an old pipeline_runs row" --
# these are read from the `scheduler_state` table (the persistent
# scheduler process's own heartbeat, see `scheduling.scheduler`), never
# inferred from `pipeline_runs` alone.
# ---------------------------------------------------------------------------

SCHEDULER_HEALTH_RUNNING = "RUNNING"
SCHEDULER_HEALTH_NOT_CONFIRMED = "NOT_CONFIRMED"
SCHEDULER_HEALTH_STALE = "STALE"

# A heartbeat older than this many seconds (3x the scheduler's own default
# interval, see `scheduling.scheduler.DEFAULT_HEARTBEAT_INTERVAL_SECONDS`)
# is treated as stale -- generous enough to absorb a normal tick's jitter,
# never so generous that a genuinely dead process still reads "RUNNING".
DEFAULT_STALE_AFTER_SECONDS = 180


@dataclass
class SchedulerStatus:
    """Everything the Dashboard's compact operational-status section needs,
    built entirely from the persisted `scheduler_state` row -- nothing here
    is inferred or guessed. `health` is one of the three `SCHEDULER_HEALTH_*`
    constants above; every other field is `None` when the scheduler has
    never reported that fact (e.g. `last_scheduled_run_id` before the first
    scheduled run has ever fired)."""

    health: str
    started_at: Optional[datetime]
    last_heartbeat_at: Optional[datetime]
    next_morning_run_at: Optional[datetime]
    next_evening_run_at: Optional[datetime]
    last_scheduled_run_id: Optional[str]
    last_scheduled_execution_mode: Optional[str]
    last_scheduled_digest_type: Optional[str]
    last_scheduled_started_at: Optional[datetime]
    last_scheduled_finished_at: Optional[datetime]
    last_scheduled_status: Optional[str]


def get_scheduler_status(
    service: PersistenceService, *, now: Optional[datetime] = None, stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
) -> SchedulerStatus:
    """`NOT_CONFIRMED` (never a heartbeat at all -- e.g. the daemon has
    never been started against this database) is deliberately distinct
    from `STALE` (it WAS running, but its last heartbeat is older than
    `stale_after_seconds`) -- the task's own explicit requirement: "Do not
    claim 'healthy' merely because an old pipeline run exists," and "the UI
    should distinguish these three states.\""""
    now = now or datetime.now(timezone.utc)
    conn = service.database.connect()
    try:
        record = service.scheduler_state.get(conn)
    finally:
        conn.close()

    if record is None or record.last_heartbeat_at is None:
        health = SCHEDULER_HEALTH_NOT_CONFIRMED
    elif (now - record.last_heartbeat_at) > timedelta(seconds=stale_after_seconds):
        health = SCHEDULER_HEALTH_STALE
    else:
        health = SCHEDULER_HEALTH_RUNNING

    return SchedulerStatus(
        health=health,
        started_at=record.started_at if record else None,
        last_heartbeat_at=record.last_heartbeat_at if record else None,
        next_morning_run_at=record.next_morning_run_at if record else None,
        next_evening_run_at=record.next_evening_run_at if record else None,
        last_scheduled_run_id=record.last_scheduled_run_id if record else None,
        last_scheduled_execution_mode=record.last_scheduled_execution_mode if record else None,
        last_scheduled_digest_type=record.last_scheduled_digest_type if record else None,
        last_scheduled_started_at=record.last_scheduled_started_at if record else None,
        last_scheduled_finished_at=record.last_scheduled_finished_at if record else None,
        last_scheduled_status=record.last_scheduled_status if record else None,
    )


# ---------------------------------------------------------------------------
# User-facing CURRENT intelligence + newsletter curation (Phase 16).
#
# This section is a pure PRESENTATION layer over the exact same persisted
# CURRENT run/digest/event data the admin "CURRENT / Live Scan" page and
# Scan Explorer already read -- it never retrieves, clusters, scores, or
# calls an LLM, and it never introduces a second relevance engine. Strict
# CURRENT-only isolation (never silently falling back to an older CURRENT
# run, and never mixing in Morning/Evening/historical data) is enforced by
# always resolving the digest via the LATEST CURRENT run's own
# `digest_id` -- never via "whatever CURRENT digest happens to be newest",
# which could be a stale row left behind by an earlier run if the latest
# run failed or found nothing.
# ---------------------------------------------------------------------------

CURRENT_STATUS_NO_RUN = "no_run"
CURRENT_STATUS_FAILED = "failed"
CURRENT_STATUS_EMPTY = "empty"
CURRENT_STATUS_OK = "ok"


@dataclass
class CurrentEventCard:
    """One event, with exactly the fields the user-facing CURRENT page can
    show without inventing anything -- every value is copied from the
    already-persisted Event/ResearchItem/Report, never recomputed here."""

    event_id: str
    canonical_title: str
    event_type: Optional[str]
    primary_company: str
    related_companies: List[str]
    event_date: Optional[datetime]
    relevance_score: float
    confidence_score: float
    is_link_only: bool
    opening_context: Optional[str]
    key_points: List[str]
    """Entities/participants pulled straight from the Report's existing
    deterministic `details` facts (companies/partnerships/technologies/
    products) -- never an invented "key insight," since the data model has
    no such narrative-bullet field; this is the closest existing-field
    equivalent (Part 8: "do not invent fields")."""
    strategic_implications: Optional[str]
    nomura_opportunity: Optional[str]
    source_publisher: Optional[str]
    source_url: Optional[str]
    """The individual article URL discovered for this event's primary
    source (already resolved by the retrieval/routing layer -- this module
    never re-derives it). `None` only if even the general source URL
    fallback is unavailable."""
    source_link_label: str
    """"Read Original" for an individual article URL, "Open Source" only
    when falling back to a source's general/newsroom URL (Part 7) -- never
    presented as if it were the specific article."""
    is_selected: bool


@dataclass
class CurrentIntelligenceView:
    """Everything the user-facing CURRENT page needs in one call.

    `run_id`/`period_start`/`period_end`/`finished_at`/`events` always
    describe one single DISPLAYED snapshot -- the newest CURRENT run that
    is actually usable (persisted digest with events), never a mix of
    runs. `is_newer_scan_running`/`latest_attempt_outcome` describe what
    is happening with the most RECENT scan ATTEMPT, which may be a
    different (newer, not-yet-usable) run than the one being displayed --
    see `get_latest_current_intelligence`."""

    status: str  # one of the CURRENT_STATUS_* constants above
    run_id: Optional[str]
    period_start: Optional[datetime]
    period_end: Optional[datetime]
    finished_at: Optional[datetime]
    relevant_count: int
    event_count: int
    source_count: int
    link_only_count: int
    events: List[CurrentEventCard]
    is_newer_scan_running: bool = False
    """True iff a CURRENT run is in flight right now (`current_run_lock`
    held) -- this is strictly a presentation signal; it never changes
    which events are displayed."""
    latest_attempt_outcome: Optional[str] = None
    """Set (`"failed"` or `"no_qualifying_events"`) only when the single
    most recent CURRENT run attempt is NOT the run being displayed here
    (i.e. this view is showing an older, still-usable snapshot while the
    newest attempt didn't produce one). `None` whenever the displayed
    snapshot IS the latest attempt, or while a newer scan is running."""


def _resolve_source_link(event: Event, config: AppConfig) -> tuple:
    """Part 7/10-11: prefer the individual article URL already on the
    event's primary source (the retrieval layer, not this UI, is what
    turns a newsroom/hub page into individual discovered article URLs --
    see `Article.url`/`SourceRef.url`). Only when that is somehow missing
    does this fall back to the entry's configured general source URL, and
    only then does the label change to "Open Source" rather than "Read
    Original" -- a newsroom URL is never presented as an individual
    article."""
    primary = event.primary_source
    if primary.url:
        return primary.url, "Read Original"

    all_entries = (
        list(config.sources.peers) + list(config.sources.ai_providers)
        + list(config.sources.analysts) + list(config.sources.secondary_sources)
    )
    for entry in all_entries:
        if entry.name == primary.publisher and entry.urls:
            return entry.urls[0], "Open Source"
    return None, "Open Source"


def _current_event_card(detail: "EventDetail", config: AppConfig, selected_ids: set) -> CurrentEventCard:
    event = detail.event
    report = detail.report
    opening_context = report.opening_context if isinstance(report, Report) else None
    strategic_implications = report.strategic_implications if isinstance(report, Report) else None
    nomura_opportunity = report.nomura_opportunity if isinstance(report, Report) else None
    key_points: List[str] = []
    if isinstance(report, Report):
        for value in (
            report.details.companies + report.details.partnerships
            + report.details.technologies + report.details.products
        ):
            if value and value not in key_points:
                key_points.append(value)
        key_points = key_points[:6]
    source_url, source_label = _resolve_source_link(event, config)
    return CurrentEventCard(
        event_id=event.id,
        canonical_title=event.canonical_title,
        event_type=event.event_type.value if event.event_type else None,
        primary_company=event.primary_company,
        related_companies=list(event.related_companies),
        event_date=event.event_date,
        relevance_score=event.relevance_score,
        confidence_score=event.confidence_score,
        is_link_only=event.grounding_type == GroundingType.LINK,
        opening_context=opening_context,
        key_points=key_points,
        strategic_implications=strategic_implications,
        nomura_opportunity=nomura_opportunity,
        source_publisher=event.primary_source.publisher,
        source_url=source_url,
        source_link_label=source_label,
        is_selected=event.id in selected_ids,
    )


_CURRENT_SNAPSHOT_LOOKBACK = 50
"""How many of the most recent CURRENT runs `get_latest_current_intelligence`
will scan, newest-first, to find a usable snapshot to fall back to. Bounded
deliberately -- this is a presentation fallback, not an unbounded history
scan; a real CURRENT deployment produces a persisted run every day at
minimum, so 50 runs is a wide margin for "last usable snapshot" lookback."""


def _current_attempt_outcome(run) -> Optional[str]:
    """Classifies the single most recent CURRENT run ATTEMPT for banner
    purposes -- `None` means this run is itself a normal, usable snapshot
    (whether or not it's the one ultimately displayed depends on the
    caller).

    Forensic finding (CURRENT run `31fa192e4d7a7175`, 2026-10-12): a zero-
    event digest can mean two very different things -- a clean scan that
    genuinely found nothing qualifying, or a scan where triage found real
    candidates but deep analysis could not finish assessing them (Ollama
    timeouts/failures) before the run's bounded budget ran out. Both
    persist `DigestPersistenceStatus.NO_QUALIFYING_EVENTS` (correctly --
    the digest really does have zero events either way), but the
    user-facing banner must not describe the second case the same way as
    the first, since "did not produce qualifying developments" reads as
    "there was nothing of substance," not "analysis didn't finish." This
    never changes `digest_persistence_status`/`run.status` themselves --
    only which banner key this presentation layer picks."""
    if run.status.value == "failed":
        return "failed"
    if run.digest_persistence_status == DigestPersistenceStatus.NO_QUALIFYING_EVENTS:
        incomplete = run.counts.get("deep_analysis_timeouts", 0) + run.counts.get("deep_analysis_failures", 0)
        triage_relevant = run.counts.get("triage_relevant", 0)
        if incomplete > 0 and triage_relevant > 0:
            return "no_qualifying_events_incomplete_analysis"
        return "no_qualifying_events"
    if run.digest_persistence_status != DigestPersistenceStatus.PERSISTED or not run.digest_id:
        return "failed"
    return None


def _load_usable_current_snapshot(
    service: PersistenceService, config: AppConfig, run,
) -> Optional[CurrentIntelligenceView]:
    """Builds a CURRENT_STATUS_OK view for `run` iff it genuinely has a
    persisted digest with at least one real, loadable event -- else
    `None`, so the caller can keep looking further back in run history.
    Never reconstructs membership from anything but this run's own
    `digest_id` -> that digest's own `events` list."""
    if run.status.value == "failed":
        return None
    if run.digest_persistence_status != DigestPersistenceStatus.PERSISTED or not run.digest_id:
        return None

    conn = service.database.connect()
    try:
        digest = service.digests.get(conn, run.digest_id)
    finally:
        conn.close()
    if digest is None or not digest.events:
        return None

    selected_ids = set(get_newsletter_selection(service))
    cards = []
    publishers: set = set()
    for event_id in digest.events:  # already in ranked order
        detail = get_event_detail(service, event_id)
        if detail is None:
            continue
        cards.append(_current_event_card(detail, config, selected_ids))
        for ref in [detail.event.primary_source, *detail.event.secondary_sources]:
            if ref.publisher:
                publishers.add(ref.publisher)
    if not cards:
        return None

    return CurrentIntelligenceView(
        status=CURRENT_STATUS_OK,
        run_id=run.run_id, period_start=run.period_start, period_end=run.period_end, finished_at=run.finished_at,
        relevant_count=run.counts.get("research_items_relevant", len(cards)),
        event_count=len(cards),
        source_count=len(publishers),
        link_only_count=sum(1 for c in cards if c.is_link_only),
        events=cards,
    )


def get_latest_current_intelligence(service: PersistenceService, config: AppConfig) -> CurrentIntelligenceView:
    """The single entry point the user-facing CURRENT page reads.

    Displays the newest USABLE completed CURRENT snapshot (a run whose own
    `digest_id` resolves to a persisted digest with real events) -- not
    necessarily the single most recent run attempt. If the most recent
    attempt is itself usable, it IS the displayed snapshot (normal case,
    no special messaging). If the most recent attempt is still running
    (`current_run_lock` held), has failed, or completed with
    `NO_QUALIFYING_EVENTS`, this keeps showing the newest OLDER usable
    snapshot instead of a blank/failed page, while `is_newer_scan_running`/
    `latest_attempt_outcome` tell the UI what happened to the newest
    attempt so it can show an honest, non-fabricated status line.

    Still strictly CURRENT-only, still never reconstructs event membership
    from `Article.retrieved_at` or any cross-run join -- every candidate
    run is resolved purely via its own `digest_id` -> that digest's own
    `events`, exactly as before this fallback was added."""
    conn = service.database.connect()
    try:
        is_running = service.current_run_lock.get(conn) is not None
        recent_runs = service.runs.get_recent(conn, digest_type=DigestType.CURRENT, limit=_CURRENT_SNAPSHOT_LOOKBACK)
    finally:
        conn.close()

    if not recent_runs:
        return CurrentIntelligenceView(
            status=CURRENT_STATUS_NO_RUN, run_id=None, period_start=None, period_end=None, finished_at=None,
            relevant_count=0, event_count=0, source_count=0, link_only_count=0, events=[],
            is_newer_scan_running=is_running, latest_attempt_outcome=None,
        )

    latest_run = recent_runs[0]
    latest_outcome = _current_attempt_outcome(latest_run)

    for candidate in recent_runs:
        view = _load_usable_current_snapshot(service, config, candidate)
        if view is not None:
            view.is_newer_scan_running = is_running
            view.latest_attempt_outcome = latest_outcome if candidate.run_id != latest_run.run_id else None
            return view

    # No usable snapshot anywhere in the lookback window -- the ORIGINAL
    # single-run semantics for the latest attempt itself (NO_QUALIFYING_
    # EVENTS vs FAILED), now also carrying the running flag.
    base_kwargs = dict(
        run_id=latest_run.run_id, period_start=latest_run.period_start, period_end=latest_run.period_end,
        finished_at=latest_run.finished_at, is_newer_scan_running=is_running, latest_attempt_outcome=None,
    )
    if latest_outcome in ("no_qualifying_events", "no_qualifying_events_incomplete_analysis"):
        return CurrentIntelligenceView(
            status=CURRENT_STATUS_EMPTY, relevant_count=latest_run.counts.get("research_items_relevant", 0),
            event_count=0, source_count=0, link_only_count=0, events=[], **base_kwargs,
        )
    return CurrentIntelligenceView(
        status=CURRENT_STATUS_FAILED, relevant_count=0, event_count=0, source_count=0, link_only_count=0,
        events=[], **base_kwargs,
    )


# ---------------------------------------------------------------------------
# Two-tier CURRENT digest -- Tier B ("relevant articles awaiting analysis").
#
# A candidate that passed the existing CURRENT date gate and received an
# EXPLICIT passing triage decision, but never completed a qualifying
# specialist analysis this run (deep-analysis budget, timeout, or a
# recoverable failure) must remain visible rather than silently vanish.
# Sourced entirely from the existing `candidate_backlog` table -- no new
# table, no migration, no LLM call, no re-fetch of the source article.
# ---------------------------------------------------------------------------

TIER_B_SELECTION_PREFIX = "tb:"
"""Newsletter-selection id prefix for a Tier B article (vs. a bare Event
id) -- stored in the SAME `newsletter_selections` table/column, no schema
change, distinguished purely by this string prefix."""


@dataclass
class TierBArticle:
    """One Tier B card -- every field is either copied verbatim from the
    already-persisted Article/Evidence/backlog data. `key_points` (when
    non-empty) are the SOURCE'S OWN authored bullet lines, reproduced
    verbatim -- never a generated summary. `triage_confidence` is the
    triage model's own preliminary relevance screen score -- a measure of
    "how confident triage was that this is worth a closer look," never a
    measure of factual accuracy or of the content itself."""

    article_id: str
    title: Optional[str]
    publisher: Optional[str]
    publication_date: Optional[datetime]
    url: str
    excerpt: str
    key_points: List[str]
    status: str
    """`"pending_deep_analysis_budget"` or `"pending_retry"` (see
    `models.enums.BacklogStatus`) -- the backlog's own real status, never
    recomputed here."""
    reason: Optional[str]
    triage_confidence: Optional[float]


# Two-tier CURRENT digest, excerpt-quality follow-up: deterministic,
# no-LLM text processing of already-retrieved content only. Every
# transformation here either DROPS a recognizable, content-free
# boilerplate fragment or REPRODUCES existing source text verbatim --
# never paraphrases, summarizes, or invents anything.

# A wire-service dateline: "<CITY(, STATE/COUNTRY)*>, <Month> <Day>, <Year>
# /<Wire Service>/ --" (e.g. "NOIDA, India, Oct. 12, 2026 /PRNewswire/ --",
# "SINGAPORE and AMSTERDAM, Oct. 12, 2026 /PRNewswire/ --"). Extremely
# standardized across PR Newswire/Business Wire/GlobeNewswire-style
# releases and carries no substantive information -- it is the one piece
# of "generic introductory wording" that can be stripped safely and
# unambiguously by pattern alone.
_DATELINE_RE = re.compile(
    r"\b[A-Z][A-Za-z.&'’\- ]{1,60}?(?:,\s*[A-Za-z.&'’\- ]{1,40}?)?"
    r",\s*[A-Za-z]{3,10}\.?\s+\d{1,2},\s+\d{4}\s*/[^/\n]{2,50}/\s*-+\s*"
)

# A line the SOURCE itself formatted as a bullet -- "- ", "* ", or "• "
# prefix. Reproduced verbatim when used; never a line this code writes.
_BULLET_LINE_RE = re.compile(r"^[\-\*•]\s+(\S.*)$")
_MIN_BULLET_CHARS = 15  # shorter than this reads as a label/fragment, not a real point
_MIN_BULLET_COUNT = 2  # a single dash-prefixed line is more likely a stray formatting artifact than a real list
_MAX_BULLETS = 6
_MAX_BULLET_CHARS = 220

# Tier A newsletter rendering: the entities/participants list is
# de-emphasized (per user feedback, 2026-10-15) rather than removed --
# bounded so a long flat dump never crowds out the actual analysis.
_MAX_ENTITIES_SHOWN = 6


_MIN_REMAINDER_AFTER_DATELINE_STRIP = 10
"""Absolute, not relative: a real dateline is a short fixed-shape prefix
regardless of how short the surrounding article text happens to be, so
the safety guard checks how much text would be LEFT after stripping, not
what fraction of the text the match covers (a relative fraction would
wrongly reject a real, short dateline in a short article)."""


def _strip_dateline(text: str) -> str:
    """Removes the FIRST recognizable wire-service dateline only.
    Defensively refuses to apply if doing so would leave next to nothing
    behind -- a guard against a false-positive match ever eating
    substantive content; in that case the text is returned unchanged
    rather than risk dropping something real."""
    match = _DATELINE_RE.search(text)
    if not match:
        return text
    remainder = text[: match.start()] + text[match.end() :]
    if len(remainder.strip()) < _MIN_REMAINDER_AFTER_DATELINE_STRIP:
        return text
    return remainder.strip()


def _truncate_at_word(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    truncated = text[:limit].rsplit(" ", 1)[0]
    return truncated.rstrip(",;:") + "…"


_MIN_SENTENCE_FRACTION = 0.5
"""A Tier B excerpt cut mid-sentence (e.g. '...gained fresh momentum on Oct…')
reads as a dangling, uninformative fragment at a glance -- a real production
complaint, 2026-10-16. Preferring the last full sentence boundary inside the
limit reads cleanly, but only when that sentence covers at least half the
budget; otherwise a single very long lead sentence would make the excerpt
needlessly short, so `_truncate_at_word`'s word-boundary+ellipsis fallback is
used instead."""


def _truncate_at_sentence(text: str, *, limit: int) -> str:
    """Prefers ending the excerpt at the last `.`/`!`/`?` boundary within
    `limit` (a complete sentence, no ellipsis needed) over an arbitrary
    word-boundary cut -- still pure text processing of already-retrieved
    content, never paraphrased or invented."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    best_end = -1
    for punct in (".", "!", "?"):
        pos = window.rfind(punct + " ")
        if pos == -1 and window.endswith(punct):
            pos = len(window) - 1
        best_end = max(best_end, pos)
    if best_end >= limit * _MIN_SENTENCE_FRACTION:
        return window[: best_end + 1].rstrip()
    return _truncate_at_word(text, limit=limit)


def _extract_key_points(lines: List[str]) -> List[str]:
    """The source's OWN authored bullet points, verbatim -- never
    generated. Returns `[]` (not an error) when there are fewer than
    `_MIN_BULLET_COUNT` qualifying lines, so a caller always has a
    well-defined "no meaningful bullets" signal rather than a single
    stray dash-prefixed fragment being presented as a "list"."""
    points: List[str] = []
    for line in lines:
        match = _BULLET_LINE_RE.match(line)
        if not match:
            continue
        content = match.group(1).strip()
        if len(content) >= _MIN_BULLET_CHARS:
            points.append(content)
    if len(points) < _MIN_BULLET_COUNT:
        return []
    return [_truncate_at_word(p, limit=_MAX_BULLET_CHARS) for p in points[:_MAX_BULLETS]]


def _build_tier_b_presentation(text: str, *, limit: int) -> "tuple[str, List[str]]":
    """Returns `(excerpt, key_points)`. `key_points` are the source's own
    bullet lines (verbatim, see `_extract_key_points`) when present and
    meaningful; the prose excerpt always excludes those same lines (they
    are shown once, not duplicated) and the leading wire-service dateline/
    generic boilerplate, then truncates at a whole word within `limit`
    characters -- never mid-word, truncation always marked explicitly.
    Pure text processing of already-retrieved content; no LLM call, no
    paraphrasing, nothing invented."""
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    key_points = _extract_key_points(raw_lines)
    prose_lines = [ln for ln in raw_lines if not _BULLET_LINE_RE.match(ln)]

    start = 0
    while start < len(prose_lines) - 1 and len(prose_lines[start]) < 20 and not prose_lines[start].endswith((".", "!", "?")):
        start += 1
    body = " ".join(prose_lines[start:]) if prose_lines else text.strip()
    body = re.sub(r"\s+", " ", body).strip()
    body = _strip_dateline(body)
    return _truncate_at_sentence(body, limit=limit), key_points


def format_triage_confidence(value: Optional[float]) -> Optional[str]:
    """Defensive display formatting -- `None`, non-numeric, NaN, or an
    out-of-[0,1]-range value (should never persist given the schema's own
    validation, but display code never trusts that blindly) all render as
    "not available" (`None` returned) rather than a misleading number."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or not (0.0 <= numeric <= 1.0):  # numeric != numeric is the NaN check
        return None
    return f"{numeric * 100:.0f}%"


def get_tier_b_article_detail(
    service: PersistenceService, config: AppConfig, article_id: str,
    *, status: str = "", reason: Optional[str] = None, triage_confidence: Optional[float] = None,
) -> Optional[TierBArticle]:
    """Loads the real, already-persisted Article/Evidence for `article_id`
    and builds its Tier B presentation -- `None` (never fabricated
    content) if neither has usable text, or if no usable source URL
    exists. Source order for the excerpt: Evidence.content (the cleaned,
    already-extracted body) -> Article.clean_text -> Article.raw_text."""
    conn = service.database.connect()
    try:
        article = service.articles.get(conn, article_id)
        if article is None:
            return None
        evidence_list = service.evidence.get_by_article_id(conn, article_id)
    finally:
        conn.close()

    text = None
    if evidence_list and evidence_list[0].content:
        text = evidence_list[0].content
    elif article.clean_text:
        text = article.clean_text
    elif article.raw_text:
        text = article.raw_text

    url = article.url or (evidence_list[0].source_url if evidence_list else None)
    if not text or not url:
        return None

    limit = config.settings.current_llm.tier_b_excerpt_chars
    publisher = article.publisher or (evidence_list[0].publisher if evidence_list else None) or article.source_name
    excerpt, key_points = _build_tier_b_presentation(text, limit=limit)
    return TierBArticle(
        article_id=article_id, title=article.title, publisher=publisher,
        publication_date=article.publication_date, url=url,
        excerpt=excerpt, key_points=key_points, status=status, reason=reason,
        triage_confidence=triage_confidence,
    )


# Confirmed real-production defect (CURRENT run `b2d8c1466a0ade52`, 48h-
# window live scan): PR Newswire syndicates the IDENTICAL press release
# under two distinct numeric release ids ("...-302907151.html" vs
# "...-302907154.html" for the same "IFS Softeon launches Fulfillment
# Flow..." release, same for a "GFI Builds Eloquens(R) AI..." release the
# same run) -- two genuinely different URLs with two different content
# hashes, so Phase 2's article-level dedup (`evidence.dedup`, keyed on
# canonical_url/content_hash) correctly never merges them, and Tier A's
# event clustering (`events.similarity.is_near_duplicate_title_same_day`)
# never sees them AT ALL because a Tier B candidate (triage-approved but
# not yet deep-analyzed) never reaches event formation. The result: the
# identical story appears twice in the Tier B list. Fixed the same way
# Tier A's near-duplicate was fixed -- a verbatim-title, same-day override
# -- but reimplemented locally (not imported from `events.similarity`)
# since this module is structurally barred from importing
# `events.clustering`/`events.scoring` (see module docstring); this is a
# narrow, independent application of the identical idea to a different
# data shape (TierBArticle, not ResearchItem).
_TIER_B_NEAR_DUPLICATE_TITLE_THRESHOLD = 0.9
_TIER_B_NEAR_DUPLICATE_MAX_DATE_GAP_DAYS = 1.0


def _tier_b_title_tokens(title: Optional[str]) -> Set[str]:
    if not title:
        return set()
    return {t for t in re.findall(r"[a-z0-9]+", title.lower())}


def _is_near_duplicate_tier_b_article(a: "TierBArticle", b: "TierBArticle") -> bool:
    """Verbatim-title, same-day override -- deliberately conservative (a
    near-exact title match is already overwhelming evidence of the same
    underlying release; this never attempts to merge merely topically
    similar articles)."""
    if a.publication_date is None or b.publication_date is None:
        return False
    delta_days = abs((a.publication_date - b.publication_date).total_seconds()) / 86400.0
    if delta_days > _TIER_B_NEAR_DUPLICATE_MAX_DATE_GAP_DAYS:
        return False
    tokens_a, tokens_b = _tier_b_title_tokens(a.title), _tier_b_title_tokens(b.title)
    if not tokens_a or not tokens_b:
        return False
    union = tokens_a | tokens_b
    similarity = len(tokens_a & tokens_b) / len(union) if union else 0.0
    return similarity >= _TIER_B_NEAR_DUPLICATE_TITLE_THRESHOLD


def _tier_b_completeness_key(article: "TierBArticle"):
    """Keeps the most informative copy of a duplicate group -- highest
    triage confidence, then most key points, then longest excerpt --
    never merges content across copies, just picks one to show."""
    return (
        article.triage_confidence if article.triage_confidence is not None else 0.0,
        len(article.key_points),
        len(article.excerpt or ""),
    )


def _merge_near_duplicate_tier_b_articles(articles: List["TierBArticle"]) -> List["TierBArticle"]:
    n = len(articles)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for i in range(n):
        for j in range(i + 1, n):
            if _is_near_duplicate_tier_b_article(articles[i], articles[j]):
                union(i, j)

    groups: "dict[int, List[int]]" = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    kept_indices = [max(group, key=lambda idx: _tier_b_completeness_key(articles[idx])) for group in groups.values()]
    kept_indices.sort()
    return [articles[i] for i in kept_indices]


def get_current_tier_b_pending(
    service: PersistenceService, config: AppConfig, run_id: str,
) -> "tuple[List[TierBArticle], int]":
    """Every Tier B-eligible article for `run_id`, plus a count of
    candidates that WOULD have qualified but were omitted for lacking
    usable text/URL (never silently dropped without a trace -- see Part
    11/15 of the task). Excludes any article that already has a
    qualifying Tier A ResearchItem this run (structurally impossible here
    anyway: a successful analysis sets the backlog row to `completed`,
    which is not one of the statuses this query selects). Near-duplicate
    (same-day, near-verbatim title) entries are then collapsed to one --
    see `_merge_near_duplicate_tier_b_articles` -- so a press release
    syndicated under multiple URLs is never shown twice; `omitted` is
    unaffected by this (it counts only missing-text/URL omissions, a
    different and unrelated reason)."""
    conn = service.database.connect()
    try:
        rows = service.backlog.list_pending_for_run(conn, run_id=run_id)
    finally:
        conn.close()

    articles: List[TierBArticle] = []
    omitted = 0
    for row in rows:
        detail = get_tier_b_article_detail(
            service, config, row.article_id,
            status=row.status, reason=row.reason, triage_confidence=row.triage_confidence,
        )
        if detail is None:
            omitted += 1
            continue
        articles.append(detail)
    return _merge_near_duplicate_tier_b_articles(articles), omitted


def get_newsletter_selection(service: PersistenceService) -> List[str]:
    """Every currently-selected event id, in selection order -- persisted
    in SQLite (Part 9), so selections survive a Streamlit rerun (and even
    a process restart), never relying on a transient Python/session
    variable."""
    conn = service.database.connect()
    try:
        return service.newsletter.list_ids(conn)
    finally:
        conn.close()


def toggle_newsletter_selection(service: PersistenceService, event_id: str, *, now: Optional[datetime] = None) -> bool:
    """Adds `event_id` if not already selected, removes it if it is --
    exactly the "+Add to Newsletter" / "Added to Newsletter" (Part 9)
    click-to-toggle behavior. The `newsletter_selections` table's own
    PRIMARY KEY makes a duplicate selection structurally impossible; this
    never touches the Event row itself. Returns the new selected state."""
    now = now or datetime.now(timezone.utc)
    conn = service.database.connect()
    try:
        already_selected = service.newsletter.is_selected(conn, event_id)
        if already_selected:
            service.newsletter.remove(conn, event_id)
            return False
        service.newsletter.add(conn, event_id, selected_at=now)
        return True
    finally:
        conn.close()


def _render_tier_b_report_markdown(report) -> List[str]:
    """`report` is a `models.report.TierBReport` -- rendered the same
    shape as a Tier A report's sections, plus the bounded `key_insights`
    list in place of Tier A's single entities line."""
    lines = [f"**What happened**  \n{report.opening_context}", ""]
    if report.key_insights:
        lines.append("**Key insights**")
        lines.extend(f"- {insight}" for insight in report.key_insights)
        lines.append("")
    lines.append(f"**Strategic implications**  \n{report.strategic_implications}")
    lines.append("")
    lines.append(f"**Opportunity for Nomura**  \n{report.nomura_opportunity}")
    lines.append("")
    if report.content_partial:
        lines.append("_Note: the retrieved source text for this article may be incomplete._")
        lines.append("")
    return lines


def _render_tier_b_markdown(
    index: int, article: TierBArticle, resolution: Optional["TierBReportResolution"] = None,
) -> List[str]:
    lines = [f"### {index}. {article.title or article.url}", ""]

    report = resolution.report if resolution is not None else None
    if report is not None:
        lines.append("_Relevant article -- on-demand analytical report generated below._")
        lines.append("")
    else:
        lines.append("_Relevant article awaiting specialist analysis -- no generated assessment below._")
        lines.append("")

    meta_bits = [b for b in (article.publisher, article.publication_date.date().isoformat() if article.publication_date else None) if b]
    confidence_label = format_triage_confidence(article.triage_confidence)
    if confidence_label:
        meta_bits.append(f"Triage confidence: {confidence_label}")
    if meta_bits:
        lines.append(" · ".join(meta_bits))
        lines.append("")

    if report is not None:
        lines.extend(_render_tier_b_report_markdown(report))
    else:
        if resolution is not None and resolution.status == "insufficient_content":
            lines.append("_A full analytical report could not be generated: the available source "
                         "content for this article is insufficient._")
            lines.append("")
        elif resolution is not None and resolution.status == "failed":
            lines.append("_A full analytical report could not be generated for this article this time._")
            lines.append("")
        if article.key_points:
            lines.append("> **Key points from the article**")
            lines.extend(f"> - {point}" for point in article.key_points)
            lines.append("")
        lines.append(article.excerpt)
        lines.append("")

    lines.append(f"**Source**  \n[{article.publisher or 'Original article'}]({article.url})")
    lines.append("")
    return lines


def render_newsletter_markdown(
    service: PersistenceService, event_ids: List[str], config: Optional[AppConfig] = None,
    *, tier_b_reports: Optional[Dict[str, "TierBReportResolution"]] = None,
) -> str:
    """Assembles the selected newsletter from each selection's ALREADY
    RESOLVED content -- pure string formatting; this function itself
    NEVER calls an LLM or the network. `event_ids` may mix real Event ids
    (Tier A -- rendered from the event's already-persisted Report, exactly
    as before) with Tier B article selections (`TIER_B_SELECTION_PREFIX`-
    prefixed -- rendered from the already-retrieved article text, or from
    an already-resolved on-demand report if `tier_b_reports` supplies one
    -- see `reporting.tier_b_newsletter_resolution.resolve_tier_b_reports`,
    which the caller must run BEFORE calling this function; this function
    never resolves one itself). For a LINK-grounded event, the existing
    conservative LINK-grounded report (see
    `agents.report.build_link_grounded_report`) is reused verbatim, so no
    fact is fabricated just because a user selected it. `config` is only
    required when `event_ids` contains a Tier B selection (needed for the
    excerpt-length setting); omit it for Tier-A-only callers, unchanged.
    `tier_b_reports` defaults to `None`, so every pre-existing call site
    keeps its exact prior behavior (excerpt-only Tier B rendering) unless
    it explicitly opts in."""
    lines = ["# NOMURA AI & FINTECH INTELLIGENCE", "", "## Selected Intelligence Newsletter", ""]
    for index, selection_id in enumerate(event_ids, start=1):
        if selection_id.startswith(TIER_B_SELECTION_PREFIX):
            if config is None:
                continue
            article_id = selection_id[len(TIER_B_SELECTION_PREFIX):]
            article = get_tier_b_article_detail(service, config, article_id)
            if article is None:
                continue
            resolution = tier_b_reports.get(article_id) if tier_b_reports else None
            lines.extend(_render_tier_b_markdown(index, article, resolution))
            continue

        event_id = selection_id
        detail = get_event_detail(service, event_id)
        if detail is None:
            continue
        event = detail.event
        report = detail.report
        lines.append(f"### {index}. {event.canonical_title}")
        lines.append("")
        if event.grounding_type == GroundingType.LINK:
            lines.append("_Link-grounded development — article content was unavailable; details are limited to "
                         "title and publication metadata._")
            lines.append("")
        if isinstance(report, Report):
            lines.append(f"**What happened**  \n{report.opening_context}")
            lines.append("")
            if report.details.key_insights:
                lines.append("**Key insights**")
                lines.extend(f"- {insight}" for insight in report.details.key_insights)
                lines.append("")
            lines.append(f"**Strategic implications**  \n{report.strategic_implications}")
            lines.append("")
            lines.append(f"**Opportunity for Nomura**  \n{report.nomura_opportunity}")
            lines.append("")
            # De-emphasized (per user feedback: the full entity dump added
            # little value next to the development's own narrative) --
            # kept, but moved below the actual analysis and bounded to a
            # handful of names rather than a long flat list.
            entities = sorted(set(report.details.companies + report.details.partnerships))
            if entities:
                shown = entities[:_MAX_ENTITIES_SHOWN]
                remainder = len(entities) - len(shown)
                suffix = f" (+{remainder} more)" if remainder > 0 else ""
                lines.append(f"**Also mentioned**  \n{', '.join(shown)}{suffix}")
                lines.append("")
            if report.sources:
                source = report.sources[0]
                lines.append(f"**Source**  \n[{source.publisher or 'Original article'}]({source.url})")
        elif isinstance(report, ReportFailure):
            lines.append("_No generated summary is available for this event._")
            if event.primary_source.url:
                lines.append(f"**Source**  \n[{event.primary_source.publisher or 'Original article'}]({event.primary_source.url})")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Email delivery (manual newsletter sends + the recipient/auto-delivery
# preference, Sections 5-6). The one deliberate exception to this module's
# "read-only, no I/O beyond SQLite" contract described at the top of this
# file -- `send_newsletter_email` sends real mail -- kept here anyway so
# `ui/app.py` continues to import data/action helpers only from `dd`,
# exactly like every other dashboard action (`toggle_newsletter_selection`
# already writes to SQLite from this same module).
# ---------------------------------------------------------------------------


@dataclass
class EmailSettingsView:
    default_recipient: str
    auto_delivery_enabled: bool
    smtp_configured: bool


def get_email_settings_view(service: PersistenceService, config: AppConfig) -> EmailSettingsView:
    from research_assistant.delivery.mailer import build_mailer_from_config
    from research_assistant.delivery.service import get_effective_recipient_and_enabled

    recipient, enabled = get_effective_recipient_and_enabled(service, config)
    return EmailSettingsView(
        default_recipient=recipient, auto_delivery_enabled=enabled,
        smtp_configured=build_mailer_from_config(config) is not None,
    )


def set_email_default_recipient(service: PersistenceService, recipient: str) -> None:
    from research_assistant.delivery.service import set_default_recipient

    set_default_recipient(service, recipient)


def set_email_auto_delivery_enabled(service: PersistenceService, enabled: bool, config: AppConfig) -> None:
    from research_assistant.delivery.service import set_auto_delivery_enabled

    set_auto_delivery_enabled(service, enabled, config)


def is_valid_email_address(address: str) -> bool:
    from research_assistant.delivery.validation import is_valid_email

    return is_valid_email(address)


def send_newsletter_email(
    service: PersistenceService, config: AppConfig, *, event_ids: List[str], recipient: str,
    tier_b_reports: Optional[Dict[str, "TierBReportResolution"]] = None,
):
    """Sends exactly the Report/Event/Tier-B content behind `event_ids` --
    the same content `render_newsletter_markdown` would assemble for the
    same ids -- to `recipient`. Never regenerates a report or reruns the
    pipeline; `tier_b_reports` (if given) is an ALREADY-RESOLVED dict from
    `reporting.tier_b_newsletter_resolution.resolve_tier_b_reports`,
    reused here so an already-generated-for-this-click Tier B report is
    never regenerated a second time just to email it. Returns the
    `EmailDeliveryRecord` (its `.status` is `'sent'` or `'failed'`;
    `.failure_reason` is set on failure)."""
    from research_assistant.delivery.service import send_manual

    return send_manual(service, config, event_ids=event_ids, recipient=recipient, tier_b_reports=tier_b_reports)
