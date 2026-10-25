"""Streamlit dashboard — NOMURA AI & FINTECH INTELLIGENCE.

Presentation/query layer only (Phase 7 Part 12): this file never retrieves,
scrapes, clusters, scores, or calls an LLM. Every piece of data comes from
`research_assistant.ui.dashboard_data`, which itself only ever reads
through `PersistenceService`'s repositories. The database path comes from
`config/settings.yaml` (`storage.database_path`) via the existing
`get_config()` — never hardcoded here.

Run with: `streamlit run src/research_assistant/ui/app.py`
"""

from __future__ import annotations

import html
import logging
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from zoneinfo import ZoneInfo

import streamlit as st

from research_assistant.config import get_config
from research_assistant.config.loader import load_app_login_password
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, EventType
from research_assistant.models.report import Report, ReportFailure
from research_assistant.storage import Database, PersistenceService
from research_assistant.storage.repositories.current_run_lock import probe_owner_liveness
from research_assistant.ui import dashboard_data as dd
from research_assistant.utils.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Nomura AI & Fintech Intelligence", layout="wide")

_SIDEBAR_PAGES = (
    "CURRENT",
    "Morning Digest",
    "Evening Digest",
    "Dashboard",
    "Peers",
    "AI Providers",
    "Startups",
    "Analysts",
    "Cross-Market Themes",
    "CURRENT / Live Scan (Admin)",
)

# Post-login landing page: the task this login gate was built for is
# "log in -> fire a live scan -> get notified when it's done -> see the
# CURRENT digest" -- all of which already lives on this one page (the
# "Run Live Current Scan now" button, the completion toast, and "Events
# in last CURRENT digest" directly below it). Only applied as the
# `st.sidebar.radio` widget's INITIAL selection (via its `index=`, which
# Streamlit only honors the first time a given `key` appears in
# `st.session_state` -- i.e. once per login) -- the user can still freely
# navigate elsewhere afterwards, exactly as before.
_DEFAULT_LANDING_PAGE = "CURRENT / Live Scan (Admin)"

# Peers/AI Providers/Analysts are ENTITY-CENTRIC (Phase 1): driven by the
# configured source registry, so every configured entity is always shown,
# even with zero events. Startups remains EVENT-CENTRIC/discovery-driven
# (there is no fixed startup registry to enumerate — see
# `retrieval.service._SCRAPABLE_CATEGORIES`).
_ENTITY_PAGE_REGISTRY = {
    "Peers": lambda config: config.sources.peers,
    "AI Providers": lambda config: config.sources.ai_providers,
    "Analysts": lambda config: config.sources.analysts,
}
_STARTUP_AGENT = AgentName.STARTUP.value


@st.cache_resource
def _get_persistence_service() -> PersistenceService:
    config = get_config()
    database_path = PROJECT_ROOT / config.settings.storage.database_path
    database = Database(database_path)
    database.initialize()
    return PersistenceService(database)


def _events_table(summaries) -> List[dict]:
    rows = []
    for summary in summaries:
        row = asdict(summary)
        row["event_type"] = summary.event_type.value if summary.event_type else None
        row["event_date"] = summary.event_date.isoformat() if summary.event_date else None
        rows.append(row)
    return rows


def _render_event_picker(service: PersistenceService, summaries, *, key: str) -> Optional[str]:
    if not summaries:
        st.info("No events match the current filters.")
        return None
    options = {f"{s.canonical_title} — {s.primary_company} ({s.id})": s.id for s in summaries}
    label = st.selectbox("Select an event for detail", list(options.keys()), key=key)
    return options[label]


def _render_event_detail(service: PersistenceService, event_id: str) -> None:
    detail = dd.get_event_detail(service, event_id)
    if detail is None:
        st.warning("Event not found.")
        return

    event = detail.event
    st.subheader(event.canonical_title)
    st.caption(
        f"{event.primary_company} · "
        f"{event.event_type.value if event.event_type else 'unclassified'} · "
        f"{event.event_date.isoformat() if event.event_date else 'date unknown'}"
    )

    col1, col2 = st.columns(2)
    col1.metric("Relevance score", f"{event.relevance_score:.2f}")
    col2.metric("Confidence score", f"{event.confidence_score:.2f}")

    if event.description:
        st.write(event.description)
    if event.related_companies:
        st.write("**Related companies:**", ", ".join(event.related_companies))

    with st.expander("Scoring breakdown"):
        st.json(event.scoring_breakdown.model_dump(mode="json"))

    with st.expander("Cluster provenance"):
        st.json(event.cluster_metadata.model_dump(mode="json"))

    st.markdown("**Linked research items**")
    for item in detail.research_items:
        st.write(f"- [{item.agent.value}] {item.title or item.url}")

    st.markdown("**Sources**")
    for source in [event.primary_source, *event.secondary_sources]:
        st.write(f"- ({source.source_type.value}/{source.source_priority.value}) {source.publisher or 'unknown'}: {source.url}")

    st.markdown("**Generated report**")
    if detail.report is None:
        st.info("No report has been generated for this event yet.")
    elif isinstance(detail.report, ReportFailure):
        st.warning(f"Report generation failed: {detail.report.reason} — {detail.report.detail}")
    else:
        report: Report = detail.report
        st.write("**Opening context**")
        st.write(report.opening_context)
        st.write("**Details**")
        st.write(report.details.narrative)
        st.write("**Strategic implications**")
        st.write(report.strategic_implications)
        st.write("**Opportunity for Nomura**")
        st.write(report.nomura_opportunity)

    with st.expander("Source evidence"):
        for evidence in dd.get_source_evidence(service, event_id):
            st.write(f"**{evidence.publisher or 'unknown publisher'}** — {evidence.title or evidence.source_url}")
            st.caption(
                f"{evidence.source_type.value}/{evidence.source_priority.value} · "
                f"{evidence.publication_date.isoformat() if evidence.publication_date else 'date unknown'} · "
                f"{evidence.source_url}"
            )
            st.write(evidence.content)


def _render_filters(service: PersistenceService):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    days = st.sidebar.selectbox("Date range (days)", [7, 30, 90], index=1)
    company = st.sidebar.text_input("Company contains")
    event_type_label = st.sidebar.selectbox("Event type", ["Any"] + [e.value for e in EventType])
    min_relevance = st.sidebar.slider("Minimum relevance", 0.0, 1.0, 0.0, 0.05)
    min_confidence = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    event_type = EventType(event_type_label) if event_type_label != "Any" else None

    return dd.search_events(
        service,
        since=since,
        until=now,
        company=company or None,
        event_type=event_type,
        min_relevance=min_relevance or None,
        min_confidence=min_confidence or None,
        limit=200,
    )


def _render_run_status(service: PersistenceService) -> None:
    """Read-only "last run" status. This page never triggers a pipeline run
    itself (Streamlit stays a presentation layer — see the module
    docstring); it only reports the most recently persisted
    `PipelineRunResult` and tells the operator how to trigger a new one."""
    run = dd.get_latest_run_summary(service)
    with st.container(border=True):
        if run is None:
            st.warning("No pipeline run has completed yet.")
        else:
            icon = {"success": "🟢", "partial_success": "🟡", "failed": "🔴"}.get(run.status.value, "⚪")
            st.write(f"{icon} **Last run:** {run.status.value} — finished {run.finished_at.isoformat()}")
            st.caption(f"run_id={run.run_id} · events_formed={run.counts.get('events_formed', 'n/a')}")
        st.caption(
            "To run a new analysis now: "
            "`python -m research_assistant.scheduling.cli --digest-type manual` "
            "(covers today's business-day start through the current time)."
        )


_SCHEDULER_HEALTH_ICON = {
    dd.SCHEDULER_HEALTH_RUNNING: "🟢",
    dd.SCHEDULER_HEALTH_STALE: "🔴",
    dd.SCHEDULER_HEALTH_NOT_CONFIRMED: "⚪",
}


def _fmt(ts: Optional[datetime], config) -> str:
    if ts is None:
        return "—"
    tz = ZoneInfo(config.settings.system.timezone)
    return ts.astimezone(tz).strftime("%Y-%m-%d %H:%M %Z")


def _local_date_isoformat(ts: Optional[datetime]) -> Optional[str]:
    """A date BADGE shown next to an article/event must read as the same
    calendar day a reader opening the source article in their own browser
    (local wall-clock time) would see -- manager-reported confusion,
    2026-10-16: an article opened directly showed "15th Oct" while this
    page's badge showed "16th". Root cause: every OTHER timestamp in this
    UI (`_fmt`/`_fmt_period`, run start/finish times) is already converted
    to `config.settings.system.timezone` (Asia/Kolkata) before display, but
    these three badges called `.date()` directly on the raw, UTC-stored
    `event_date`/`publication_date` -- UTC late-evening instants (e.g.
    22:00-23:59 UTC) land on the NEXT calendar day once shifted +5:30,
    exactly the one-day-ahead mismatch reported. Uses `get_config()`
    directly (the same singleton every other call site already reads)
    rather than threading a `config` parameter through three card-rendering
    functions that don't otherwise need it."""
    if ts is None:
        return None
    tz = ZoneInfo(get_config().settings.system.timezone)
    return ts.astimezone(tz).date().isoformat()


def _render_operational_status(service: PersistenceService, config) -> None:
    """Phase 14: makes the forensic-diagnosis's core finding immediately
    visible -- "no operational digest" and "the scheduler isn't running"
    are different facts, and a reader must never have to infer the second
    from the first. Built entirely from `get_scheduler_status` (the
    persisted scheduler heartbeat), never from `pipeline_runs` alone."""
    status = dd.get_scheduler_status(service)
    with st.container(border=True):
        st.markdown("**Operational status**")
        icon = _SCHEDULER_HEALTH_ICON.get(status.health, "⚪")
        label = {
            dd.SCHEDULER_HEALTH_RUNNING: "RUNNING",
            dd.SCHEDULER_HEALTH_STALE: "STALE — heartbeat is older than expected; the scheduler process may have stopped",
            dd.SCHEDULER_HEALTH_NOT_CONFIRMED: "NOT CONFIRMED — no scheduler heartbeat has ever been recorded against this database",
        }.get(status.health, status.health)
        st.write(f"{icon} **Scheduler:** {label}")

        cols = st.columns(4)
        cols[0].caption("Last heartbeat")
        cols[0].write(_fmt(status.last_heartbeat_at, config))
        cols[1].caption("Next Morning run")
        cols[1].write(_fmt(status.next_morning_run_at, config))
        cols[2].caption("Next Evening run")
        cols[2].write(_fmt(status.next_evening_run_at, config))
        cols[3].caption("Scheduler started")
        cols[3].write(_fmt(status.started_at, config))

        if status.last_scheduled_run_id:
            st.caption(
                f"Last scheduled run: {status.last_scheduled_digest_type} · {status.last_scheduled_status} · "
                f"finished {_fmt(status.last_scheduled_finished_at, config)} (run_id={status.last_scheduled_run_id})"
            )
        else:
            st.caption(
                "No scheduled run has ever completed against this database -- every run below is distinguishable "
                "as manual vs. scheduled via its own execution_mode."
            )

        if status.health != dd.SCHEDULER_HEALTH_RUNNING:
            st.warning(
                "If a digest page shows 'No operational digest', check here first: an absent digest for a "
                "scheduler that is NOT RUNNING or STALE means the pipeline likely never executed for that "
                "period -- it is not evidence that nothing relevant happened."
            )


def _render_dashboard_page(service: PersistenceService, config) -> None:
    st.header("Dashboard")
    _render_operational_status(service, config)
    _render_run_status(service)

    st.subheader("Latest Intelligence")
    latest = dd.get_latest_intelligence(service, limit=20)
    st.dataframe(_events_table(latest), use_container_width=True)

    st.subheader("Market Activity")
    activity = dd.get_market_activity(service)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**By event type**")
        st.dataframe([asdict(c) for c in activity["by_event_type"]], use_container_width=True)
    with col2:
        st.write("**By company**")
        st.dataframe([asdict(c) for c in activity["by_company"]], use_container_width=True)

    st.subheader("High-Relevance Events")
    high_relevance = dd.get_high_relevance_events(service, min_relevance=0.5, limit=50)
    st.dataframe(_events_table(high_relevance), use_container_width=True)

    st.subheader("Filtered Events")
    filtered = _render_filters(service)
    st.dataframe(_events_table(filtered), use_container_width=True)

    st.subheader("Event Detail")
    event_id = _render_event_picker(service, filtered or latest, key="dashboard_event_picker")
    if event_id:
        _render_event_detail(service, event_id)


def _render_digest_page(service: PersistenceService, digest_type: DigestType, config) -> None:
    st.header(f"{digest_type.value.title()} Digest")

    now = datetime.now(timezone.utc)
    from research_assistant.scheduling.period import compute_digest_period

    expected_start, expected_end = compute_digest_period(digest_type, now, config)
    digest = dd.get_operational_digest(service, digest_type, config, now=now)
    if digest is None:
        tz = ZoneInfo(config.settings.system.timezone)
        st.info(
            f"No operational {digest_type.value} digest for "
            f"{expected_start.astimezone(tz).strftime('%Y-%m-%d %H:%M %Z')} – "
            f"{expected_end.astimezone(tz).strftime('%Y-%m-%d %H:%M %Z')}. "
            "A historical/validation digest, if one exists, is never shown "
            "here in its place."
        )
        return

    st.caption(f"Generated {digest.generated_at.isoformat()} — period {digest.period_start.isoformat()} to {digest.period_end.isoformat()}")

    st.subheader("Themes")
    for theme in digest.themes:
        st.write(f"**{theme.label}** — {theme.description or ''} ({len(theme.event_ids)} event(s))")

    st.subheader("Events")
    for event_id in digest.events:
        detail = dd.get_event_detail(service, event_id)
        if detail is None:
            continue
        _render_digest_event(detail)
        st.divider()


def _fmt_period(ts: Optional[datetime], config) -> str:
    if ts is None:
        return "—"
    tz = ZoneInfo(config.settings.system.timezone)
    return ts.astimezone(tz).strftime("%b %d, %Y %I:%M %p %Z")


_CURRENT_PAGE_CSS = """
<style>
.nomura-eyebrow {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
    color: #6b7280; margin-bottom: 0.1rem;
}
.nomura-badge {
    display: inline-block; padding: 0.14rem 0.55rem; border-radius: 999px; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.02em; margin-right: 0.4rem; margin-bottom: 0.3rem;
    border: 1px solid transparent;
}
.nomura-badge-score { background: #eef2ff; color: #3730a3; border-color: #c7d2fe; }
.nomura-badge-muted { background: #f3f4f6; color: #374151; border-color: #e5e7eb; }
.nomura-badge-link { background: #fef3c7; color: #92400e; border-color: #fde68a; }
.nomura-badge-source { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
.nomura-badge-date { background: #fdf2f8; color: #9d174d; border-color: #fbcfe8; }
.nomura-section-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase;
    color: #4b5563; margin-top: 0.7rem; margin-bottom: 0.2rem;
}
.nomura-divider { border: none; border-top: 1px solid #e5e7eb; margin: 0.6rem 0 0.9rem 0; }
.nomura-key-points-box {
    background: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 6px;
    padding: 0.5rem 0.8rem; margin: 0.4rem 0 0.6rem 0; font-size: 0.92rem;
}
.nomura-key-points-box ul { margin: 0.2rem 0 0 0; padding-left: 1.1rem; }
.nomura-summary-box {
    background: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 6px;
    padding: 0.5rem 0.8rem; margin: 0.4rem 0 0.6rem 0; font-size: 0.92rem; color: #1f2937;
}
</style>
"""


def _inject_current_page_css() -> None:
    st.markdown(_CURRENT_PAGE_CSS, unsafe_allow_html=True)


def _badge(text: str, *, variant: str = "muted") -> str:
    return f'<span class="nomura-badge nomura-badge-{variant}">{text}</span>'


def _render_summary_box(text: str) -> None:
    """Shared "what happened" styling -- same box, same color, for both
    Tier A's `opening_context` and Tier B's raw excerpt, so the summary
    section reads identically regardless of which tier a card is in."""
    st.markdown(f'<div class="nomura-summary-box">{html.escape(text)}</div>', unsafe_allow_html=True)


def _render_key_points_box(points: List[str], *, label: str) -> None:
    """Shared key-points styling -- same label treatment and box color
    for both Tier A's entity bullets and Tier B's verbatim source bullets."""
    st.markdown(f'<div class="nomura-section-label">{html.escape(label)}</div>', unsafe_allow_html=True)
    points_html = "<ul>" + "".join(f"<li>{html.escape(point)}</li>" for point in points) + "</ul>"
    st.markdown(f'<div class="nomura-key-points-box">{points_html}</div>', unsafe_allow_html=True)


def _render_current_event_card(card: "dd.CurrentEventCard", service: PersistenceService) -> None:
    """One user-facing event card (Part 8-9) -- every field is read
    straight off the already-computed `CurrentEventCard`; this function
    only renders, it never queries the database or recomputes anything."""
    with st.container(border=True):
        st.markdown(f"**{card.canonical_title}**")

        badges = [_badge(f"Score {card.relevance_score * 100:.0f}", variant="score")]
        if card.event_type:
            badges.append(_badge(card.event_type.replace("_", " ").title()))
        if card.is_link_only:
            badges.append(_badge("LINK-GROUNDED", variant="link"))
        if card.source_publisher:
            badges.append(_badge(f"Source: {html.escape(card.source_publisher)}", variant="source"))
        if card.event_date:
            badges.append(_badge(_local_date_isoformat(card.event_date), variant="date"))
        st.markdown(" ".join(badges), unsafe_allow_html=True)

        if card.is_link_only:
            st.caption(
                "Article content was unavailable for automated retrieval; this development was identified from "
                "its title and publication metadata only — no claim below is based on reading the article body."
            )

        if card.opening_context:
            _render_summary_box(card.opening_context)

        if card.strategic_implications:
            st.markdown('<div class="nomura-section-label">Strategic Implications</div>', unsafe_allow_html=True)
            st.write(card.strategic_implications)

        if card.nomura_opportunity:
            st.markdown('<div class="nomura-section-label">Opportunity for Nomura</div>', unsafe_allow_html=True)
            st.write(card.nomura_opportunity)

        if card.related_companies:
            st.caption("Related: " + ", ".join(card.related_companies))

        st.markdown('<hr class="nomura-divider" />', unsafe_allow_html=True)
        link_col, add_col = st.columns([1, 1])
        with link_col:
            if card.source_url:
                st.link_button(f"{card.source_link_label} ↗", card.source_url, use_container_width=True)
            else:
                st.caption("No source link available")
        with add_col:
            button_label = "✓ Added to Newsletter" if card.is_selected else "+ Add to Newsletter"
            if st.button(button_label, key=f"nl_toggle_{card.event_id}", type=("secondary" if card.is_selected else "primary"), use_container_width=True):
                dd.toggle_newsletter_selection(service, card.event_id)
                st.rerun()


_TIER_B_STATUS_LABELS = {
    "pending_deep_analysis_budget": "Awaiting specialist analysis (deep-analysis budget not yet reached)",
    "pending_retry": "Awaiting specialist analysis (previous attempt timed out or failed; eligible for retry)",
}


def _render_tier_b_card(article: "dd.TierBArticle", service: PersistenceService) -> None:
    """One Tier B card -- deliberately NO relevance/confidence-of-fact
    score, no generated summary, no inferred event type, no strategic
    implications. `triage_confidence` (when available) is shown labelled
    exactly as what it is -- the triage model's preliminary relevance
    screen, never a measure of the article's factual accuracy. Every
    other field is either copied verbatim from the already-retrieved
    Article/Evidence (`key_points` are the SOURCE's own authored bullets,
    reproduced as-is) or the backlog's own recorded status."""
    with st.container(border=True):
        st.markdown(f"**{article.title or article.url}**")
        badges = [_badge(_TIER_B_STATUS_LABELS.get(article.status, "Awaiting specialist analysis"), variant="link")]
        if article.publisher:
            badges.append(_badge(f"Source: {article.publisher}", variant="source"))
        confidence_label = dd.format_triage_confidence(article.triage_confidence)
        if confidence_label:
            badges.append(_badge(f"Triage confidence {confidence_label}", variant="score"))
        if article.publication_date:
            badges.append(_badge(_local_date_isoformat(article.publication_date), variant="date"))
        st.markdown(" ".join(badges), unsafe_allow_html=True)
        if confidence_label:
            st.caption("Triage confidence reflects a preliminary relevance screen, not factual accuracy.")
        if article.key_points:
            _render_key_points_box(article.key_points, label="Key points from the article")
        if article.excerpt:
            _render_summary_box(article.excerpt)
        st.markdown('<hr class="nomura-divider" />', unsafe_allow_html=True)
        selection_id = f"{dd.TIER_B_SELECTION_PREFIX}{article.article_id}"
        is_selected = selection_id in dd.get_newsletter_selection(service)
        link_col, add_col = st.columns([1, 1])
        with link_col:
            st.link_button("Read Original ↗", article.url, use_container_width=True)
        with add_col:
            button_label = "✓ Added to Newsletter" if is_selected else "+ Add to Newsletter"
            if add_col.button(
                button_label, key=f"nl_toggle_tb_{article.article_id}",
                type=("secondary" if is_selected else "primary"), use_container_width=True,
            ):
                dd.toggle_newsletter_selection(service, selection_id)
                st.rerun()


def _render_tier_b_section(service: PersistenceService, config) -> None:
    """Tier B -- "Relevant articles awaiting analysis": renders whenever a
    latest CURRENT run exists, independent of whether Tier A (analysed
    developments) has anything to show, so a run where every winner timed
    out/was budget-skipped still surfaces its triage-approved candidates
    instead of presenting an empty page. Scoped to the LATEST run attempt
    specifically (not necessarily the Tier A snapshot being displayed,
    which may fall back to an older usable run) -- Tier B is about what
    THIS run's own triage approved but could not finish analyzing."""
    latest_run = dd.get_latest_run_for_digest_type(service, DigestType.CURRENT)
    st.subheader("Relevant articles awaiting analysis")
    if latest_run is None:
        st.caption("No articles currently awaiting specialist analysis.")
        return
    articles, omitted = dd.get_current_tier_b_pending(service, config, latest_run.run_id)
    if not articles:
        st.caption("No articles currently awaiting specialist analysis for the latest scan.")
        if omitted:
            st.caption(f"({omitted} candidate(s) were eligible but omitted -- no usable retrieved text or source URL.)")
        return
    st.caption(
        f"{len(articles)} article(s) passed relevance triage this run but have not yet completed specialist "
        "analysis (deep-analysis budget, timeout, or a recoverable failure) -- shown here as retrieved, with "
        "no generated assessment."
    )
    for article in articles:
        _render_tier_b_card(article, service)
    if omitted:
        st.caption(f"({omitted} additional candidate(s) were eligible but omitted -- no usable retrieved text or source URL.)")


def _render_newsletter_section(service: PersistenceService, config) -> None:
    """The bottom-of-page newsletter curation panel (Part 12-14) -- reads
    only the persisted `newsletter_selections` table and each selected
    event's already-persisted Report; never triggers a scan, retrieval, or
    a new LLM call. Extended (email-delivery feature) with Send-to-self /
    Send-to-someone-else controls using exactly the generated newsletter's
    content -- never a regenerated or re-selected one."""
    st.divider()
    st.header("Newsletter")
    _render_automatic_delivery_settings_isolated(service, config)

    selected_ids = dd.get_newsletter_selection(service)
    st.markdown(f"**{len(selected_ids)} event(s) selected**")

    if not selected_ids:
        st.caption("Select one or more developments to generate a newsletter.")
        st.session_state.pop("newsletter_markdown", None)
        st.session_state.pop("newsletter_event_ids", None)
        return

    st.caption("Selected developments will be used to generate the newsletter.")
    for selection_id in selected_ids:
        is_tier_b = selection_id.startswith(dd.TIER_B_SELECTION_PREFIX)
        tier_b_article_id = selection_id[len(dd.TIER_B_SELECTION_PREFIX):] if is_tier_b else None
        if is_tier_b:
            article = dd.get_tier_b_article_detail(service, config, tier_b_article_id)
            title = f"{article.title or article.url} (awaiting analysis)" if article else selection_id
        else:
            detail = dd.get_event_detail(service, selection_id)
            title = detail.event.canonical_title if detail else selection_id
        row = st.columns([4, 1.4, 1])
        row[0].write(f"- {title}")
        if is_tier_b:
            from research_assistant.reporting.tier_b_newsletter_resolution import is_content_sufficient_for_report

            sufficient = is_content_sufficient_for_report(service, config, tier_b_article_id)
            if sufficient is False:
                if row[1].button("Fetch full article", key=f"nl_fetch_{selection_id}"):
                    from research_assistant.reporting.tier_b_newsletter_resolution import fetch_full_article_for_tier_b

                    with st.spinner("Fetching the full article (one request, this URL only)..."):
                        result = fetch_full_article_for_tier_b(service, config, tier_b_article_id)
                    if result.status == "improved":
                        st.success(f"Got more content ({result.new_text_chars} chars) -- ready for report generation.")
                    elif result.status == "no_improvement":
                        st.info("The live page didn't yield more usable text than what's already stored.")
                    else:
                        st.warning(f"Could not fetch more content ({result.detail or result.status}).")
                    st.rerun()
        else:
            # Tier A only -- re-runs the EXISTING ReportAgent/link-grounded
            # path for this one already-formed event, from its already-
            # persisted ResearchItem facts (no re-discovery, no re-fetch,
            # no specialist re-analysis), overwriting its existing `reports`
            # row. Explicit, per-event, never automatic -- selecting an
            # item or clicking "Generate Newsletter" never calls this. See
            # `reporting.tier_a_report_regeneration.regenerate_tier_a_report`.
            if row[1].button("Regenerate report", key=f"nl_regen_{selection_id}"):
                from research_assistant.reporting.tier_a_report_regeneration import regenerate_tier_a_report

                with st.spinner("Regenerating this report (one local model call)..."):
                    result = regenerate_tier_a_report(service, config, selection_id)
                if result is None:
                    st.warning("That event could no longer be found.")
                elif isinstance(result, dd.Report):
                    st.success("Report regenerated.")
                else:
                    st.warning(f"Regeneration failed ({getattr(result, 'reason', 'unknown')}).")
                st.rerun()
        if row[-1].button("Remove", key=f"nl_remove_{selection_id}"):
            dd.toggle_newsletter_selection(service, selection_id)
            st.rerun()

    if st.button("Generate Newsletter", type="primary", disabled=not selected_ids):
        tier_b_ids = [
            sid[len(dd.TIER_B_SELECTION_PREFIX):] for sid in selected_ids if sid.startswith(dd.TIER_B_SELECTION_PREFIX)
        ]
        tier_b_resolutions = {}
        if tier_b_ids:
            # The ONE place a Tier B newsletter report may be generated --
            # never on selection/deselection, never inside rendering. A
            # bounded, strictly sequential operation (no concurrency, no
            # background job queue): see `reporting.
            # tier_b_newsletter_resolution.resolve_tier_b_reports` and its
            # own `max_tier_b_reports_per_newsletter_request` cap.
            try:
                with st.spinner("Resolving Tier B reports (this may call the local model once per new article)..."):
                    from research_assistant.reporting.tier_b_newsletter_resolution import resolve_tier_b_reports

                    tier_b_resolutions = resolve_tier_b_reports(service, config, tier_b_ids)
            except Exception:
                logger.exception("ui: Tier B report resolution failed; falling back to excerpt-only rendering")
                st.warning("Could not generate full Tier B reports this time; showing excerpts instead.")
                tier_b_resolutions = {}
        st.session_state["newsletter_markdown"] = dd.render_newsletter_markdown(
            service, selected_ids, config, tier_b_reports=tier_b_resolutions,
        )
        st.session_state["newsletter_event_ids"] = list(selected_ids)
        st.session_state["newsletter_tier_b_reports"] = tier_b_resolutions

    generated_ids = st.session_state.get("newsletter_event_ids")
    markdown = st.session_state.get("newsletter_markdown")
    if markdown and generated_ids:
        with st.container(border=True):
            st.markdown(markdown)
        dl_col1, dl_col2 = st.columns(2)
        dl_col1.download_button(
            "Download newsletter (.md)", data=markdown, file_name="nomura_newsletter.md", mime="text/markdown",
        )
        # Pure, in-memory re-render of the SAME already-generated Markdown
        # above -- never a second data-gathering pass, never an LLM call
        # (see `reporting.newsletter_pdf`'s own docstring). A failure here
        # only disables the PDF button for this render; it never affects
        # the Markdown download, the selection, or any persisted report.
        try:
            from research_assistant.reporting.newsletter_pdf import render_newsletter_pdf

            pdf_bytes = render_newsletter_pdf(markdown)
            dl_col2.download_button(
                "Download newsletter (.pdf)", data=pdf_bytes, file_name="nomura_newsletter.pdf", mime="application/pdf",
            )
        except Exception:
            logger.exception("ui: newsletter PDF generation failed")
            dl_col2.caption("PDF download is unavailable right now.")
        if generated_ids != list(selected_ids):
            st.caption("Selection changed since this newsletter was generated — regenerate before sending.")
        else:
            tier_b_resolutions = st.session_state.get("newsletter_tier_b_reports") or {}
            _render_newsletter_send_controls_isolated(service, config, generated_ids, tier_b_resolutions)


def _render_automatic_delivery_settings_isolated(service: PersistenceService, config) -> None:
    """Newsletter generation/selection/viewing must work even if the entire
    email subsystem is unavailable (disabled config, missing SMTP, a
    raising email-settings lookup, etc.) -- this wrapper is the isolation
    boundary: any exception from the email-settings panel is caught and
    reported HERE, in the email section itself, and never propagates up
    into `_render_newsletter_section` and breaks newsletter generation."""
    try:
        _render_automatic_delivery_settings(service, config)
    except Exception:
        logger.exception("ui: automatic delivery settings panel failed to render")
        st.error("Automatic email delivery settings are unavailable right now. Newsletter generation is unaffected.")


def _render_newsletter_send_controls_isolated(service: PersistenceService, config, event_ids: list, tier_b_reports: Optional[dict] = None) -> None:
    """Same isolation boundary as `_render_automatic_delivery_settings_isolated`,
    for the manual Send controls -- a failure here must never hide the
    already-generated newsletter content displayed just above it."""
    try:
        _render_newsletter_send_controls(service, config, event_ids, tier_b_reports)
    except Exception:
        logger.exception("ui: newsletter send controls failed to render")
        st.error("Email sending controls are unavailable right now. The newsletter above is unaffected.")


def _render_automatic_delivery_settings(service: PersistenceService, config) -> None:
    """Section 5: a simple way to see/change the automatic-delivery
    recipient and to turn automatic delivery on/off -- never exposes an
    SMTP credential. Applies to automatic CURRENT/Morning/Evening digest
    delivery, not to a specific manual newsletter send below."""
    settings_view = dd.get_email_settings_view(service, config)
    with st.expander("Automatic digest email delivery settings", expanded=False):
        st.caption(
            "Successfully generated CURRENT/Morning/Evening digests are automatically emailed to this address."
            + ("" if settings_view.smtp_configured else " (SMTP is not configured yet, so delivery is currently inactive.)")
        )
        st.markdown(f"**Current recipient:** `{settings_view.default_recipient}`")

        new_enabled = st.checkbox(
            "Automatic delivery enabled", value=settings_view.auto_delivery_enabled, key="email_auto_delivery_enabled",
        )
        if new_enabled != settings_view.auto_delivery_enabled:
            dd.set_email_auto_delivery_enabled(service, new_enabled, config)
            st.rerun()

        new_recipient = st.text_input(
            "Change default recipient", value=settings_view.default_recipient, key="email_default_recipient_input",
        )
        if st.button("Save recipient", key="email_save_recipient"):
            if dd.is_valid_email_address(new_recipient):
                dd.set_email_default_recipient(service, new_recipient)
                st.success(f"Default recipient updated to {new_recipient}.")
                st.rerun()
            else:
                st.error("Not a valid email address.")


def _render_newsletter_send_controls(service: PersistenceService, config, event_ids: list, tier_b_reports: Optional[dict] = None) -> None:
    """Option A (send to the saved default recipient) and Option B (send
    to an explicitly entered address) -- Section 6. Each send is an
    explicit, separate button click; nothing here runs automatically.
    `tier_b_reports` (already resolved by the "Generate Newsletter" click
    above, if this selection had any Tier B entries) is reused as-is --
    sending an email never triggers a second report-generation pass for
    the same selection."""
    settings_view = dd.get_email_settings_view(service, config)
    st.subheader("Email this newsletter")
    if not settings_view.smtp_configured:
        st.caption("Email delivery is not configured yet (SMTP settings/credentials are missing) — sending is disabled.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Send to yourself**  \n`{settings_view.default_recipient}`")
        if st.button(
            "Send to myself", key="nl_send_self", disabled=not settings_view.smtp_configured,
            type="primary",
        ):
            result = dd.send_newsletter_email(
                service, config, event_ids=event_ids, recipient=settings_view.default_recipient,
                tier_b_reports=tier_b_reports,
            )
            if result is not None and result.status == "sent":
                st.success(f"Newsletter sent to {settings_view.default_recipient}.")
            else:
                reason = result.failure_reason if result is not None else "unknown error"
                st.error(f"Failed to send to {settings_view.default_recipient}: {reason}")

    with col_b:
        other_recipient = st.text_input("Send to someone else", key="nl_other_recipient", placeholder="name@example.com")
        is_valid = dd.is_valid_email_address(other_recipient) if other_recipient else False
        if other_recipient and not is_valid:
            st.caption("Not a valid email address.")
        if st.button("Send to this address", key="nl_send_other", disabled=not is_valid or not settings_view.smtp_configured):
            result = dd.send_newsletter_email(
                service, config, event_ids=event_ids, recipient=other_recipient, tier_b_reports=tier_b_reports,
            )
            if result is not None and result.status == "sent":
                st.success(f"Newsletter sent to {other_recipient}.")
            else:
                reason = result.failure_reason if result is not None else "unknown error"
                st.error(f"Failed to send to {other_recipient}: {reason}")


_CURRENT_ATTEMPT_BANNERS = {
    "failed": "The latest live scan did not produce a new available snapshot. Showing the most recent available intelligence.",
    "no_qualifying_events": "Latest scan completed but did not produce qualifying developments. Showing the most recent available intelligence.",
    # Distinct from the above (forensic finding, run `31fa192e4d7a7175`):
    # triage found genuinely relevant candidates, but specialist analysis
    # could not finish assessing them (model timeout/failure) before the
    # run's bounded budget ran out -- never worded the same as "nothing
    # qualifying was found," which would misleadingly suggest a complete,
    # successful analysis found no substance.
    "no_qualifying_events_incomplete_analysis": (
        "Latest scan found candidates worth a closer look, but specialist analysis could not finish "
        "assessing them this run (model timeout/failure). They remain queued for a later scan. "
        "Showing the most recent available intelligence."
    ),
}


def _render_current_header(view: "dd.CurrentIntelligenceView", config) -> None:
    st.markdown('<div class="nomura-eyebrow">Latest Available Intelligence</div>', unsafe_allow_html=True)
    if view.finished_at:
        st.subheader(f"Last completed live scan: {_fmt_period(view.finished_at, config)}")

    if view.is_newer_scan_running:
        st.caption("A new live scan is currently running. Results will appear here when the scan completes successfully.")
    elif view.latest_attempt_outcome in _CURRENT_ATTEMPT_BANNERS:
        st.caption(_CURRENT_ATTEMPT_BANNERS[view.latest_attempt_outcome])


def _render_user_current_page(service: PersistenceService, config) -> None:
    """The normal USER's dashboard (Part 2-13): presents the latest
    persisted CURRENT run's result only. No "Run Live Scan" control, no
    source/LLM/retrieval diagnostics -- those remain admin-only on the
    "CURRENT / Live Scan (Admin)" page. Opening or refreshing this page
    never triggers a scan or any network/LLM call -- it only reads
    `dashboard_data.get_latest_current_intelligence`, itself a read-only
    query over already-persisted rows. `main()` already renders the
    app-wide title, so this page starts at its own section header."""
    _inject_current_page_css()
    st.header("CURRENT — Live Intelligence")

    view = dd.get_latest_current_intelligence(service, config)

    # CURRENT_STATUS_OK is the only status in which a usable persisted
    # CURRENT snapshot (this run's own, or an older one still being shown
    # while a newer attempt hasn't produced a usable one yet) exists.
    # NO_RUN/FAILED/EMPTY here specifically mean no CURRENT snapshot has
    # EVER been usable -- there is nothing to fall back to.
    if view.status != dd.CURRENT_STATUS_OK:
        with st.container(border=True):
            st.subheader("No Current Intelligence Available")
            st.write("No completed CURRENT live scan has produced qualifying intelligence yet.")
            st.caption("Please check back after the next administrative live scan.")
            if view.is_newer_scan_running:
                st.caption(
                    "A new live scan is currently running. Results will appear here when the scan "
                    "completes successfully."
                )
        st.divider()
        _render_tier_b_section(service, config)
        return

    _render_current_header(view, config)

    metric_cols = st.columns(4)
    with metric_cols[0].container(border=True):
        st.metric("Relevant Articles", view.relevant_count)
    with metric_cols[1].container(border=True):
        st.metric("Events", view.event_count)
    with metric_cols[2].container(border=True):
        st.metric("Sources", view.source_count)
    with metric_cols[3].container(border=True):
        st.metric("Link-only Developments", view.link_only_count)

    st.subheader("Relevant Developments")
    for card in view.events:
        _render_current_event_card(card, service)

    st.divider()
    _render_tier_b_section(service, config)

    _render_newsletter_section(service, config)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _launch_current_scan_subprocess() -> "tuple[int, str, object]":
    """Launches the existing manual CLI entry point
    (`python -m research_assistant.scheduling.cli --digest-type current`)
    as a detached OS process -- never a new scheduler, never a second
    pipeline implementation. `stdout`/`stderr` go to a real log file
    (never a `PIPE`, which would risk a buffer-full deadlock for a run
    that can produce a lot of log output over 20-40+ minutes); `stdin` is
    `DEVNULL` so the child never blocks waiting on input. On Windows,
    `CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS` detaches it from this
    Streamlit process's console/job object, so a Streamlit crash/restart
    cannot take the scan down with it, and vice versa. Returns
    `(pid, log_path, proc)` -- the lock/ownership mechanism inside the
    subprocess (`PipelineRunner.run`'s existing `current_run_lock.
    try_acquire`) remains the actual source of truth for whether the scan
    is RUNNING; `proc` (the live `Popen` handle) is kept only so the
    caller can durably detect a startup failure (a real OS exit code)
    before that lock ever exists -- see `_render_current_page`'s own
    docstring section on this. Raises `OSError` (e.g. `FileNotFoundError`,
    `PermissionError`) if the log directory/file or the process itself
    could not be created -- the caller is responsible for catching this
    and showing a clean failure message; this function never swallows it."""
    import subprocess
    import sys

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"current_scan_{_utcnow().strftime('%Y%m%dT%H%M%SZ')}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    try:
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        proc = subprocess.Popen(
            [sys.executable, "-m", "research_assistant.scheduling.cli", "--digest-type", "current"],
            cwd=str(PROJECT_ROOT), stdout=log_file, stderr=log_file, stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    finally:
        log_file.close()  # the child has its own duplicated handle; safe to close here
    return proc.pid, str(log_path), proc


def _render_current_page(service: PersistenceService, config) -> None:
    """CURRENT / Live Scan (final production gate extension): the ONE page
    in this file that is allowed to trigger a real pipeline run -- every
    other page here stays strictly read-only (see module docstring). It
    still uses the exact same `PipelineRunner`/`run_current_digest` the CLI
    uses -- never a second, duplicate pipeline -- and never touches the
    scheduler's own lifecycle (no daemon start/stop/restart here)."""
    window_hours = config.settings.research.current_window_hours
    st.header("CURRENT / Live Scan")
    st.caption(
        f"A manually-triggered, rolling {window_hours}-hour scan covering [now - {window_hours}h, now), "
        "independent of the 08:00/20:00 scheduler. This is a separate operational product from Morning/"
        "Evening (anchored to those scheduled hours) and from the historical inventory "
        "(a much wider rolling retention window, shown elsewhere) -- never mixed together."
    )

    last_run = dd.get_latest_run_for_digest_type(service, DigestType.CURRENT)
    last_digest = dd.get_latest_digest(service, DigestType.CURRENT)

    with st.container(border=True):
        st.markdown("**Last CURRENT run**")
        if last_run is None:
            st.info("No CURRENT scan has ever been run against this database.")
        else:
            icon = {"success": "🟢", "partial_success": "🟡", "failed": "🔴"}.get(last_run.status.value, "⚪")
            st.write(f"{icon} status={last_run.status.value} — finished {_fmt_period(last_run.finished_at, config)}")
            st.caption(
                f"Period scanned: {_fmt_period(last_run.period_start, config)} .. "
                f"{_fmt_period(last_run.period_end, config)} (run_id={last_run.run_id})"
            )
            counts = last_run.counts
            # Full funnel, stage by stage -- deliberately more granular than
            # a single "Agent candidates analyzed" number, which previously
            # read "0" even on a run where Qwen triage had genuinely run
            # several real batches (it only ever reflected DEEP-analysis
            # attempts, a later and much narrower stage). Every number here
            # is an already-computed, already-persisted `counts_json` key --
            # this only changes what the admin page displays, not what is
            # measured or stored.
            st.caption("Retrieval & evidence")
            row1 = st.columns(4)
            row1[0].metric("Sources attempted", counts.get("sources_attempted", "n/a"))
            row1[1].metric("Candidates discovered", counts.get("candidates_discovered", "n/a"))
            row1[2].metric("Evidence created", counts.get("evidence_created", "n/a"))
            row1[3].metric("Date-qualified (hard gate)", counts.get("current_date_qualified", "n/a"))

            st.caption("Routing & triage (Qwen)")
            row2 = st.columns(4)
            row2[0].metric("Structurally eligible", counts.get("candidates_after_deterministic_gate", "n/a"))
            row2[1].metric("Sent to triage", counts.get("candidates_sent_to_triage", "n/a"))
            row2[2].metric("Triage Qwen batches", counts.get("triage_batches_attempted", "n/a"))
            row2[3].metric("Triage relevant winners", counts.get("triage_relevant", "n/a"))

            st.caption("Deep analysis (Qwen) & output")
            row3 = st.columns(4)
            row3[0].metric("Deep Qwen calls attempted", counts.get("deep_analysis_attempted", "n/a"))
            row3[1].metric("Deep Qwen successes", counts.get("deep_analysis_success", "n/a"))
            row3[2].metric("Deep Qwen timeouts/failures", counts.get("deep_analysis_timeouts", 0) + counts.get("deep_analysis_failures", 0))
            row3[3].metric("ResearchItems created", counts.get("research_items_total", "n/a"))

            st.caption("Events & digest")
            row4 = st.columns(4)
            row4[0].metric("Relevant developments", counts.get("research_items_relevant", "n/a"))
            row4[1].metric("Events formed", counts.get("events_formed", "n/a"))
            row4[2].metric("Digest-qualified events", counts.get("digest_event_count", "n/a"))
            row4[3].write(f"**Digest status:**\n\n{last_run.digest_persistence_status.value}")
            if last_run.diagnostics:
                with st.expander(f"Diagnostics ({len(last_run.diagnostics)})"):
                    for diagnostic in last_run.diagnostics:
                        st.caption(f"[{diagnostic.severity}] {diagnostic.stage}: {diagnostic.message}")

    conn = service.database.connect()
    try:
        lock = service.current_run_lock.get(conn)
    finally:
        conn.close()

    # Connection-reliability fix: the scan previously ran SYNCHRONOUSLY
    # inside this Streamlit script execution (`run_current_digest(runner)`
    # called directly, blocking under `st.spinner` for the scan's entire
    # duration -- often 20-40+ minutes of real retrieval + sequential
    # local-Ollama calls). Any interruption of THIS Streamlit process
    # during that window (observed repeatedly: the host's own memory-
    # pressure process reaper killing the Streamlit server mid-scan) took
    # the in-flight scan down with it -- from the browser's perspective,
    # exactly the reported "Connection error" (the server process itself
    # stopped responding, not a mere WebSocket hiccup). The scan is now
    # launched as a genuinely separate, detached OS process -- the EXACT
    # same entry point the existing manual CLI uses
    # (`python -m research_assistant.scheduling.cli --digest-type
    # current`, which itself calls the unchanged `PipelineRunner.build()` +
    # `run_current_digest`) -- so Streamlit returns immediately and stays
    # responsive, and the scan survives a Streamlit restart/crash (and vice
    # versa: a scan crash can no longer take Streamlit down with it).
    # Mutual exclusion is unchanged -- still the existing `current_run_lock`
    # singleton, now keyed to the SUBPROCESS's own pid rather than
    # Streamlit's, which is strictly more correct for the dead-pid-reclaim
    # logic in `current_run_lock.try_acquire`.
    recently_launched = st.session_state.get("current_scan_launch_at")
    launch_cooldown_active = recently_launched is not None and (_utcnow() - recently_launched).total_seconds() < 10

    # Audit fix (stale-lock UI deadlock): the first version of this fix
    # disabled the button whenever ANY lock row existed, dead owner or
    # not -- since reclaiming a stale lock only happens inside the
    # EXISTING `try_acquire` (which only ever runs on a real launch
    # attempt), that made a stale lock permanently unrecoverable from the
    # UI (the button that would trigger reclaim could never be clicked).
    # `probe_owner_liveness` is purely a PRESENTATION signal -- it never
    # touches the lock row itself, and the actual reclaim decision stays
    # entirely inside `try_acquire`'s own unchanged, authoritative
    # `_is_process_alive` check. "indeterminate" is deliberately treated
    # the same as "alive" (held) here, never as reclaimable -- the task's
    # own "if the owner cannot be determined to be dead, treat the lock
    # as held" rule.
    owner_liveness = probe_owner_liveness(lock.owner_pid) if lock is not None else None
    lock_is_stale = owner_liveness == "dead"

    # Completion notification: while a (non-stale) CURRENT scan is in
    # flight, this page keeps itself watching via `current_scan_watching`
    # in `st.session_state` -- purely a presentation flag, server-memory
    # only, never a new table. The moment the run-ownership lock this
    # session was watching disappears (the scan released it -- success or
    # failure, it no longer matters which), a one-shot toast fires so the
    # user doesn't have to keep manually re-checking this page. The
    # digest/run panels above already re-read fresh from the database on
    # every rerun, so by the time this toast appears the page has in fact
    # already updated itself -- "refresh to see it" is true regardless.
    if lock is not None and not lock_is_stale:
        st.session_state["current_scan_watching"] = True
    elif st.session_state.pop("current_scan_watching", False):
        st.toast("CURRENT scan finished -- the digest below is up to date.", icon="✅")

    # Popen-failure/startup-failure tracking (audit fixes B and C): the
    # live `Popen` object for THIS session's own most recent launch
    # attempt, kept in `st.session_state` (server-memory only -- no new
    # table, no migration) so its real OS-level exit code can be polled
    # across reruns/refreshes within the same browser session. Cleared
    # the moment it is no longer informative: once the real lock exists
    # (normal takeover -- the authoritative run/lock status takes over
    # completely) or once the child has actually exited (resolved, one
    # way or the other). Never present for a NEW session/reconnect that
    # didn't itself launch anything -- that is an explicit, documented
    # limitation (see the final report), not a bug: there is no durable,
    # schema-free way to recover "my last launch's exact exit code" across
    # a Streamlit server restart or a different browser session.
    pending_proc = st.session_state.get("current_scan_launch_proc")
    startup_failure_message = None
    still_initializing = False
    if pending_proc is not None:
        if lock is not None:
            # Normal takeover: the child reached the lock. Stop polling --
            # the authoritative lock/run status above is now the whole
            # story.
            st.session_state.pop("current_scan_launch_proc", None)
        else:
            exit_code = pending_proc.poll()  # None iff the child is still running -- a real OS fact, never a guess/timeout
            if exit_code is None:
                still_initializing = True
            else:
                st.session_state.pop("current_scan_launch_proc", None)
                if exit_code != 0:
                    startup_failure_message = (
                        f"The last scan launch (pid={pending_proc.pid}) exited immediately with code "
                        f"{exit_code} -- it never reached the point of starting a run (no lock was ever "
                        f"acquired). See its log for details: {st.session_state.get('current_scan_launch_log')}"
                    )
                # exit_code == 0 with no lock ever having appeared is left
                # unexplained rather than guessed at -- see the report's
                # documented limitation; it does not happen in practice
                # (the CLI only exits 0 after persisting a run, which
                # requires having held the lock), but this code never
                # asserts a specific cause it cannot verify.

    button_disabled = (
        (lock is not None and not lock_is_stale) or launch_cooldown_active or still_initializing
    )

    if lock is not None:
        if lock_is_stale:
            st.warning(
                f"A previous CURRENT scan (run_id={lock.run_id}, started "
                f"{_fmt_period(lock.started_at, config)}) appears to have stopped without releasing its "
                f"run lock -- its owning process is no longer running. Click the button below to safely "
                f"start a new scan; the existing lock-reclaim mechanism will take over this stale lock "
                f"automatically (nothing is deleted manually here)."
            )
        else:
            liveness_note = " (liveness could not be verified -- treated conservatively as active)" if owner_liveness == "indeterminate" else ""
            st.warning(
                f"CURRENT scan already running\nrun_id = {lock.run_id} "
                f"(started {_fmt_period(lock.started_at, config)}){liveness_note}"
            )
    elif still_initializing:
        st.info(f"Scan launch (pid={pending_proc.pid}) is starting up — waiting for it to register...")
    elif startup_failure_message:
        st.error(startup_failure_message)
    elif launch_cooldown_active:
        st.info("Scan launch just requested — waiting for it to register before allowing another click.")

    if st.button("Run Live Current Scan now", type="primary", disabled=button_disabled):
        try:
            pid, log_path, proc = _launch_current_scan_subprocess()
        except OSError as exc:
            logger.exception("ui: failed to launch the CURRENT scan subprocess")
            st.error(f"Failed to launch the CURRENT scan: {exc}")
        else:
            st.session_state["current_scan_launch_at"] = _utcnow()
            st.session_state["current_scan_launch_proc"] = proc
            st.session_state["current_scan_launch_log"] = log_path
            st.success(
                f"CURRENT scan launched in the background (pid={pid}). It calls the real local Ollama model "
                f"sequentially and can take 20-40+ minutes. This page stays responsive — refresh it (or "
                f"revisit this page) any time to check status; the status above is read from the same "
                f"run-ownership lock the scan itself uses, so it survives a page reload. Log: {log_path}"
            )
            st.rerun()

    st.subheader("Events in last CURRENT digest")
    if last_digest is None:
        st.info("No CURRENT digest has ever been persisted.")
    elif not last_digest.events:
        st.info(f"No qualifying developments found in the last {window_hours} hours. This is a valid, successful empty result.")
    else:
        for event_id in last_digest.events:
            detail = dd.get_event_detail(service, event_id)
            if detail is None:
                continue
            _render_digest_event(detail)
            st.divider()

    st.divider()
    _render_current_scan_explorer(service, config)

    # Auto-poll while a scan this page is watching is still in flight, so
    # the completion toast above fires on its own -- the user never has to
    # click anything to learn a scan finished, only to keep this browser
    # tab open. Deliberately a BROWSER-side `<meta http-equiv="refresh">`
    # reload rather than a server-side `time.sleep` + `st.rerun()` loop:
    # the latter would block a real server thread for the sleep duration
    # on every single poll (expensive for a local, long-running scan this
    # page may watch for 20-40+ minutes) and -- confirmed by this file's
    # own `AppTest`-based tests in `tests/ui/test_current_scan_launch.py`,
    # several of which leave an active lock in place across multiple
    # `.run()` calls -- would spin that harness forever waiting for a lock
    # that the test never releases. A static `<meta>` tag is inert under
    # `AppTest` (no real browser, so no reload ever happens there) and
    # costs nothing server-side between reloads.
    if st.session_state.get("current_scan_watching"):
        st.markdown('<meta http-equiv="refresh" content="8">', unsafe_allow_html=True)


def _render_current_pipeline_funnel(m: dict, digest_persistence_status: str) -> None:
    """Stage-by-stage CURRENT funnel, reusing ONLY this run's own already-
    persisted `counts_json` (`m`, verbatim, never recomputed) -- so a zero-
    Event run can be explained by WHERE in the pipeline candidates were
    lost, not just "0 events formed". Every number here already existed in
    `pipeline_runs.counts_json` before this change; this only surfaces it.
    A key absent from an older run's persisted counts (before it existed)
    renders as "n/a", never a fabricated 0."""

    def g(key):
        return m.get(key, "n/a")

    st.subheader("Pipeline Funnel")
    st.caption(
        "Where candidates were lost at each stage, using this run's own persisted counters -- "
        "never recomputed or guessed. \"n/a\" means this run predates that counter, not zero."
    )

    st.markdown("**1. Discovery**")
    c = st.columns(4)
    c[0].metric("Sources attempted", g("sources_attempted"))
    c[1].metric("Sources failed", g("sources_failed"))
    c[2].metric("Candidates discovered", g("candidates_discovered"))
    c[3].metric("After dedup", g("articles_after_dedup"))

    st.markdown("**2. Wide-lookback retention (`normalize`)**")
    c = st.columns(3)
    c[0].metric("Excluded, out of lookback", g("articles_excluded_out_of_lookback"))
    c[1].metric("...of which had a known date", g("articles_excluded_out_of_lookback_with_known_date"))
    c[2].metric("Evidence created", g("evidence_created"))

    st.markdown("**3. CURRENT hard date gate** (`[period_start, period_end)`)")
    c = st.columns(4)
    c[0].metric("Period candidates", g("current_period_candidates"))
    c[1].metric("Date-qualified", g("current_date_qualified"))
    c[2].metric("Dropped — undated", g("current_dropped_undated"))
    c[3].metric("Dropped — stale / future", f"{g('current_dropped_stale')} / {g('current_dropped_future')}")

    st.markdown("**4. Routing**")
    c = st.columns(4)
    c[0].metric("Peer", g("routed_peer"))
    c[1].metric("AI/Technology", g("routed_ai_technology"))
    c[2].metric("Startup/Funding", g("routed_startup_funding"))
    c[3].metric("Unrouted", g("routed_unrouted"))

    st.markdown("**5. Qwen triage** (batched relevance screen)")
    c = st.columns(4)
    c[0].metric("Sent to triage", g("candidates_sent_to_triage"))
    c[1].metric("Triage: relevant", g("triage_relevant"))
    c[2].metric("Triage: irrelevant", g("triage_irrelevant"))
    c[3].metric("Triage: failures", g("triage_failures"))

    st.markdown("**6. Deep analysis** (bounded budget)")
    c = st.columns(4)
    c[0].metric("Budget", g("deep_analysis_budget"))
    c[1].metric("Attempted", g("deep_analysis_attempted"))
    c[2].metric("Succeeded", g("deep_analysis_success"))
    c[3].metric("Timeouts / failures", f"{g('deep_analysis_timeouts')} / {g('deep_analysis_failures')}")

    st.markdown("**7. Events & digest**")
    c = st.columns(3)
    c[0].metric("Events formed", g("events_formed"))
    c[1].metric("Digest events", g("digest_event_count"))
    c[2].metric("Digest persistence status", digest_persistence_status)

    skipped = m.get("current_candidates_skipped_by_budget")
    if skipped:
        st.caption(
            f"{skipped} candidate(s) that passed the deterministic date gate were never sent to triage at all "
            "this run, purely due to the LLM batch/budget cap -- not judged irrelevant."
        )


def _render_current_scan_explorer(service: PersistenceService, config) -> None:
    """Read-only CURRENT Scan Explorer: what the system actually retrieved,
    analyzed, and formed into events for a selected CURRENT run -- even
    when the digest itself is empty. Never triggers retrieval, an LLM
    call, or a database write; every value comes from already-persisted
    rows (see `dashboard_data.get_current_scan_explorer`)."""
    st.header("CURRENT Scan Explorer")
    st.caption(
        "Transparent, read-only view of one CURRENT run's retrieved articles and formed events -- "
        "including the items an empty digest would otherwise hide."
    )

    run_options = dd.get_current_run_options(service, limit=20)
    if not run_options:
        st.info("No CURRENT run has ever been recorded.")
        return

    labels = [
        f"{_fmt_period(r.started_at, config)} — {r.status} ({r.run_id})" for r in run_options
    ]

    # Refresh fix (forensic finding, run `b9eb35cb5362b752`): `st.selectbox`'s
    # `key`-bound session state persists the user's selection across every
    # rerun, so `index=0` is only ever consulted the FIRST time this widget
    # renders in a session -- a newly-completed CURRENT run appearing at
    # the top of a freshly-requeried `labels` list never became the
    # selection on its own, because the OLD label (still present further
    # down the same list) remained "selected". Tracked separately from the
    # widget's own key: only when the newest run's id actually CHANGES
    # since the last time this page rendered do we clear the stored
    # selection so the widget falls back to its `index=0` default (the new
    # newest run) -- an ordinary rerun with no new run, or the user's own
    # deliberate selection of an older run, is never disturbed.
    newest_run_id = run_options[0].run_id
    if st.session_state.get("scan_explorer_newest_seen_run_id") != newest_run_id:
        st.session_state["scan_explorer_newest_seen_run_id"] = newest_run_id
        st.session_state.pop("scan_explorer_run", None)

    selected_label = st.selectbox("CURRENT runs", labels, index=0, key="scan_explorer_run")
    selected_run_id = run_options[labels.index(selected_label)].run_id

    data = dd.get_current_scan_explorer(service, config, selected_run_id)
    if data is None:
        st.warning("Could not load this run.")
        return

    st.caption(data.reconstruction_note)
    if data.counts_mismatch_warning:
        st.warning(data.counts_mismatch_warning)

    st.subheader("CURRENT Scan")
    cols = st.columns(3)
    cols[0].write(f"**Run ID:** {data.run_id}")
    cols[1].write(f"**Status:** {data.status}")
    cols[2].write(f"**Started:** {_fmt_period(data.started_at, config)}")
    cols = st.columns(3)
    cols[0].write(f"**Finished:** {_fmt_period(data.finished_at, config)}")
    cols[1].write(f"**Period start:** {_fmt_period(data.period_start, config)}")
    cols[2].write(f"**Period end:** {_fmt_period(data.period_end, config)}")

    m = data.metrics
    row1 = st.columns(3)
    row1[0].metric("Candidates discovered", m.get("candidates_discovered", "n/a"))
    row1[1].metric("Evidence created", m.get("evidence_created", "n/a"))
    row1[2].metric("Agent candidates analyzed", m.get("agent_candidates_analyzed", "n/a"))
    row2 = st.columns(3)
    row2[0].metric("Relevant developments", m.get("research_items_relevant", "n/a"))
    row2[1].metric("Events formed", m.get("events_formed", "n/a"))
    row2[2].metric("Digest-qualified events", len([e for e in data.events if e.digest_qualified]))

    _render_current_pipeline_funnel(m, data.digest_persistence_status)

    st.subheader("Analysis coverage")
    cov_cols = st.columns(5)
    cov_cols[0].write(f"Evidence items (reconstructed): **{data.coverage_evidence}**")
    cov_cols[1].write(f"Analyzed by agents: **{data.coverage_analyzed}**")
    cov_cols[2].write(f"Rejected by an agent: **{data.coverage_rejected_by_agent}**")
    cov_cols[3].write(f"Not analyzed — reason unavailable: **{data.coverage_not_analyzed}**")
    cov_cols[4].write(f"Content unavailable: **{data.coverage_content_unavailable}**")
    st.caption(
        "\"Rejected by an agent\" means triage or deep analysis actually ran and judged the candidate not "
        "relevant -- it was analyzed, just not kept. \"Not analyzed — reason unavailable\" means it has "
        "Evidence but never produced a ResearchItem and was never flagged irrelevant by triage/deep analysis "
        "diagnostics either -- this run does not persist a per-article record of WHY (it could be the CURRENT "
        "hard date gate dropping it as undated/stale/future before routing, the LLM triage/deep-analysis "
        "budget cap, or routing to no agent at all). See the Pipeline Funnel above for this run's own "
        "AGGREGATE counts for each of those causes -- they just cannot currently be attributed back to this "
        "specific article. \"Content unavailable\" means a URL was discovered but its article body could not "
        "be retrieved (robots.txt, anti-bot, HTTP error, or browser-extraction failure) -- it was never "
        "analyzed and is never counted as irrelevant."
    )
    cov_cols = st.columns(3)
    cov_cols[0].write(f"Relevant: **{data.coverage_relevant}**")
    cov_cols[1].write(f"Irrelevant: **{data.coverage_irrelevant}**")
    cov_cols[2].write(f"Ambiguous: **{data.coverage_ambiguous}**")

    # Part 12: agent budget transparency -- reused verbatim from this run's
    # own persisted diagnostics (the existing top_k-truncation notices),
    # never recomputed/invented.
    budget_messages = dd.get_run_diagnostic_messages(service, data.run_id, contains="top_k=")
    if budget_messages:
        with st.expander("Agent budget transparency"):
            for message in budget_messages:
                st.caption(message)

    st.subheader("Retrieved Articles")
    if not data.articles:
        st.info("No articles reconstructed for this run.")
    else:
        sources = sorted({a.source_name for a in data.articles})
        categories = sorted({a.category for a in data.articles})
        filter_cols = st.columns(4)
        selected_sources = filter_cols[0].multiselect("Source", sources, default=[], key="scan_src")
        selected_categories = filter_cols[1].multiselect("Agent / category", categories, default=[], key="scan_cat")
        analysis_filter = filter_cols[2].selectbox(
            "Analysis status",
            ["All", "Analyzed", "Link-grounded", "Rejected by deep analysis", "Rejected by triage", "Not analyzed", "Content unavailable"],
            key="scan_analysis",
        )
        relevance_filter = filter_cols[3].selectbox(
            "Relevance", ["All", "Relevant", "Irrelevant", "Ambiguous", "Not analyzed"], key="scan_relevance",
        )
        event_filter = st.selectbox("Event status", ["All", "Has Event", "No Event"], key="scan_event")

        filtered = data.articles
        if selected_sources:
            filtered = [a for a in filtered if a.source_name in selected_sources]
        if selected_categories:
            filtered = [a for a in filtered if a.category in selected_categories]
        if analysis_filter != "All":
            filtered = [a for a in filtered if a.analysis_status == analysis_filter]
        if relevance_filter != "All":
            if relevance_filter == "Not analyzed":
                filtered = [a for a in filtered if a.relevance is None]
            else:
                filtered = [a for a in filtered if a.relevance == relevance_filter]
        if event_filter != "All":
            filtered = [a for a in filtered if a.has_event == (event_filter == "Has Event")]

        st.caption(f"{len(filtered)} of {len(data.articles)} retrieved articles shown")
        table_rows = [
            {
                "Source": a.source_name,
                "Article": a.title or "(untitled)",
                "Published": a.publication_date.isoformat() if a.publication_date else "Unknown",
                "Precision": a.publication_date_precision,
                "Category": a.category,
                "Analysis status": a.analysis_status,
                "Relevance": a.relevance or "Not analyzed",
                "Event": a.event_title or "",
                "Content unavailable reason": a.content_unavailable_reason or "",
                "URL": a.url or "URL unavailable",
            }
            for a in filtered
        ]
        st.dataframe(table_rows, width="stretch", hide_index=True)

        article_lookup = {f"{a.title or a.article_id} ({a.source_name})": a for a in filtered}
        if article_lookup:
            chosen_label = st.selectbox("Open article detail", list(article_lookup.keys()), key="scan_article_detail")
            chosen_row = article_lookup[chosen_label]
            detail = dd.get_scan_article_detail(service, config, chosen_row.article_id)
            if detail is not None:
                with st.container(border=True):
                    st.write(f"**{detail.title or '(untitled)'}**")
                    url_display = detail.url or "URL unavailable"
                    if detail.url:
                        st.markdown(f"[{url_display}]({detail.url})")
                    else:
                        st.write(url_display)
                    st.caption(
                        f"Source: {detail.source_name} | Category: {detail.category} | "
                        f"Published: {detail.publication_date.isoformat() if detail.publication_date else 'Unknown'} "
                        f"({detail.publication_date_precision}) | Retrieved: {detail.retrieved_at.isoformat()}"
                    )
                    st.caption(
                        f"Clean text available: {detail.clean_text_available} | "
                        f"Evidence ID: {detail.evidence_id or 'n/a'} | "
                        f"ResearchItem ID: {detail.research_item_id or 'n/a'} | "
                        f"Event ID: {detail.event_id or 'n/a'}"
                    )
                    if detail.agent and detail.grounding_type == "link":
                        st.info(
                            "🔗 **Link-grounded** — relevant development identified from title/metadata only; "
                            "article content was unavailable. No claim below is based on reading the article body."
                        )
                        st.write(f"**Agent:** {detail.agent} — **Relevance:** {detail.relevance}")
                        st.write(f"Reason: {detail.relevance_reason}")
                        st.write(
                            f"is_dated_development: {detail.is_dated_development} — "
                            f"rationale: {detail.development_date_rationale or 'n/a'}"
                        )
                    elif detail.agent:
                        st.write(f"**Agent:** {detail.agent} — **Relevance:** {detail.relevance}")
                        st.write(f"Reason: {detail.relevance_reason}")
                        st.write(
                            f"is_dated_development: {detail.is_dated_development} — "
                            f"rationale: {detail.development_date_rationale or 'n/a'}"
                        )
                    elif detail.evidence_id is None:
                        reason = detail.content_unavailable_reason or "unknown"
                        st.warning(f"Discovered, but article content could not be retrieved ({reason}).")
                    elif chosen_row.analysis_status == "Rejected by deep analysis":
                        st.warning(
                            "An agent ran deep analysis on this candidate and judged it not relevant "
                            "(no deterministic relevance override applied). See this run's diagnostics "
                            "(Agent budget transparency / candidate-level log) for the exact reasoning."
                        )
                    elif chosen_row.analysis_status == "Rejected by triage":
                        st.warning(
                            "Triage judged this candidate not relevant before it reached deep analysis."
                        )
                    else:
                        st.info(
                            "Reason/stage unavailable: this article has Evidence but produced no ResearchItem, "
                            "and no triage/deep-analysis diagnostic marks it irrelevant either -- this run does "
                            "not persist a per-article record of why processing stopped here. Possible causes "
                            "include the CURRENT hard date gate (undated/stale/future), the LLM triage/deep-"
                            "analysis budget cap, or not being routed to any agent at all -- see the Pipeline "
                            "Funnel's aggregate counts for each cause above, which cannot currently be "
                            "attributed back to this specific article."
                        )

    st.subheader("Events")
    if not data.events:
        st.info("No events reconstructed for this run.")
    else:
        for event in data.events:
            with st.container(border=True):
                badge = "✅ Digest-qualified" if event.digest_qualified else f"⛔ {event.qualification_reason}"
                st.write(f"**{event.canonical_title}** — {badge}")
                st.caption(
                    f"Event ID: {event.event_id} | Type: {event.event_type or 'n/a'} | "
                    f"Company: {event.primary_company} | Related: {', '.join(event.related_companies) or 'none'}"
                )
                st.caption(
                    f"Event date: {event.event_date.isoformat() if event.event_date else 'None'} | "
                    f"is_dated_development: {event.is_dated_development} | "
                    f"Relevance: {event.relevance_score:.2f} | Confidence: {event.confidence_score:.2f}"
                )
                st.caption(
                    f"ResearchItems: {', '.join(event.research_item_ids) or 'none'} | "
                    f"Report: {event.report_title or 'n/a'}"
                )


def _render_digest_event(detail: "dd.EventDetail") -> None:
    """One digest entry: title/company/relevance, then whatever of
    summary/why-it-matters/source the existing Event+Report data already
    has -- never a new LLM call, never a fabricated field. A field with no
    existing data is simply omitted."""
    event = detail.event
    st.markdown(f"#### {event.canonical_title}")
    caption_bits = [event.primary_company]
    if event.event_type:
        caption_bits.append(event.event_type.value)
    if event.event_date:
        caption_bits.append(_local_date_isoformat(event.event_date))
    caption_bits.append(f"relevance {event.relevance_score:.2f}")
    caption_bits.append(f"confidence {event.confidence_score:.2f}")
    st.caption(" · ".join(caption_bits))

    report = detail.report
    if isinstance(report, Report):
        st.write(report.opening_context)
        st.markdown(f"**Why it matters:** {report.strategic_implications}")
    elif isinstance(report, ReportFailure):
        st.caption("No generated summary available for this event (report generation failed).")

    source = event.primary_source
    if source and source.url:
        label = source.publisher or event.primary_company
        st.markdown(f"**Source:** [{label}]({source.url})")


def _render_startups_page(service: PersistenceService) -> None:
    """Startups remains event-centric/discovery-driven — there is no fixed
    startup registry to enumerate as entity cards."""
    st.header("Startups")
    summaries = dd.search_events(service, agent=_STARTUP_AGENT, limit=100)
    st.dataframe(_events_table(summaries), use_container_width=True)
    event_id = _render_event_picker(service, summaries, key="startups_event_picker")
    if event_id:
        _render_event_detail(service, event_id)


_STATUS_ICON = {
    dd.STATUS_RECENT_DEVELOPMENTS: "🟢",
    dd.STATUS_NO_QUALIFYING_DEVELOPMENTS: "⚪",
    dd.STATUS_NO_ARTICLES_RETRIEVED: "⚪",
    dd.STATUS_RETRIEVED_NOT_RELEVANT: "⚪",
    dd.STATUS_RETRIEVAL_FAILED: "🔴",
    dd.STATUS_NOT_YET_ANALYZED: "⚪",
}


def _render_entity_card(
    status: "dd.EntityStatus", articles: Optional[List["dd.ArticleStatus"]] = None,
    *, history_since: Optional[datetime] = None, history_until: Optional[datetime] = None, config=None,
) -> None:
    with st.container(border=True):
        title_col, link_col = st.columns([4, 1])
        title_col.subheader(f"{_STATUS_ICON.get(status.status, '⚪')} {status.name}")
        if status.official_url:
            link_col.markdown(f"[Official source]({status.official_url})")
        else:
            link_col.caption("No official source configured")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Articles retrieved", status.articles_retrieved)
        metric_cols[1].metric("Relevant items", status.relevant_items)
        metric_cols[2].metric("Events", status.event_count)

        if status.status == dd.STATUS_RETRIEVAL_FAILED:
            st.error(f"Status: {status.status}\n\nReason: {status.source_failure_reason or 'unknown'}")
        elif status.status == dd.STATUS_RECENT_DEVELOPMENTS and status.latest_development is not None:
            dev = status.latest_development
            st.success(f"Status: {status.status}")
            st.write(f"**Latest:** {dev.canonical_title}")
            timestamp = status.latest_timestamp.isoformat() if status.latest_timestamp else "date unknown"
            st.caption(
                f"{timestamp} · relevance {dev.relevance_score:.2f} · confidence {dev.confidence_score:.2f}"
            )
            if status.latest_article_url:
                st.markdown(f"[Open article]({status.latest_article_url})")
        else:
            st.info(f"Status: {status.status}")

        # Human-in-the-loop (Phase 13D): the card above highlights one
        # "latest" development, but the analyst must be able to inspect
        # every article that was actually retrieved for this entity in the
        # window -- not just whichever one a summary happens to surface.
        if articles:
            # Phase 14: forensic-diagnosis Part 6/7 fix -- this is a ROLLING
            # HISTORICAL inventory (the sidebar's "Monitoring window (days)"
            # selector, measured from real wall-clock now), never the
            # current operational digest period. The old label ("...in this
            # window") could be read as meaning the digest's own window;
            # this spells out the actual date range so that confusion is
            # structurally impossible.
            range_text = (
                f" ({_fmt(history_since, config)} – {_fmt(history_until, config)})"
                if history_since and history_until and config else ""
            )
            with st.expander(f"All {len(articles)} article(s) published in this historical lookback{range_text}"):
                st.caption(
                    "This is a rolling historical article inventory, not the current Morning/Evening "
                    "digest's operational period -- see the Morning/Evening Digest pages for that."
                )
                for article in articles:
                    date_str = article.publication_date.date().isoformat() if article.publication_date else "date unknown"
                    bits = [date_str, article.status]
                    if article.event_relevance_score is not None:
                        bits.append(f"relevance {article.event_relevance_score:.2f}")
                    if article.event_confidence_score is not None:
                        bits.append(f"confidence {article.event_confidence_score:.2f}")
                    st.markdown(f"**[{article.title or article.url}]({article.url})**")
                    st.caption(" · ".join(bits))


def _render_entity_monitoring_page(service: PersistenceService, config, page: str) -> None:
    st.header(page)
    st.caption(
        "Every configured entity is always shown here, whether or not it has "
        "recent qualifying developments — this page never hides a monitored "
        "entity and never fabricates a development to fill it in."
    )

    days = st.sidebar.selectbox("Historical lookback (days)", [7, 30, 90], index=0, key=f"{page}_days")
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    st.caption(
        f"Articles published: {_fmt(since, config)} – {_fmt(now, config)} "
        "(a rolling historical lookback from now, independent of and NOT limited to the current "
        "Morning/Evening digest's operational period)."
    )

    entries = _ENTITY_PAGE_REGISTRY[page](config)
    statuses = dd.get_entity_statuses(service, entries, since=since, until=now)

    for entry, status in zip(entries, statuses):
        articles = dd.get_entity_articles(service, entry.name, since=since, until=now) if status.articles_retrieved else None
        _render_entity_card(status, articles, history_since=since, history_until=now, config=config)


def _render_cross_market_themes_page(service: PersistenceService) -> None:
    st.header("Cross-Market Themes")
    digests = dd.get_recent_digests(service, limit=10)
    if not digests:
        st.info("No digests have been generated yet.")
        return
    seen_labels = set()
    for digest in digests:
        for theme in digest.themes:
            if theme.label in seen_labels:
                continue
            seen_labels.add(theme.label)
            st.write(f"**{theme.label}** — {theme.description or ''} ({len(theme.event_ids)} event(s))")


def _credentials_are_valid(username: str, password: str, config, *, configured_password: Optional[str]) -> bool:
    """Pure comparison, never touching Streamlit/session state -- kept
    separate from `_render_login_gate` so it is unit-testable without the
    `AppTest` harness. `configured_password` is passed in rather than
    re-read here so a caller (or a test) controls exactly what "the
    configured secret" is for that check, mirroring how every mailer in
    this codebase takes its resolved secret as a parameter rather than
    reaching into the environment itself."""
    if not configured_password:
        return False
    return username == config.settings.auth.username and password == configured_password


def _render_login_gate(config) -> bool:
    """A single shared username/password gate for the entire app -- not a
    multi-user system (see `AuthSettings`'s own docstring). Returns True
    once `st.session_state["authenticated"]` is set, which `main()` uses
    to decide whether to render anything else at all. The password is
    never stored in `st.session_state`/anywhere else beyond this one
    comparison -- only the boolean "authenticated" result is kept."""
    if not config.settings.auth.enabled:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("NOMURA AI & FINTECH INTELLIGENCE")
    st.subheader("Log in")
    configured_password = load_app_login_password()
    if configured_password is None:
        st.error(
            "Login is not configured yet -- set APP_LOGIN_PASSWORD in this project's .env file, then "
            "restart the app."
        )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", disabled=configured_password is None)

    if submitted:
        if _credentials_are_valid(username, password, config, configured_password=configured_password):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid username or password.")
    return False


def main() -> None:
    config = get_config()
    if not _render_login_gate(config):
        return

    st.title("NOMURA AI & FINTECH INTELLIGENCE")
    if config.settings.auth.enabled and st.sidebar.button("Log out"):
        st.session_state.pop("authenticated", None)
        st.rerun()

    service = _get_persistence_service()

    default_index = _SIDEBAR_PAGES.index(_DEFAULT_LANDING_PAGE)
    page = st.sidebar.radio("Navigation", _SIDEBAR_PAGES, index=default_index, key="nav_page")

    if page == "CURRENT":
        _render_user_current_page(service, config)
    elif page == "Dashboard":
        _render_dashboard_page(service, config)
    elif page == "Morning Digest":
        _render_digest_page(service, DigestType.MORNING, config)
    elif page == "Evening Digest":
        _render_digest_page(service, DigestType.EVENING, config)
    elif page == "CURRENT / Live Scan (Admin)":
        _render_current_page(service, config)
    elif page in _ENTITY_PAGE_REGISTRY:
        _render_entity_monitoring_page(service, config, page)
    elif page == "Startups":
        _render_startups_page(service)
    elif page == "Cross-Market Themes":
        _render_cross_market_themes_page(service)


if __name__ == "__main__":
    main()
