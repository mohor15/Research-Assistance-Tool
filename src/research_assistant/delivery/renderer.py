"""Email rendering for both automatic digest delivery and manual newsletter
sends (Section 7). Reuses exactly the same per-event data
`ui/dashboard_data.render_newsletter_markdown` already assembles (the
Event + its already-persisted Report) -- never an LLM call, never a new
summary, and never anything beyond the already-generated/selected content.
Produces an HTML body (escaped) with a parallel plain-text fallback.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from research_assistant.events.models import Event
from research_assistant.models.enums import GroundingType
from research_assistant.models.report import Report, ReportFailure
from research_assistant.storage.persistence_service import PersistenceService

from research_assistant.delivery.validation import is_safe_link

if TYPE_CHECKING:
    from research_assistant.config.schema import AppConfig

# Kept identical to `ui.dashboard_data._MAX_ENTITIES_SHOWN` so the email
# and Markdown newsletter paths present the same bounded entity list.
_MAX_ENTITIES_SHOWN = 6


@dataclass
class EmailContent:
    subject: str
    html_body: str
    text_body: str


def _esc(value: Optional[str]) -> str:
    return html.escape(value) if value else ""


def _load_event_and_report(service: PersistenceService, event_id: str) -> Tuple[Optional[Event], object]:
    conn = service.database.connect()
    try:
        event = service.events.get(conn, event_id)
        if event is None:
            return None, None
        report = service.reports.get(conn, event_id)
        return event, report
    finally:
        conn.close()


def _render_one_event(index: int, event: Event, report: object) -> Tuple[str, str]:
    html_parts = [f"<h3>{index}. {_esc(event.canonical_title)}</h3>"]
    text_parts = [f"{index}. {event.canonical_title}"]

    if event.grounding_type == GroundingType.LINK:
        notice = (
            "Link-grounded development — article content was unavailable; details are "
            "limited to title and publication metadata."
        )
        html_parts.append(f"<p><em>{_esc(notice)}</em></p>")
        text_parts.append(notice)

    if isinstance(report, Report):
        html_parts.append(f"<p><strong>What happened</strong><br>{_esc(report.opening_context)}</p>")
        text_parts.append(f"What happened: {report.opening_context}")

        if report.details.key_insights:
            points_html = "".join(f"<li>{_esc(point)}</li>" for point in report.details.key_insights)
            html_parts.append(f"<p><strong>Key insights</strong><ul>{points_html}</ul></p>")
            text_parts.append("Key insights:")
            for point in report.details.key_insights:
                text_parts.append(f"- {point}")

        html_parts.append(f"<p><strong>Strategic implications</strong><br>{_esc(report.strategic_implications)}</p>")
        text_parts.append(f"Strategic implications: {report.strategic_implications}")

        html_parts.append(f"<p><strong>Opportunity for Nomura</strong><br>{_esc(report.nomura_opportunity)}</p>")
        text_parts.append(f"Opportunity for Nomura: {report.nomura_opportunity}")

        # De-emphasized, bounded (see ui.dashboard_data._MAX_ENTITIES_SHOWN
        # for the matching Markdown behavior/rationale).
        entities = sorted(set(report.details.companies + report.details.partnerships))
        if entities:
            shown = entities[:_MAX_ENTITIES_SHOWN]
            remainder = len(entities) - len(shown)
            joined = ", ".join(shown) + (f" (+{remainder} more)" if remainder > 0 else "")
            html_parts.append(f"<p><strong>Also mentioned</strong><br>{_esc(joined)}</p>")
            text_parts.append(f"Also mentioned: {joined}")

        if report.sources:
            source = report.sources[0]
            publisher = _esc(source.publisher or "Original article")
            if is_safe_link(source.url):
                html_parts.append(f'<p><strong>Source</strong><br><a href="{_esc(source.url)}">{publisher}</a></p>')
                text_parts.append(f"Source: {source.publisher or 'Original article'} ({source.url})")
            else:
                html_parts.append(f"<p><strong>Source</strong><br>{publisher}</p>")
                text_parts.append(f"Source: {source.publisher or 'Original article'}")
    elif isinstance(report, ReportFailure):
        html_parts.append("<p><em>No generated summary is available for this event.</em></p>")
        text_parts.append("No generated summary is available for this event.")
        primary = event.primary_source
        if primary is not None and is_safe_link(primary.url):
            publisher = _esc(primary.publisher or "Original article")
            html_parts.append(f'<p><strong>Source</strong><br><a href="{_esc(primary.url)}">{publisher}</a></p>')
            text_parts.append(f"Source: {primary.publisher or 'Original article'} ({primary.url})")

    html_parts.append("<hr>")
    return "\n".join(html_parts), "\n".join(text_parts)


def _render_tier_b_report_html_and_text(report) -> Tuple[str, str]:
    """`report` is a `models.report.TierBReport` -- mirrors
    `ui.dashboard_data._render_tier_b_report_markdown`'s section shape."""
    html_parts = [f"<p><strong>What happened</strong><br>{_esc(report.opening_context)}</p>"]
    text_parts = [f"What happened: {report.opening_context}"]
    if report.key_insights:
        points_html = "".join(f"<li>{_esc(point)}</li>" for point in report.key_insights)
        html_parts.append(f"<p><strong>Key insights</strong><ul>{points_html}</ul></p>")
        text_parts.append("Key insights:")
        for point in report.key_insights:
            text_parts.append(f"- {point}")
    html_parts.append(f"<p><strong>Strategic implications</strong><br>{_esc(report.strategic_implications)}</p>")
    text_parts.append(f"Strategic implications: {report.strategic_implications}")
    html_parts.append(f"<p><strong>Opportunity for Nomura</strong><br>{_esc(report.nomura_opportunity)}</p>")
    text_parts.append(f"Opportunity for Nomura: {report.nomura_opportunity}")
    if report.content_partial:
        note = "Note: the retrieved source text for this article may be incomplete."
        html_parts.append(f"<p><em>{_esc(note)}</em></p>")
        text_parts.append(note)
    return "\n".join(html_parts), "\n".join(text_parts)


def _render_one_tier_b_article(index: int, article, resolution=None) -> Tuple[str, str]:
    from research_assistant.ui.dashboard_data import format_triage_confidence

    html_parts = [f"<h3>{index}. {_esc(article.title or article.url)}</h3>"]
    text_parts = [f"{index}. {article.title or article.url}"]
    report = resolution.report if resolution is not None else None
    notice = (
        "Relevant article -- on-demand analytical report generated below."
        if report is not None
        else "Relevant article awaiting specialist analysis -- no generated assessment below."
    )
    html_parts.append(f"<p><em>{_esc(notice)}</em></p>")
    text_parts.append(notice)
    if article.publisher:
        html_parts.append(
            f'<p><span style="background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;'
            f'border-radius:999px;padding:0.1rem 0.6rem;font-size:0.85em;">'
            f"Source: {_esc(article.publisher)}</span></p>"
        )
        text_parts.append(f"Source: {article.publisher}")
    confidence_label = format_triage_confidence(getattr(article, "triage_confidence", None))
    if confidence_label:
        html_parts.append(
            f'<p><span style="background:#eef2ff;color:#3730a3;border:1px solid #c7d2fe;'
            f'border-radius:999px;padding:0.1rem 0.6rem;font-size:0.85em;">'
            f"Triage confidence {_esc(confidence_label)}</span></p>"
        )
        text_parts.append(f"Triage confidence: {confidence_label}")

    if report is not None:
        html_piece, text_piece = _render_tier_b_report_html_and_text(report)
        html_parts.append(html_piece)
        text_parts.append(text_piece)
    else:
        if resolution is not None and resolution.status == "insufficient_content":
            note = "A full analytical report could not be generated: the available source content is insufficient."
            html_parts.append(f"<p><em>{_esc(note)}</em></p>")
            text_parts.append(note)
        elif resolution is not None and resolution.status == "failed":
            note = "A full analytical report could not be generated for this article this time."
            html_parts.append(f"<p><em>{_esc(note)}</em></p>")
            text_parts.append(note)
        if article.key_points:
            points_html = "".join(f"<li>{_esc(point)}</li>" for point in article.key_points)
            html_parts.append(
                f'<div style="background:#f0f9ff;border-left:4px solid #38bdf8;border-radius:6px;'
                f'padding:0.5rem 0.8rem;margin:0.4rem 0 0.6rem 0;">'
                f"<strong>Key points from the article</strong><ul>{points_html}</ul></div>"
            )
            text_parts.append("Key points from the article:")
            for point in article.key_points:
                text_parts.append(f"- {point}")
        html_parts.append(f"<p>{_esc(article.excerpt)}</p>")
        text_parts.append(article.excerpt)

    publisher = _esc(article.publisher or "Original article")
    if is_safe_link(article.url):
        html_parts.append(f'<p><strong>Source</strong><br><a href="{_esc(article.url)}">{publisher}</a></p>')
        text_parts.append(f"Source: {article.publisher or 'Original article'} ({article.url})")
    else:
        html_parts.append(f"<p><strong>Source</strong><br>{publisher}</p>")
        text_parts.append(f"Source: {article.publisher or 'Original article'}")
    html_parts.append("<hr>")
    return "\n".join(html_parts), "\n".join(text_parts)


def render_event_list_email(
    service: PersistenceService, event_ids: List[str], *, subject: str, intro: str = "",
    config: Optional["AppConfig"] = None, tier_b_reports: Optional[dict] = None,
) -> EmailContent:
    """`event_ids` is already the FINAL, caller-decided ordered list -- a
    Digest's own ranked `events`, or a user's newsletter selection (which
    may include Tier B article selections -- see `ui.dashboard_data.
    TIER_B_SELECTION_PREFIX`) -- this function never queries for or
    re-selects events itself, and never fabricates content for an id that
    fails to resolve (it is simply skipped, exactly like
    `render_newsletter_markdown`). `config` is only required when
    `event_ids` contains a Tier B selection. `tier_b_reports`, if given,
    is an ALREADY-RESOLVED `{article_id: TierBReportResolution}` dict from
    `reporting.tier_b_newsletter_resolution.resolve_tier_b_reports` --
    this function itself NEVER calls an LLM or the network, exactly like
    before; it only ever reads what the caller already resolved."""
    from research_assistant.ui.dashboard_data import TIER_B_SELECTION_PREFIX, get_tier_b_article_detail

    html_sections: List[str] = []
    text_sections: List[str] = [intro] if intro else []
    rendered_any = False

    for index, selection_id in enumerate(event_ids, start=1):
        if selection_id.startswith(TIER_B_SELECTION_PREFIX):
            if config is None:
                continue
            article_id = selection_id[len(TIER_B_SELECTION_PREFIX):]
            article = get_tier_b_article_detail(service, config, article_id)
            if article is None:
                continue
            rendered_any = True
            resolution = tier_b_reports.get(article_id) if tier_b_reports else None
            html_piece, text_piece = _render_one_tier_b_article(index, article, resolution)
            html_sections.append(html_piece)
            text_sections.append(text_piece)
            continue

        event, report = _load_event_and_report(service, selection_id)
        if event is None:
            continue
        rendered_any = True
        html_piece, text_piece = _render_one_event(index, event, report)
        html_sections.append(html_piece)
        text_sections.append(text_piece)

    if not rendered_any:
        body_html = "<p>No developments to report.</p>"
        text_sections.append("No developments to report.")
    else:
        body_html = "".join(html_sections)

    html_body = (
        '<html><body style="font-family: Arial, sans-serif; color: #1a1a1a;">'
        f"<h2>{_esc(subject)}</h2>"
        + (f"<p>{_esc(intro)}</p>" if intro else "")
        + body_html
        + "</body></html>"
    )
    text_body = "\n\n".join(text_sections)
    return EmailContent(subject=subject, html_body=html_body, text_body=text_body)
