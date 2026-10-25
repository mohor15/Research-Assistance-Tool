"""Email rendering: reuses the same persisted Event/Report data the
Markdown newsletter renderer uses, but must escape untrusted content and
only ever render a clickable link for a safe absolute http(s) URL."""

from __future__ import annotations

from research_assistant.delivery.renderer import render_event_list_email
from research_assistant.models.report import Report, ReportDetails, ReportSource
from research_assistant.models.enums import SourcePriority, SourceType

from tests.storage.conftest import make_article, make_event, make_report, make_report_failure, make_research_item


def _persist_basic_event(persistence_service, *, event_id: str, url: str, report=None, report_failure=None):
    article = make_article(f"art-{event_id}", url=url)
    item = make_research_item(f"ri-{event_id}", article_id=article.id)
    event = make_event(event_id, research_item_ids=[item.id], article_ids=[article.id])
    persistence_service.persist_pipeline_result(
        articles=[article], research_items=[item], events=[event],
        reports=[report] if report is not None else (),
        report_failures=[report_failure] if report_failure is not None else (),
    )
    return event


def test_render_includes_event_title_and_report_sections(persistence_service):
    _persist_basic_event(persistence_service, event_id="ev1", url="https://example.com/ev1", report=make_report("ev1"))

    content = render_event_list_email(persistence_service, ["ev1"], subject="Test Digest")

    assert "An event title" in content.html_body
    assert "An event title" in content.text_body
    assert "What happened" in content.html_body
    assert "Strategic implications" in content.html_body
    assert "Opportunity for Nomura" in content.html_body


def test_render_escapes_untrusted_html_in_report_content(persistence_service):
    malicious_report = Report(
        event_id="ev2", title="<img src=x onerror=alert(1)>",
        opening_context="<script>alert('xss')</script> breaking news",
        details=ReportDetails(companies=["<b>Evil Corp</b>"], narrative="n/a"),
        strategic_implications="Also <script>evil()</script> here.",
        nomura_opportunity="Fine.",
        sources=[ReportSource(publisher="Pub", url="https://example.com/ev2", source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH)],
        relevance_score=0.5, confidence_score=0.5,
    )
    _persist_basic_event(persistence_service, event_id="ev2", url="https://example.com/ev2", report=malicious_report)

    content = render_event_list_email(persistence_service, ["ev2"], subject="Test Digest")

    assert "<script>" not in content.html_body
    assert "<img" not in content.html_body
    assert "&lt;script&gt;" in content.html_body


def test_render_makes_a_safe_source_link_clickable(persistence_service):
    _persist_basic_event(persistence_service, event_id="ev3", url="https://example.com/ev3", report=make_report("ev3"))

    content = render_event_list_email(persistence_service, ["ev3"], subject="Test Digest")

    # `make_report`'s source URL is always https://example.com/art1, regardless
    # of the event id -- see tests/storage/conftest.py.
    assert '<a href="https://example.com/art1">' in content.html_body


def test_render_never_renders_an_unsafe_source_link(persistence_service):
    unsafe_report = Report(
        event_id="ev5", title="t", opening_context="o",
        details=ReportDetails(narrative="n"), strategic_implications="s", nomura_opportunity="n",
        sources=[ReportSource(publisher="Pub", url="javascript:alert(1)", source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH)],
        relevance_score=0.5, confidence_score=0.5,
    )
    _persist_basic_event(persistence_service, event_id="ev5", url="https://example.com/ev5", report=unsafe_report)

    content = render_event_list_email(persistence_service, ["ev5"], subject="Test Digest")

    assert "<a href=" not in content.html_body


def test_render_handles_report_failure_without_raising(persistence_service):
    _persist_basic_event(
        persistence_service, event_id="ev4", url="https://example.com/ev4", report_failure=make_report_failure("ev4"),
    )

    content = render_event_list_email(persistence_service, ["ev4"], subject="Test Digest")

    assert "No generated summary is available" in content.html_body


def test_render_skips_missing_event_ids_without_raising(persistence_service):
    content = render_event_list_email(persistence_service, ["does-not-exist"], subject="Test Digest")
    assert "No developments to report" in content.html_body


def test_render_empty_event_list_produces_a_safe_fallback_body(persistence_service):
    content = render_event_list_email(persistence_service, [], subject="Test Digest")
    assert "No developments to report" in content.html_body


def test_render_always_includes_both_html_and_text_bodies(persistence_service):
    _persist_basic_event(persistence_service, event_id="ev6", url="https://example.com/ev6", report=make_report("ev6"))

    content = render_event_list_email(persistence_service, ["ev6"], subject="My Subject")

    assert content.subject == "My Subject"
    assert "<html>" in content.html_body
    assert "<html>" not in content.text_body
