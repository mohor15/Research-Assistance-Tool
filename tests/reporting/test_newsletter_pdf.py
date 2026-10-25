"""`render_newsletter_pdf`: a pure, in-memory Markdown->PDF transform of
the already-generated newsletter text -- never touches the database, never
calls an LLM, never triggers a scan. Tests only check structural/byte-level
properties (valid PDF header, non-trivial size, no exception on the real
Markdown shapes `ui.dashboard_data.render_newsletter_markdown`/
`_render_tier_b_markdown` actually produce) -- never parse the PDF's visual
layout."""

from __future__ import annotations

from research_assistant.reporting.newsletter_pdf import render_newsletter_pdf


def test_produces_a_valid_pdf_byte_stream():
    pdf_bytes = render_newsletter_pdf("# NOMURA AI & FINTECH INTELLIGENCE\n\nSome body text.\n")
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 200


def test_handles_the_real_tier_a_markdown_shape():
    markdown = (
        "# NOMURA AI & FINTECH INTELLIGENCE\n\n"
        "## Selected Intelligence Newsletter\n\n"
        "### 1. JPMorgan launches AI copilot\n\n"
        "**What happened**  \nJPMorgan Chase announced a new generative AI copilot for advisors.\n\n"
        "**Key insights**\n"
        "- Expands JPMorgan's AI product suite.\n"
        "- Targets financial advisors specifically.\n\n"
        "**Strategic implications**  \nThis may intensify competition among peer banks.\n\n"
        "**Opportunity for Nomura**  \nWorth monitoring for competitive response.\n\n"
        "**Also mentioned**  \nOpenAI, Anthropic\n\n"
        "**Source**  \n[JPMorgan Chase](https://jpmorganchase.com/a)\n"
    )
    pdf_bytes = render_newsletter_pdf(markdown)
    assert pdf_bytes.startswith(b"%PDF-")


def test_handles_the_real_tier_b_markdown_shape_with_blockquote_key_points():
    markdown = (
        "### 2. ASUS Unveils XA P2N-E2\n\n"
        "_Relevant article awaiting specialist analysis -- no generated assessment below._\n\n"
        "PR Newswire · 2026-10-12 · Triage confidence: 90%\n\n"
        "> **Key points from the article**\n"
        "> - NVIDIA Vera Rubin NVL72 Platform: unifying 72 NVIDIA Rubin GPUs.\n"
        "> - NVIDIA Vera CPU Rack: Built for Agentic AI.\n\n"
        "ASUS is expanding its AI infrastructure portfolio.\n\n"
        "**Source**  \n[PR Newswire](https://www.prnewswire.com/news-releases/asus-a)\n"
    )
    pdf_bytes = render_newsletter_pdf(markdown)
    assert pdf_bytes.startswith(b"%PDF-")


def test_handles_a_link_grounded_event_notice():
    markdown = (
        "### 1. Some Development\n\n"
        "_Link-grounded development — article content was unavailable; details are limited to "
        "title and publication metadata._\n\n"
        "_No generated summary is available for this event._\n"
        "**Source**  \n[Original article](https://example.com/a)\n"
    )
    pdf_bytes = render_newsletter_pdf(markdown)
    assert pdf_bytes.startswith(b"%PDF-")


def test_empty_markdown_does_not_raise():
    pdf_bytes = render_newsletter_pdf("")
    assert pdf_bytes.startswith(b"%PDF-")


def test_special_characters_are_escaped_not_interpreted_as_markup():
    """A title containing a literal `<`/`&` (e.g. "R&D < 5%") must never be
    interpreted as ReportLab markup -- it must render as plain escaped
    text, and critically must never raise inside `doc.build`."""
    markdown = "### 1. Company's R&D spend < 5% this year\n\nSome body text with & an ampersand.\n"
    pdf_bytes = render_newsletter_pdf(markdown)
    assert pdf_bytes.startswith(b"%PDF-")


def test_is_pure_with_no_side_effects_across_repeated_calls():
    markdown = "# Title\n\nSome **bold** and _italic_ text with a [link](https://example.com).\n"
    first = render_newsletter_pdf(markdown)
    second = render_newsletter_pdf(markdown)
    assert first.startswith(b"%PDF-") and second.startswith(b"%PDF-")
