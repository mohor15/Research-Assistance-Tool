"""Newsletter PDF download (Section 3): a pure, in-memory transform of the
ALREADY-GENERATED newsletter Markdown (`ui.dashboard_data.
render_newsletter_markdown`) into a PDF -- never a second, independent
data-gathering pass over events/reports, and never an LLM call. This keeps
"Preserve Markdown generation" literal: the Markdown string remains the
single source of truth for newsletter content; this module only renders
that exact same text differently.

No side effects at all: `render_newsletter_pdf` reads nothing from the
database and writes nothing to it. A failure here can therefore never lose
a selection or alter a persisted report -- there is nothing for it to
alter.

Supports the limited Markdown subset `render_newsletter_markdown`/
`_render_tier_b_markdown` actually produce: `#`/`##`/`###` headers,
`**bold**`, `_italic_`, `[text](url)` links, `- ` bullets, and `> `
blockquotes (used for Tier B's "Key points from the article" box) --
never a general-purpose Markdown parser.
"""

from __future__ import annotations

import html
import io
import re
from typing import List

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\w)_(.+?)_(?!\w)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_LINK_COLOR_HEX = "#1a5fb4"


def _inline_to_reportlab_markup(text: str) -> str:
    """Escapes raw text first (so stray `<`/`&` in article titles can never
    be interpreted as markup), then converts the small inline-Markdown
    subset above into ReportLab's own mini-markup tags."""
    escaped = html.escape(text, quote=False)
    escaped = _BOLD_RE.sub(r"<b>\1</b>", escaped)
    escaped = _ITALIC_RE.sub(r"<i>\1</i>", escaped)
    escaped = _LINK_RE.sub(rf'<link href="\2"><u><font color="{_LINK_COLOR_HEX}">\1</font></u></link>', escaped)
    return escaped


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="NewsletterTitle", parent=styles["Title"], spaceAfter=12))
    styles.add(ParagraphStyle(name="NewsletterH2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="NewsletterH3", parent=styles["Heading3"], spaceBefore=12, spaceAfter=4))
    styles.add(ParagraphStyle(name="NewsletterBody", parent=styles["BodyText"], spaceAfter=4, leading=14))
    styles.add(ParagraphStyle(name="NewsletterBullet", parent=styles["BodyText"], leftIndent=14, spaceAfter=2, leading=14))
    styles.add(ParagraphStyle(name="NewsletterQuote", parent=styles["BodyText"], leftIndent=14, textColor=HexColor("#444444"), spaceAfter=2, leading=14))
    return styles


def render_newsletter_pdf(markdown_text: str) -> bytes:
    """Converts `markdown_text` (as produced by `render_newsletter_markdown`)
    into a PDF, returned as raw bytes ready for `st.download_button`. Pure
    function, no I/O beyond the in-memory buffer -- safe to call on every
    Streamlit rerun."""
    styles = _build_styles()
    story: List = []
    paragraph_buffer: List[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            story.append(Paragraph("<br/>".join(paragraph_buffer), styles["NewsletterBody"]))
            story.append(Spacer(1, 4))
            paragraph_buffer.clear()

    for raw_line in markdown_text.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(_inline_to_reportlab_markup(stripped[4:]), styles["NewsletterH3"]))
        elif stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(_inline_to_reportlab_markup(stripped[3:]), styles["NewsletterH2"]))
        elif stripped.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(_inline_to_reportlab_markup(stripped[2:]), styles["NewsletterTitle"]))
        elif stripped.startswith("> "):
            inner = stripped[2:]
            flush_paragraph()
            if inner.startswith("- "):
                story.append(Paragraph("&bull;&nbsp;" + _inline_to_reportlab_markup(inner[2:]), styles["NewsletterQuote"]))
            else:
                story.append(Paragraph(_inline_to_reportlab_markup(inner), styles["NewsletterQuote"]))
        elif stripped.startswith("- "):
            flush_paragraph()
            story.append(Paragraph("&bull;&nbsp;" + _inline_to_reportlab_markup(stripped[2:]), styles["NewsletterBullet"]))
        else:
            paragraph_buffer.append(_inline_to_reportlab_markup(stripped))
    flush_paragraph()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter, topMargin=54, bottomMargin=54, leftMargin=54, rightMargin=54,
        title="Nomura AI & Fintech Intelligence Newsletter",
    )
    doc.build(story)
    return buffer.getvalue()
