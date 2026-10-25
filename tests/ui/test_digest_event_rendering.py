"""Phase 13A: Morning/Evening Digest presentation (`_render_digest_event` in
`ui/app.py`) must show title/company/event-type/date/relevance/confidence,
a factual summary and "why it matters" sourced ONLY from the event's
already-generated Report (never a new LLM call, never fabricated text),
and a clickable source link -- gracefully omitting summary/why-it-matters
when no successful Report exists, rather than fabricating one.

Uses Streamlit's own `AppTest` harness against a tiny target script
(`tests/ui/fixtures/digest_event_render_target.py`) that calls
`_render_digest_event` directly with hand-built domain objects -- this
never touches `_get_persistence_service()`/`get_config()`/the real
`data/research.db`, so it cannot be affected by `st.cache_resource` or
whatever database the real app is currently configured to read.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

_TARGET = str(Path(__file__).parent / "fixtures" / "digest_event_render_target.py")


def _all_text(at) -> str:
    bits = []
    for el in at.markdown:
        bits.append(str(el.value))
    for el in at.caption:
        bits.append(str(el.value))
    return "\n".join(bits)


def test_digest_event_with_a_report_shows_summary_why_it_matters_and_source_link():
    at = AppTest.from_file(_TARGET, default_timeout=30)
    at.query_params["case"] = "with_report"
    at.run()

    assert at.exception == []
    text = _all_text(at)
    assert "Example Event Title" in text
    assert "REAL_OPENING_CONTEXT_MARKER" in text
    assert "Why it matters" in text
    assert "REAL_STRATEGIC_IMPLICATIONS_MARKER" in text
    assert "[Example Publisher](https://example.com/real-article)" in text
    # caption carries company/type/date/relevance/confidence
    assert "Example Co" in text
    assert "product_launch" in text
    assert "2026-09-28" in text
    assert "relevance 0.84" in text
    assert "confidence 0.62" in text


def test_digest_event_with_a_failed_report_omits_summary_but_does_not_crash():
    at = AppTest.from_file(_TARGET, default_timeout=30)
    at.query_params["case"] = "report_failure"
    at.run()

    assert at.exception == []
    text = _all_text(at)
    assert "REAL_OPENING_CONTEXT_MARKER" not in text  # never fabricated
    assert "report generation failed" in text.lower()
    # the source link is independent of report success and must still appear
    assert "https://example.com/real-article" in text


def test_digest_event_with_no_report_at_all_omits_summary_but_still_shows_the_source():
    at = AppTest.from_file(_TARGET, default_timeout=30)
    at.query_params["case"] = "no_report"
    at.run()

    assert at.exception == []
    text = _all_text(at)
    assert "REAL_OPENING_CONTEXT_MARKER" not in text
    assert "https://example.com/real-article" in text
