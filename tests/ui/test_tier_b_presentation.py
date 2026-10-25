"""Tier B presentation improvements: triage confidence display, and
deterministic (no-LLM) dateline stripping / key-points extraction / prose
fallback. Every transformation either drops a recognizable, content-free
boilerplate fragment or reproduces existing source text verbatim --
never paraphrases, summarizes, or invents anything."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.config import get_config
from research_assistant.models.enums import BacklogStatus
from research_assistant.storage.repositories.backlog import BacklogRow
from research_assistant.ui import dashboard_data as dd

from tests.storage.conftest import make_article

D = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


def _seed(service, *, article_id="art1", clean_text="Clean article text.", triage_confidence=0.9):
    conn = service.database.connect()
    service.articles.upsert(conn, make_article(article_id, clean_text=clean_text))
    service.backlog.upsert(conn, BacklogRow(
        article_id=article_id, agent="startup", status=BacklogStatus.PENDING_DEEP_ANALYSIS_BUDGET.value,
        reason="test", attempt_count=0, defer_count=1, first_seen_run_id="run1", first_seen_at=D,
        last_attempt_run_id="run1", last_attempt_at=D, triage_relevant=True, triage_confidence=triage_confidence,
        research_item_id=None, updated_at=D,
    ))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Triage confidence display
# ---------------------------------------------------------------------------


def test_triage_confidence_formats_as_a_percentage():
    assert dd.format_triage_confidence(0.9) == "90%"
    assert dd.format_triage_confidence(0.0) == "0%"
    assert dd.format_triage_confidence(1.0) == "100%"


def test_missing_confidence_is_handled_safely():
    assert dd.format_triage_confidence(None) is None


def test_invalid_confidence_values_are_handled_safely():
    assert dd.format_triage_confidence("not a number") is None
    assert dd.format_triage_confidence(float("nan")) is None
    assert dd.format_triage_confidence(1.5) is None  # out of [0,1] range -- never persisted, but display never trusts blindly
    assert dd.format_triage_confidence(-0.1) is None


def test_tier_b_article_carries_the_real_triage_confidence(persistence_service):
    config = get_config()
    _seed(persistence_service, triage_confidence=0.85)

    articles, _ = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert articles[0].triage_confidence == 0.85


def test_tier_b_article_handles_missing_confidence(persistence_service):
    config = get_config()
    _seed(persistence_service, triage_confidence=None)

    articles, _ = dd.get_current_tier_b_pending(persistence_service, config, "run1")

    assert articles[0].triage_confidence is None


# ---------------------------------------------------------------------------
# Dateline stripping
# ---------------------------------------------------------------------------


def test_dateline_is_stripped_from_the_excerpt():
    text = "NOIDA, India, Oct. 12, 2026 /PRNewswire/ -- Flozic today announced the launch of its platform."
    excerpt, _ = dd._build_tier_b_presentation(text, limit=400)
    assert "PRNewswire" not in excerpt
    assert "NOIDA" not in excerpt
    assert "Flozic today announced the launch of its platform." in excerpt


def test_multi_city_dateline_is_stripped():
    text = "SINGAPORE and AMSTERDAM, Oct. 12, 2026 /PRNewswire/ -- SWI Group today announces an increase."
    excerpt, _ = dd._build_tier_b_presentation(text, limit=400)
    assert "SINGAPORE" not in excerpt
    assert "SWI Group today announces an increase." in excerpt


def test_text_without_a_dateline_is_left_untouched():
    text = "This article has no wire-service dateline at all, just plain prose about a product launch."
    excerpt, _ = dd._build_tier_b_presentation(text, limit=400)
    assert excerpt == text


def test_strip_dateline_refuses_when_nothing_substantive_would_remain():
    """Protection against removing substantive content: if stripping the
    matched dateline would leave next to nothing behind, the text is
    returned unchanged rather than risk eating real content."""
    text = "SHORT, Jan. 1, 2026 /Wire/ --"  # the "match" IS basically the whole string
    result = dd._strip_dateline(text)
    assert result == text  # refused -- stripping would leave < 10 real characters


# ---------------------------------------------------------------------------
# Key-points (bullet) extraction
# ---------------------------------------------------------------------------


def test_meaningful_bullets_are_extracted_verbatim():
    text = (
        "ASUS Unveils XA P2N-E2\n"
        "SAN JOSE, Calif., Oct. 12, 2026 /PRNewswire/ -- ASUS is expanding its AI infrastructure portfolio.\n"
        "- NVIDIA Vera Rubin NVL72 Platform: unifying 72 NVIDIA Rubin GPUs for frontier AI training.\n"
        "- NVIDIA Vera CPU Rack: Built for Agentic AI and Reinforcement Learning at Scale.\n"
        "- NVIDIA CMX Context Memory Storage Platform for agentic reasoning at scale.\n"
    )
    excerpt, key_points = dd._build_tier_b_presentation(text, limit=400)
    assert len(key_points) == 3
    assert key_points[0] == "NVIDIA Vera Rubin NVL72 Platform: unifying 72 NVIDIA Rubin GPUs for frontier AI training."
    assert "- NVIDIA Vera Rubin" not in excerpt  # not duplicated into the prose excerpt


def test_a_single_stray_dash_line_does_not_count_as_a_bullet_list():
    text = (
        "Some article text here with real substance about a company.\n"
        "- Just one line\n"
        "More article text continues here with additional substantive detail."
    )
    _excerpt, key_points = dd._build_tier_b_presentation(text, limit=400)
    assert key_points == []


def test_trivially_short_bullet_lines_are_not_counted():
    text = (
        "Real substantive prose paragraph describing a company development in detail.\n"
        "- Hi\n"
        "- Ok\n"
    )
    _excerpt, key_points = dd._build_tier_b_presentation(text, limit=400)
    assert key_points == []  # both lines under the minimum meaningful length


def test_no_bullets_falls_back_to_prose_excerpt():
    text = "A plain prose press release with no bullet-formatted lines anywhere in its body text at all."
    excerpt, key_points = dd._build_tier_b_presentation(text, limit=400)
    assert key_points == []
    assert excerpt == text


# ---------------------------------------------------------------------------
# Prose excerpt quality / truncation
# ---------------------------------------------------------------------------


def test_excerpt_prefers_a_complete_sentence_over_a_mid_sentence_cut():
    # 2026-10-16 production complaint: a mid-sentence word-boundary cut (e.g.
    # "...gained fresh momentum on Oct…") reads as a dangling, uninformative
    # fragment at a glance. One full 56-char sentence fits within limit=100;
    # a second would not, so the excerpt should stop there -- a complete
    # sentence, no ellipsis needed.
    text = "Real substantive sentence about a company development. " * 20
    excerpt, _ = dd._build_tier_b_presentation(text, limit=100)
    assert excerpt == "Real substantive sentence about a company development."
    assert not excerpt.endswith("…")


def test_excerpt_falls_back_to_word_boundary_when_no_sentence_end_is_near():
    # No period anywhere near the limit -- the sentence-boundary preference
    # must not stretch the excerpt arbitrarily far past `limit` looking for
    # one, so it falls back to the original word-boundary+ellipsis behavior.
    text = "Real substantive prose about a company development with no periods anywhere " * 5
    excerpt, _ = dd._build_tier_b_presentation(text, limit=100)
    assert len(excerpt) <= 101  # +1 for the truncation marker
    assert excerpt.endswith("…")
    assert not excerpt[:-1].endswith(" ")  # never truncated mid-word


def test_short_boilerplate_leading_lines_are_still_skipped():
    text = "Home\nMenu\nA full, substantive paragraph describing the real development in detail here."
    excerpt, _ = dd._build_tier_b_presentation(text, limit=400)
    assert "Home" not in excerpt
    assert "A full, substantive paragraph" in excerpt
