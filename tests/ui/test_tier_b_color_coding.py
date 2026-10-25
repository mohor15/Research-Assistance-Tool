"""Tier B color-coding: the triage-confidence badge, the source badge,
and the key-points box each get their own distinct CSS treatment."""

from __future__ import annotations

from research_assistant.ui.app import _CURRENT_PAGE_CSS, _badge


def test_source_badge_uses_its_own_css_variant():
    markup = _badge("Source: PR Newswire", variant="source")
    assert 'class="nomura-badge nomura-badge-source"' in markup
    assert "Source: PR Newswire" in markup


def test_confidence_badge_uses_the_score_variant():
    markup = _badge("Triage confidence 90%", variant="score")
    assert 'class="nomura-badge nomura-badge-score"' in markup


def test_status_badge_uses_the_link_variant():
    markup = _badge("Awaiting specialist analysis", variant="link")
    assert 'class="nomura-badge nomura-badge-link"' in markup


def test_source_badge_css_is_visually_distinct_from_score_and_status():
    assert ".nomura-badge-source" in _CURRENT_PAGE_CSS
    assert ".nomura-badge-score" in _CURRENT_PAGE_CSS
    assert ".nomura-badge-link" in _CURRENT_PAGE_CSS
    # Three distinct color declarations, not the same color reused.
    source_color = _CURRENT_PAGE_CSS.split(".nomura-badge-source")[1].split("}")[0]
    score_color = _CURRENT_PAGE_CSS.split(".nomura-badge-score")[1].split("}")[0]
    link_color = _CURRENT_PAGE_CSS.split(".nomura-badge-link")[1].split("}")[0]
    assert len({source_color, score_color, link_color}) == 3


def test_key_points_box_has_its_own_distinct_styling():
    assert ".nomura-key-points-box" in _CURRENT_PAGE_CSS
    box_style = _CURRENT_PAGE_CSS.split(".nomura-key-points-box {")[1].split("}")[0]
    assert "background" in box_style
