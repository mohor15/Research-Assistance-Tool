"""Targeted fix: `TriageItemResult.confidence` normalization.

Confirmed real production failure (2026-10-07 CURRENT run
`2a55bf02ff1b7d1f`): qwen2.5:3b sometimes returns `confidence` as a 0-100
integer (90, 95, 75, 25, 5) instead of the documented 0.0-1.0 float, which
previously failed the field's `ge=0.0, le=1.0` constraint and discarded the
ENTIRE triage batch (13/24 real candidates lost in that one run). This only
tests the schema-level normalization; no triage batching/routing/date logic
is touched."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_assistant.agents.triage import TriageItemResult


def _result(confidence) -> TriageItemResult:
    return TriageItemResult(candidate_id="c1", relevant=True, confidence=confidence)


# ---------------------------------------------------------------------------
# Values already in [0.0, 1.0] pass through unchanged
# ---------------------------------------------------------------------------


def test_confidence_0_0_is_unchanged():
    assert _result(0.0).confidence == 0.0


def test_confidence_0_5_is_unchanged():
    assert _result(0.5).confidence == 0.5


def test_confidence_1_0_is_unchanged():
    assert _result(1.0).confidence == 1.0


# ---------------------------------------------------------------------------
# A 0-100 percentage scale is normalized by dividing by 100
# ---------------------------------------------------------------------------


def test_confidence_90_is_normalized_to_0_90():
    assert _result(90).confidence == pytest.approx(0.90)


def test_confidence_95_is_normalized_to_0_95():
    assert _result(95).confidence == pytest.approx(0.95)


def test_confidence_75_is_normalized_to_0_75():
    assert _result(75).confidence == pytest.approx(0.75)


def test_confidence_25_is_normalized_to_0_25():
    assert _result(25).confidence == pytest.approx(0.25)


def test_confidence_5_is_normalized_to_0_05():
    assert _result(5).confidence == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Genuinely invalid values are still rejected, not silently accepted
# ---------------------------------------------------------------------------


def test_confidence_negative_value_is_rejected():
    with pytest.raises(ValidationError):
        _result(-5)


def test_confidence_above_100_is_rejected():
    with pytest.raises(ValidationError):
        _result(150)


# ---------------------------------------------------------------------------
# Existing triage behavior (construction without a `reason` field, required
# fields present) remains unchanged
# ---------------------------------------------------------------------------


def test_existing_triage_item_construction_still_works_unchanged():
    result = TriageItemResult(candidate_id="c1", relevant=True, confidence=0.9)
    assert result.relevant is True
    assert result.confidence == 0.9
    assert "reason" not in TriageItemResult.model_fields
