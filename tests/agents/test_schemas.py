"""AgentAssessment / ResearchItem schema validation: valid construction, missing
required fields, invalid enum values, and that unrequested extra fields from
the LLM don't silently attach new unvalidated data."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_assistant.agents.schemas import AgentAssessment
from research_assistant.models.enums import EntityType
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
    TypedEntity,
)


def _valid_assessment_kwargs(**overrides):
    kwargs = dict(
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="Concrete GenAI product launch."),
        category=TopicCategory.ENTERPRISE_AI,
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="Directly applies to FS."),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="Signals adoption trend."),
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_assessment_parses_with_minimal_fields():
    assessment = AgentAssessment(**_valid_assessment_kwargs())
    assert assessment.relevance.label == RelevanceLabel.RELEVANT
    assert assessment.entities == []
    assert assessment.deal_value is None


def test_valid_assessment_parses_with_full_extraction_fields():
    assessment = AgentAssessment(
        **_valid_assessment_kwargs(
            entities=[TypedEntity(name="JPMorgan Chase", type=EntityType.COMPANY)],
            people=["Jane Doe"], products=["AI Copilot"],
            deal_value="$50 million", partnerships=["Acme AI"], technologies=["LLM"],
            geography=["United States"], summary="A summary grounded in the evidence.",
        )
    )
    assert assessment.deal_value == "$50 million"
    assert assessment.people == ["Jane Doe"]


def test_missing_relevance_field_is_rejected():
    kwargs = _valid_assessment_kwargs()
    del kwargs["relevance"]
    with pytest.raises(ValidationError):
        AgentAssessment(**kwargs)


def test_missing_required_relevance_assessment_is_rejected():
    kwargs = _valid_assessment_kwargs()
    del kwargs["financial_services_relevance"]
    with pytest.raises(ValidationError):
        AgentAssessment(**kwargs)


def test_invalid_relevance_label_is_rejected():
    with pytest.raises(ValidationError):
        AgentAssessment(**_valid_assessment_kwargs(relevance={"label": "definitely-maybe", "reason": "x"}))


def test_invalid_category_value_is_rejected():
    kwargs = _valid_assessment_kwargs()
    kwargs["category"] = "not-a-real-category"
    with pytest.raises(ValidationError):
        AgentAssessment(**kwargs)


def test_empty_relevance_reason_is_rejected():
    with pytest.raises(ValidationError):
        AgentAssessment(**_valid_assessment_kwargs(relevance={"label": "relevant", "reason": ""}))


def test_from_raw_llm_json_missing_a_required_field_raises_clear_error():
    """Simulates what OllamaClient.generate_structured sees: raw dict from the model."""
    raw = {
        "relevance": {"label": "relevant", "reason": "ok"},
        "category": "enterprise_ai",
        # financial_services_relevance intentionally omitted, as an LLM might do
        "strategic_significance": {"level": "medium", "rationale": "x"},
    }
    with pytest.raises(ValidationError):
        AgentAssessment.model_validate(raw)


def test_unrequested_extra_fields_from_llm_output_do_not_attach_new_data():
    raw = _valid_assessment_kwargs()
    raw = {
        "relevance": raw["relevance"].model_dump(),
        "category": raw["category"].value,
        "is_dated_development": raw["is_dated_development"],
        "financial_services_relevance": raw["financial_services_relevance"].model_dump(),
        "strategic_significance": raw["strategic_significance"].model_dump(),
        "confidence_score": 0.97,  # a hallucinated/unrequested field a model might add
        "made_up_field": "should not appear anywhere",
    }
    assessment = AgentAssessment.model_validate(raw)
    assert not hasattr(assessment, "confidence_score")
    assert not hasattr(assessment, "made_up_field")
    assert "confidence_score" not in assessment.model_dump()
