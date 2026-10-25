"""AnalystAgent: exactly the 5 configured firms, thesis/evidence-point/implication
extraction, reported-numeric preservation, anti-hallucination prompt guarantees,
malformed output, provenance, mocked Ollama failure."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_assistant.agents.analyst import AnalystAgent
from research_assistant.agents.schemas import AnalystAssessment
from research_assistant.llm.errors import LLMResponseError, LLMUnavailableError
from research_assistant.models.research_item import (
    AnalystFindings,
    ReportedFinding,
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
)

from tests.agents.conftest import FakeLLMClient, analyst_source_for, make_article_and_evidence


def _analyst_assessment(label: RelevanceLabel, reason: str, *, findings: AnalystFindings, **overrides) -> AnalystAssessment:
    kwargs = dict(
        relevance=RelevanceDecision(label=label, reason=reason),
        category=TopicCategory.FINANCIAL_SERVICES,
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="x"),
        analyst_findings=findings,
    )
    kwargs.update(overrides)
    return AnalystAssessment(**kwargs)


@pytest.mark.parametrize(
    "firm_name,text",
    [
        (
            "McKinsey",
            "McKinsey's QuantumBlack practice published new research today on "
            "agentic AI adoption patterns among global banks, finding that "
            "enterprise AI deployment is accelerating across capital markets functions.",
        ),
        (
            "BCG",
            "BCG released a new report on AI economics and AI productivity gains "
            "in financial services, highlighting how banks are adopting "
            "generative AI for research workflows.",
        ),
        (
            "Deloitte",
            "Deloitte's AI Institute published a new study on AI governance and "
            "the future of work, with findings on how enterprise AI "
            "transformation is reshaping banking operations.",
        ),
        (
            "KPMG",
            "KPMG published new AI insights on AI investment trends and "
            "agentic AI use cases in capital markets and banking.",
        ),
        (
            "PwC",
            "PwC's latest research covers AI adoption in banking, examining "
            "enterprise AI operating models and their impact on financial "
            "services transformation.",
        ),
    ],
)
def test_relevant_article_from_each_configured_firm_produces_research_item(config, firm_name, text):
    source = analyst_source_for(config, firm_name)
    article, evidence = make_article_and_evidence(
        source, url=f"https://example.com/{firm_name.lower()}/insight", title=f"{firm_name} AI research", text=text,
    )
    findings = AnalystFindings(thesis=f"{firm_name}'s thesis grounded in the evidence.")
    assessment = _analyst_assessment(RelevanceLabel.RELEVANT, "Substantive AI-strategy research.", findings=findings)
    agent = AnalystAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item is not None
    assert item.company == firm_name
    assert item.analyst_findings is not None
    assert item.analyst_findings.thesis == f"{firm_name}'s thesis grounded in the evidence."


def test_irrelevant_consulting_article_is_rejected(config):
    source = analyst_source_for(config, "McKinsey")
    article, evidence = make_article_and_evidence(
        source, url="https://example.com/mckinsey/event",
        title="McKinsey hosts AI adoption panel",
        text=(
            "McKinsey announced it will host a panel discussion on AI adoption "
            "trends at an upcoming industry conference, with no report, finding, "
            "or thesis published alongside the announcement."
        ),
    )
    assessment = _analyst_assessment(
        RelevanceLabel.IRRELEVANT, "Event announcement only, no research finding.",
        findings=AnalystFindings(),
    )
    llm = FakeLLMClient([assessment])
    agent = AnalystAgent(llm, config)

    item = agent.analyze(article, evidence)

    assert item is None
    assert len(llm.calls) == 1


def test_thesis_extraction_is_preserved_on_research_item(config):
    source = analyst_source_for(config, "BCG")
    article, evidence = make_article_and_evidence(
        source, url="https://example.com/bcg/thesis",
        title="BCG on agentic AI adoption",
        text="BCG's new report argues that agentic AI adoption in banking will define enterprise AI competitiveness through 2027.",
    )
    findings = AnalystFindings(thesis="Agentic AI adoption will define enterprise AI competitiveness in banking through 2027.")
    assessment = _analyst_assessment(RelevanceLabel.RELEVANT, "Clear thesis stated.", findings=findings)
    agent = AnalystAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item.analyst_findings.thesis.startswith("Agentic AI adoption")


def test_evidence_point_extraction_marks_quantitative_claims(config):
    source = analyst_source_for(config, "KPMG")
    article, evidence = make_article_and_evidence(
        source, url="https://example.com/kpmg/survey",
        title="KPMG AI adoption survey",
        text=(
            "KPMG's survey of 400 bank executives found that 62% have already "
            "deployed generative AI in at least one enterprise AI workflow, "
            "and that banking leaders view AI investment as a top priority."
        ),
    )
    findings = AnalystFindings(
        evidence_points=[
            ReportedFinding(statement="62% of surveyed bank executives have deployed generative AI in at least one workflow.", is_quantitative=True),
            ReportedFinding(statement="Banking leaders view AI investment as a top priority.", is_quantitative=False),
        ],
    )
    assessment = _analyst_assessment(RelevanceLabel.RELEVANT, "Survey-backed finding.", findings=findings)
    agent = AnalystAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    quantitative = [p for p in item.analyst_findings.evidence_points if p.is_quantitative]
    non_quantitative = [p for p in item.analyst_findings.evidence_points if not p.is_quantitative]
    assert len(quantitative) == 1
    assert "62%" in quantitative[0].statement
    assert len(non_quantitative) == 1


def test_financial_implication_is_preserved(config):
    """`financial_implication` covers both FS and capital-markets relevance
    (merged from the former two-field split -- CPU-only qwen2.5:3b
    deep-analysis cost reduction: neither sub-field had an independent
    downstream consumer, so asking for two free-text paragraphs instead of
    one was pure extra generation cost)."""
    source = analyst_source_for(config, "Deloitte")
    article, evidence = make_article_and_evidence(
        source, url="https://example.com/deloitte/fs",
        title="Deloitte on banking AI",
        text="Deloitte's AI Institute finds that enterprise AI transformation in banking is reshaping how financial services firms allocate technology budgets.",
    )
    findings = AnalystFindings(
        financial_implication="Financial services firms are reallocating technology budgets toward enterprise AI transformation.",
    )
    assessment = _analyst_assessment(RelevanceLabel.RELEVANT, "Direct FS implication stated.", findings=findings)
    agent = AnalystAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert "technology budgets" in item.analyst_findings.financial_implication


def test_reported_numerical_finding_is_preserved_verbatim(config):
    source = analyst_source_for(config, "PwC")
    article, evidence = make_article_and_evidence(
        source, url="https://example.com/pwc/market-size",
        title="PwC AI market forecast",
        text="PwC's report forecasts the enterprise AI market in financial services will reach $45 billion by 2028, according to its own modeling.",
    )
    findings = AnalystFindings(
        evidence_points=[
            ReportedFinding(statement="PwC forecasts the enterprise AI market in financial services will reach $45 billion by 2028.", is_quantitative=True),
        ],
    )
    assessment = _analyst_assessment(RelevanceLabel.RELEVANT, "Forecast reported by the evidence.", findings=findings)
    agent = AnalystAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    point = item.analyst_findings.evidence_points[0]
    assert point.is_quantitative is True
    assert "$45 billion" in point.statement


def test_system_prompt_forbids_inventing_historical_context(config):
    agent = AnalystAgent(FakeLLMClient([]), config)
    prompt_text = " ".join(agent._system_prompt.lower().split())
    assert "never invent a timeline, prior event, or" in prompt_text
    assert "historical fact from general knowledge" in prompt_text


def test_system_prompt_forbids_hallucinated_quantitative_claims(config):
    agent = AnalystAgent(FakeLLMClient([]), config)
    prompt_text = " ".join(agent._system_prompt.lower().split())
    assert "never invent a quantitative finding" in prompt_text
    assert "do not adjust, round, or reinterpret" in prompt_text


def test_malformed_llm_output_raises_llm_response_error(config):
    source = analyst_source_for(config, "BCG")
    article, evidence = make_article_and_evidence(
        source, url="https://example.com/bcg/x", title="x",
        text="BCG's new report on AI economics and enterprise AI adoption in banking.",
    )
    agent = AnalystAgent(FakeLLMClient([LLMResponseError("malformed model output")]), config)

    with pytest.raises(LLMResponseError):
        agent.analyze(article, evidence)


def test_provenance_is_preserved_on_the_research_item(config):
    source = analyst_source_for(config, "KPMG")
    article, evidence = make_article_and_evidence(
        source, url="https://example.com/kpmg/x", title="x",
        text="KPMG's new AI insights on AI investment and banking adoption trends.",
    )
    findings = AnalystFindings(thesis="A thesis.")
    assessment = _analyst_assessment(RelevanceLabel.RELEVANT, "ok", findings=findings)
    agent = AnalystAgent(FakeLLMClient([assessment]), config)

    item = agent.analyze(article, evidence)

    assert item.article_id == article.id
    assert item.evidence == [evidence.evidence_id]
    assert item.company == "KPMG"
    assert item.source_quality.source_type == article.source_type


def test_mocked_ollama_failure_propagates_cleanly(config):
    source = analyst_source_for(config, "PwC")
    article, evidence = make_article_and_evidence(
        source, url="https://example.com/pwc/x", title="x",
        text="PwC's new AI insights on AI investment and banking adoption trends.",
    )
    agent = AnalystAgent(FakeLLMClient([LLMUnavailableError("Ollama not running")]), config)

    with pytest.raises(LLMUnavailableError):
        agent.analyze(article, evidence)


def test_supports_restricts_to_exactly_the_five_configured_firms(config):
    agent = AnalystAgent(FakeLLMClient([]), config)
    assert agent._monitored_names == {"McKinsey", "BCG", "Deloitte", "KPMG", "PwC"}

    for firm in ("McKinsey", "BCG", "Deloitte", "KPMG", "PwC"):
        source = analyst_source_for(config, firm)
        article, _ = make_article_and_evidence(source, url=f"https://example.com/{firm}", title="t", text="AI research.")
        assert agent.supports(article) is True

    from research_assistant.models.enums import SourcePriority, SourceType
    from research_assistant.retrieval.models import SourceCategory, SourceDefinition

    bain = SourceDefinition(
        name="Bain", domain="bain.com", urls=[], source_type=SourceType.PRIMARY,
        priority=SourcePriority.MEDIUM, category=SourceCategory.ANALYST,
    )
    other_article, _ = make_article_and_evidence(bain, url="https://bain.com/x", title="t", text="AI research.")
    assert agent.supports(other_article) is False


def test_no_bain_or_accenture_configured_as_analyst_sources(config):
    names = {s.name for s in config.sources.analysts}
    assert names == {"McKinsey", "BCG", "Deloitte", "KPMG", "PwC"}
    assert "Bain" not in names
    assert "Accenture" not in names


def test_analyst_agent_never_imports_retrieval_or_makes_http_calls():
    import research_assistant.agents.analyst as analyst_module

    lines = Path(analyst_module.__file__).read_text(encoding="utf-8").splitlines()
    import_lines = [line.strip() for line in lines if line.strip().startswith(("import ", "from "))]
    assert not any("research_assistant.retrieval" in line for line in import_lines)
    assert not any(line == "import requests" for line in import_lines)
