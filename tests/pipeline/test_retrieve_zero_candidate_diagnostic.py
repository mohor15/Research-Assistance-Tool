"""Phase 13E: a source that was reachable but yielded zero candidates must
be diagnosable as such, distinct both from a hard retrieval failure and
from silent, indistinguishable "healthy" success -- otherwise there is no
way to tell "genuinely nothing new this run" apart from "something is
quietly wrong with this source's discovery" (Section 16).
"""

from __future__ import annotations

from research_assistant.pipeline.nodes import make_retrieve_node
from research_assistant.pipeline.state import PipelineState

from tests.pipeline.conftest import FakeRetrievalService, jpmorgan_source
from tests.pipeline.test_graph import _build_services, _jpmorgan_article


def test_a_reachable_source_with_zero_candidates_gets_an_explicit_diagnostic(config):
    services = _build_services(
        config,
        peer_llm_responses=[], report_llm_responses=[],
        retrieval_service=FakeRetrievalService(sources=[jpmorgan_source()], articles_by_source={}),
    )
    node = make_retrieve_node(services)

    result = node(PipelineState(sources=[jpmorgan_source()]))

    zero_candidate_diagnostics = [
        d for d in result["diagnostics"]
        if d.stage == "retrieve" and "zero candidates" in d.message
    ]
    assert len(zero_candidate_diagnostics) == 1
    assert zero_candidate_diagnostics[0].severity == "warning"
    assert "JPMorgan Chase" in zero_candidate_diagnostics[0].message
    assert result["metrics"]["sources_succeeded"] == 1  # still a successful fetch, just empty


def test_a_source_with_real_candidates_gets_no_zero_candidate_diagnostic(config):
    services = _build_services(
        config,
        peer_llm_responses=[], report_llm_responses=[],
        retrieval_service=FakeRetrievalService(
            sources=[jpmorgan_source()], articles_by_source={"JPMorgan Chase": [_jpmorgan_article()]},
        ),
    )
    node = make_retrieve_node(services)

    result = node(PipelineState(sources=[jpmorgan_source()]))

    assert not any("zero candidates" in d.message for d in result["diagnostics"])


def test_a_hard_retrieval_failure_is_unaffected_by_this_diagnostic(config):
    """The pre-existing SourceUnreachableError-style failure diagnostic is
    unchanged -- the new zero-candidate diagnostic is additive, never a
    replacement."""
    services = _build_services(
        config,
        peer_llm_responses=[], report_llm_responses=[],
        retrieval_service=FakeRetrievalService(
            sources=[jpmorgan_source()], articles_by_source={}, failing_sources={"JPMorgan Chase"},
        ),
    )
    node = make_retrieve_node(services)

    result = node(PipelineState(sources=[jpmorgan_source()]))

    assert not any("zero candidates" in d.message for d in result["diagnostics"])
    assert any(d.severity == "error" for d in result["diagnostics"])
    assert result["metrics"]["sources_failed"] == 1
