"""Phase-9-hotfix regression test (Section 14): if a peer article
(`source_type=primary`, `source_name` in `config.sources.peers`) survives
the publication-period filter and produces Evidence, it MUST be eligible
for the Peer Agent — there must be no accidental path where peer source
metadata disappears between article -> evidence -> routing -> agent
candidates. This is the exact real-run regression
(`agent_analysis: peer candidates=0` despite peer sources being
successfully retrieved) reproduced at the full-node level, not just at the
router-unit level (see tests/agents/test_router.py for the router-only
coverage).
"""

from __future__ import annotations

from research_assistant.models.enums import AgentName
from research_assistant.pipeline.nodes import make_agent_analysis_node
from research_assistant.pipeline.state import PipelineState

from tests.pipeline.test_graph import FakeLLMClient, _build_services, _jpmorgan_article, _relevant_assessment


def test_peer_agent_is_not_empty_when_a_peer_article_with_evidence_exists(config):
    services = _build_services(config, peer_llm_responses=[_relevant_assessment()], report_llm_responses=[])
    article = services.normalization_service.normalize(_jpmorgan_article(), source=services.retrieval_service.list_sources()[0])
    evidence = services.normalization_service.to_evidence(article)
    assert article.source_name == "JPMorgan Chase"  # sanity: this really is a configured peer source

    state = PipelineState(articles=[article], evidence=[evidence])
    node = make_agent_analysis_node(services)
    result = node(state)

    assert result["metrics"]["agent_candidates_considered"] >= 1
    assert len(result["research_items"]) == 1
    assert result["research_items"][0].agent == AgentName.PEER
    assert result["metrics"]["agent_failures"] == 0
