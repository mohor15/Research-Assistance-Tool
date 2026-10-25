"""LangGraph orchestration of the Phase 0–6 pipeline.

Every node (`pipeline.nodes`) is a thin wrapper around an already-existing,
already-tested service — RetrievalService, NormalizationService, the
research agents, EventService, ReportAgent, DigestService. This module
contains no retrieval/normalization/scoring/clustering logic of its own;
LangGraph is only responsible for sequencing and state-passing.

    START -> load_config -> discover_sources -> retrieve -> normalize ->
    evidence -> agent_analysis -> event_scoring -> report_generation ->
    digest_generation -> persistence -> END

`persistence` (Phase 7) is a thin final node wrapping `PersistenceService`;
it is a no-op when `services.persistence_service` is `None`, so this graph
behaves exactly as it did in Phase 6 for any caller that doesn't want
SQLite involved.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_assistant.pipeline.nodes import (
    PipelineServices,
    make_agent_analysis_node,
    make_digest_generation_node,
    make_discover_sources_node,
    make_event_scoring_node,
    make_evidence_node,
    make_load_config_node,
    make_normalize_node,
    make_persistence_node,
    make_report_generation_node,
    make_retrieve_node,
)
from research_assistant.pipeline.state import PipelineState

NODE_ORDER = (
    "load_config",
    "discover_sources",
    "retrieve",
    "normalize",
    "evidence",
    "agent_analysis",
    "event_scoring",
    "report_generation",
    "digest_generation",
    "persistence",
)


def build_graph(services: PipelineServices) -> CompiledStateGraph:
    graph: StateGraph = StateGraph(PipelineState)

    graph.add_node("load_config", make_load_config_node(services))
    graph.add_node("discover_sources", make_discover_sources_node(services))
    graph.add_node("retrieve", make_retrieve_node(services))
    graph.add_node("normalize", make_normalize_node(services))
    graph.add_node("evidence", make_evidence_node(services))
    graph.add_node("agent_analysis", make_agent_analysis_node(services))
    graph.add_node("event_scoring", make_event_scoring_node(services))
    graph.add_node("report_generation", make_report_generation_node(services))
    graph.add_node("digest_generation", make_digest_generation_node(services))
    graph.add_node("persistence", make_persistence_node(services))

    graph.add_edge(START, NODE_ORDER[0])
    for current_node, next_node in zip(NODE_ORDER, NODE_ORDER[1:]):
        graph.add_edge(current_node, next_node)
    graph.add_edge(NODE_ORDER[-1], END)

    return graph.compile()
