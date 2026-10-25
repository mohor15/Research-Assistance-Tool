"""LangGraph orchestration of the end-to-end research pipeline (Phase 6),
plus the `PipelineRunner`/run-identification layer used by scheduling and
manual runs (Phase 8).

`build_graph(services)` compiles the graph; `PipelineServices` is the
dependency-injection point for every real service (RetrievalService,
NormalizationService, the research agents, EventService, ReportAgent,
DigestService, PersistenceService) — nothing here reimplements their
logic, and nothing here holds hidden global state.

`PipelineRunner` (Phase 8) wraps `build_graph` with run identification (a
deterministic `run_id`), status determination, and structured counts — it
is orchestration only and never duplicates a pipeline stage. Import it
from its own submodule (`research_assistant.pipeline.runner`), not from
this package's top level: `runner.py` reaches into `research_assistant.storage`
(for `PipelineRunner.build()`'s real `Database`/`PersistenceService`), and
`storage.repositories.run` reaches back into `pipeline.state` — eagerly
re-exporting `PipelineRunner` here would recreate the exact import-cycle
class of bug fixed in Phase 2 (see `research_assistant.models`'s docstring).
"""

from research_assistant.pipeline.graph import build_graph
from research_assistant.pipeline.nodes import PipelineServices
from research_assistant.pipeline.state import ExtractedBundle, PipelineDiagnostic, PipelineState

__all__ = ["build_graph", "PipelineServices", "PipelineState", "PipelineDiagnostic", "ExtractedBundle"]
