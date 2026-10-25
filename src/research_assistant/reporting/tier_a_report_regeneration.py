"""Explicit, user-triggered regeneration of an already-persisted Tier A
(Event) report.

This is NOT a new reporting concept: it reuses the exact same
`ReportAgent.generate`/`build_link_grounded_report` path
`pipeline.nodes.make_report_generation_node` already uses during a
CURRENT scan, applied to exactly ONE already-formed `Event`, standing
alone outside any `PipelineState`/run context. It re-aggregates facts
from the event's already-persisted `ResearchItem`s (never re-running
specialist analysis, never re-discovering or re-fetching anything) and
overwrites that event's existing `reports` row by its own `event_id`
primary key -- a plain UPDATE of an existing report by its own identity,
not a new table, not a new concept.

The only reason this exists is so a report generated/sanitized under an
OLDER prompt/guardrail version (e.g. before a grounding-guardrail fix)
can be refreshed on demand, without needing a fresh CURRENT scan. It is
deliberately never called from article selection or from
`render_newsletter_markdown`/`render_event_list_email` -- those stay
pure. The newsletter UI is the only caller, and only in response to an
explicit per-event "Regenerate report" click.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from research_assistant.agents.report import ReportAgent, build_link_grounded_report
from research_assistant.config.schema import AppConfig
from research_assistant.llm.client import create_ollama_client
from research_assistant.llm.errors import LLMError
from research_assistant.models.enums import GroundingType
from research_assistant.models.report import Report, ReportFailure
from research_assistant.storage.persistence_service import PersistenceService

logger = logging.getLogger(__name__)


def regenerate_tier_a_report(
    service: PersistenceService, config: AppConfig, event_id: str,
    *, report_agent: Optional[ReportAgent] = None,
) -> Optional[Union[Report, ReportFailure]]:
    """Returns `None` only if `event_id` does not exist. Otherwise always
    persists its result (a successful `Report` overwrites the existing
    row; a failure persists a `ReportFailure`, exactly like a first-time
    generation during a CURRENT scan) and returns it. Raises nothing the
    caller needs to handle beyond inspecting the returned type -- every
    LLM failure is already caught and converted to a `ReportFailure`,
    matching `ReportAgent.generate_many`'s own contract."""
    conn = service.database.connect()
    try:
        event = service.events.get(conn, event_id)
        if event is None:
            return None

        item_ids = service.events.get_research_item_ids(conn, event_id)
        items = service.research_items.get_many(conn, item_ids)

        if event.grounding_type == GroundingType.LINK:
            result: Union[Report, ReportFailure] = build_link_grounded_report(event, items)
        else:
            agent = report_agent or ReportAgent(create_ollama_client(config))
            try:
                result = agent.generate(event, items)
            except LLMError as exc:
                logger.warning("Tier A report regeneration failed for event %s: %s", event_id, exc)
                result = ReportFailure(event_id=event_id, reason=type(exc).__name__, detail=str(exc))

        if isinstance(result, Report):
            service.reports.upsert_report(conn, result)
        else:
            service.reports.upsert_failure(conn, result)
        conn.commit()
        return result
    finally:
        conn.close()
