"""StartupAgent — discovery-driven coverage of AI/fintech startups.

Unlike PeerAgent/AIProductAgent, there is no fixed company registry to
restrict to (config/sources.yaml's `startups` section is
categories/event_types/discovery metadata, not a list of named sources —
see `research_assistant.retrieval.service._SCRAPABLE_CATEGORIES`, which
deliberately excludes `startups` for the same reason). `supports()`
therefore accepts any article (`monitored_names=None`), and the configured
`categories`/`funding_events` are folded into the keyword pre-filter
instead of an enormous hardcoded startup list.
"""

from __future__ import annotations

from typing import ClassVar, List

from research_assistant.agents.base import ResearchAgent
from research_assistant.config.schema import AppConfig
from research_assistant.llm.client import LLMClient
from research_assistant.models.enums import AgentName


class StartupAgent(ResearchAgent):
    agent_name: ClassVar[AgentName] = AgentName.STARTUP
    system_prompt_file: ClassVar[str] = "startup_system.md"

    def __init__(self, llm_client: LLMClient, config: AppConfig) -> None:
        super().__init__(llm_client, config, monitored_names=None)

    def _extra_keywords(self) -> List[str]:
        startups = self._config.sources.startups
        return list(startups.categories) + list(startups.funding_events) + list(startups.event_types)
