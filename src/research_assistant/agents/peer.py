"""PeerAgent — monitors the configured peer banks (config/sources.yaml: peers)
for meaningful AI/GenAI developments relevant to financial services."""

from __future__ import annotations

from typing import ClassVar

from research_assistant.agents.base import ResearchAgent
from research_assistant.config.schema import AppConfig
from research_assistant.llm.client import LLMClient
from research_assistant.models.enums import AgentName


class PeerAgent(ResearchAgent):
    agent_name: ClassVar[AgentName] = AgentName.PEER
    system_prompt_file: ClassVar[str] = "peer_system.md"

    def __init__(self, llm_client: LLMClient, config: AppConfig) -> None:
        monitored_names = {source.name for source in config.sources.peers}
        super().__init__(llm_client, config, monitored_names=monitored_names)
