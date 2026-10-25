"""AIProductAgent — monitors the configured AI providers (config/sources.yaml:
ai_providers) as market intelligence on the broader AI ecosystem."""

from __future__ import annotations

from typing import ClassVar

from research_assistant.agents.base import ResearchAgent
from research_assistant.config.schema import AppConfig
from research_assistant.llm.client import LLMClient
from research_assistant.models.enums import AgentName


class AIProductAgent(ResearchAgent):
    agent_name: ClassVar[AgentName] = AgentName.AI_PRODUCT
    system_prompt_file: ClassVar[str] = "ai_product_system.md"

    def __init__(self, llm_client: LLMClient, config: AppConfig) -> None:
        monitored_names = {source.name for source in config.sources.ai_providers}
        super().__init__(llm_client, config, monitored_names=monitored_names)
