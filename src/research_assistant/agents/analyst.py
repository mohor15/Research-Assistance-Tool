"""AnalystAgent — monitors exactly the five configured consulting/strategy
sources (config/sources.yaml: analysts) for meaningful AI-strategy research."""

from __future__ import annotations

from typing import ClassVar, List, Type

from research_assistant.agents.base import ResearchAgent
from research_assistant.agents.schemas import AgentAssessment, AnalystAssessment
from research_assistant.config.schema import AppConfig
from research_assistant.llm.client import LLMClient
from research_assistant.models.enums import AgentName


class AnalystAgent(ResearchAgent):
    agent_name: ClassVar[AgentName] = AgentName.ANALYST
    system_prompt_file: ClassVar[str] = "analyst_system.md"
    assessment_schema: ClassVar[Type[AgentAssessment]] = AnalystAssessment

    def __init__(self, llm_client: LLMClient, config: AppConfig) -> None:
        monitored_names = {source.name for source in config.sources.analysts}
        super().__init__(llm_client, config, monitored_names=monitored_names)

    def _extra_keywords(self) -> List[str]:
        """Analyst focus (AI economics/adoption/investment, banking, capital
        markets, future of work) spans config/topics.yaml's strategic_themes
        and financial_services buckets, not just ai_topics — widen the
        pre-filter accordingly, still entirely config-driven."""
        topics = self._config.topics
        return (
            list(topics.strategic_themes)
            + list(topics.financial_services.core)
            + list(topics.financial_services.markets)
        )
