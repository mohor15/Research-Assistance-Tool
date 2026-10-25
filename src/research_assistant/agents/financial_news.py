"""FinancialNewsAgent — monitors the configured secondary/press-distribution
financial & technology news sources (config/sources.yaml: secondary_sources
-- Reuters, Financial Times, Bloomberg, CNBC, Barron's, Wall Street Journal,
TechCrunch, The Information, PR Newswire) for substantive AI, fintech, and
financial-services developments.

Unlike PeerAgent/AIProductAgent/AnalystAgent, these sources carry no
primary-source confirmation requirement: a Reuters or FT article is
independently valid intelligence even when the subject bank/AI company's
own newsroom has not published anything about it (see `agents.router`,
which routes a secondary-source article here by source membership, exactly
like the other three agents route by their own registries -- source
identity controls ROUTING only, never the final relevance decision, which
remains the LLM's via the unchanged `ResearchAgent.analyze()` contract).
"""

from __future__ import annotations

from typing import ClassVar, List

from research_assistant.agents.base import ResearchAgent
from research_assistant.config.schema import AppConfig
from research_assistant.llm.client import LLMClient
from research_assistant.models.enums import AgentName


class FinancialNewsAgent(ResearchAgent):
    agent_name: ClassVar[AgentName] = AgentName.FINANCIAL_NEWS
    system_prompt_file: ClassVar[str] = "financial_news_system.md"

    def __init__(self, llm_client: LLMClient, config: AppConfig) -> None:
        monitored_names = {source.name for source in config.sources.secondary_sources}
        super().__init__(llm_client, config, monitored_names=monitored_names)

    def _extra_keywords(self) -> List[str]:
        """Widen the pre-filter beyond generic AI topics to the full
        financial-services vocabulary (banking, capital markets, fintech,
        risk, operations) -- the same, already-configured
        `config/topics.yaml` buckets `AnalystAgent` already reuses, not a
        new or duplicated list."""
        topics = self._config.topics
        return (
            list(topics.strategic_themes)
            + list(topics.financial_services.core)
            + list(topics.financial_services.markets)
            + list(topics.financial_services.wealth)
            + list(topics.financial_services.risk)
            + list(topics.financial_services.operations)
        )
