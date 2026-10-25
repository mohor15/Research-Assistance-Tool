"""Shared fixtures for agent tests. Every test here uses a FakeLLMClient — none
requires (or waits on) a live Ollama server."""

from __future__ import annotations

from typing import List, Optional, Tuple, Type, TypeVar

import pytest
from pydantic import BaseModel

from research_assistant.config import get_config
from research_assistant.evidence.conversion import article_to_evidence
from research_assistant.evidence.normalizer import normalize_article
from research_assistant.llm.errors import LLMError
from research_assistant.models.article import Article
from research_assistant.models.evidence import Evidence
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition

T = TypeVar("T", bound=BaseModel)


class FakeLLMClient:
    """Scripted stand-in for `OllamaClient`. `responses` is a list popped in
    order; an item that is an Exception is raised instead of returned."""

    def __init__(self, responses: List):
        self._responses = list(responses)
        self.calls: List[Tuple[str, str, Type]] = []

    def generate_structured(self, *, system: str, prompt: str, schema: Type[T]) -> T:
        self.calls.append((system, prompt, schema))
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def config():
    return get_config()


@pytest.fixture
def peer_source() -> SourceDefinition:
    return SourceDefinition(
        name="JPMorgan Chase",
        domain="jpmorganchase.com",
        urls=["https://www.jpmorganchase.com/newsroom"],
        source_type=SourceType.PRIMARY,
        priority=SourcePriority.HIGH,
        category=SourceCategory.PEER,
    )


@pytest.fixture
def ai_provider_source() -> SourceDefinition:
    return SourceDefinition(
        name="OpenAI",
        domain="openai.com",
        urls=["https://openai.com/news/"],
        source_type=SourceType.PRIMARY,
        priority=SourcePriority.HIGH,
        category=SourceCategory.AI_PROVIDER,
    )


@pytest.fixture
def startup_source() -> SourceDefinition:
    """A representative discovered-startup source — StartupAgent has no
    fixed registry, so this is just illustrative, not a configured entry."""
    return SourceDefinition(
        name="Acme AI",
        domain="acmeai.example.com",
        urls=[],
        source_type=SourceType.PRIMARY,
        priority=SourcePriority.MEDIUM,
        category=SourceCategory.STARTUP,
    )


def analyst_source_for(config, name: str) -> SourceDefinition:
    """Build the SourceDefinition for one of the five configured analyst firms."""
    entry = next(s for s in config.sources.analysts if s.name == name)
    return SourceDefinition(
        name=entry.name,
        domain=entry.domain,
        urls=entry.urls,
        source_type=entry.source_type,
        priority=entry.priority,
        category=SourceCategory.ANALYST,
    )


def secondary_source_for(config, name: str) -> SourceDefinition:
    """Build the SourceDefinition for one of the configured secondary/
    financial-news sources (Reuters, Financial Times, Bloomberg, ...)."""
    entry = next(s for s in config.sources.secondary_sources if s.name == name)
    return SourceDefinition(
        name=entry.name,
        domain=entry.domain,
        urls=entry.urls,
        source_type=entry.source_type,
        priority=entry.priority,
        category=SourceCategory.SECONDARY,
    )


@pytest.fixture
def reuters_source(config) -> SourceDefinition:
    return secondary_source_for(config, "Reuters")


def make_article_and_evidence(
    source: SourceDefinition, *, url: str, title: Optional[str], text: Optional[str]
) -> Tuple[Article, Optional[Evidence]]:
    extracted = ExtractedArticle(url=url, title=title, text=text)
    article = normalize_article(extracted, source=source)
    evidence = article_to_evidence(article)
    return article, evidence
