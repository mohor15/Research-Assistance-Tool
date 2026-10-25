"""Fakes for LangGraph pipeline tests. Nothing here touches the network or a
live Ollama server — every service is a scripted/fake stand-in."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pytest

from research_assistant.agents.triage import TriageBatchResponse, TriageItemResult
from research_assistant.config import get_config
from research_assistant.evidence.service import NormalizationService
from research_assistant.events.clock import FixedClock
from research_assistant.events.service import EventService
from research_assistant.models.enums import EventType, SourcePriority, SourceType
from research_assistant.models.research_item import TopicCategory
from research_assistant.reporting.digest_service import DigestService
from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def config():
    return get_config()


def jpmorgan_source() -> SourceDefinition:
    return SourceDefinition(
        name="JPMorgan Chase", domain="jpmorganchase.com", urls=["https://jpmorganchase.com/newsroom"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )


class FakeRetrievalService:
    """Returns canned ExtractedArticles for configured sources — no HTTP."""

    def __init__(self, *, sources: List[SourceDefinition], articles_by_source: Dict[str, List[ExtractedArticle]], failing_sources: Optional[set] = None):
        self._sources = sources
        self._articles_by_source = articles_by_source
        self._failing_sources = failing_sources or set()

    def list_sources(self) -> List[SourceDefinition]:
        return list(self._sources)

    def retrieve_source(self, source: SourceDefinition, *, cache_policy=None) -> List[ExtractedArticle]:
        # `cache_policy` accepted (and ignored) only to match the real
        # RetrievalService's signature -- see pipeline.nodes.make_retrieve_node,
        # which always passes it; this fake never touches a real cache.
        if source.name in self._failing_sources:
            raise ConnectionError(f"simulated retrieval failure for {source.name}")
        return list(self._articles_by_source.get(source.name, []))


_CANDIDATE_ID_RE = re.compile(r"--- Candidate (\S+) ---")


class FakeTriageLLMClient:
    """Stand-in for CURRENT's dedicated triage LLM client -- never touches
    a live Ollama server. Parses the real candidate ids straight out of the
    prompt text `agents.triage.build_triage_prompt` produces (tests can't
    know a candidate's hash-derived article id in advance), so it works
    against any real routed pool without per-test wiring. By default marks
    every candidate in a batch relevant with a generic PARTNERSHIP/
    ENTERPRISE_AI classification; `relevant_ids`/`irrelevant_ids` (both
    optional) override specific ids either way. `calls` records each
    prompt seen, for tests that want to assert batching/call counts."""

    def __init__(
        self, *, relevant_ids: Optional[set] = None, irrelevant_ids: Optional[set] = None,
        default_relevant: bool = True,
    ) -> None:
        self._relevant_ids = relevant_ids or set()
        self._irrelevant_ids = irrelevant_ids or set()
        self._default_relevant = default_relevant
        self.calls: List[str] = []

    def generate_structured(self, *, system: str, prompt: str, schema):
        self.calls.append(prompt)
        candidate_ids = _CANDIDATE_ID_RE.findall(prompt)
        results = []
        for candidate_id in candidate_ids:
            if candidate_id in self._irrelevant_ids:
                relevant = False
            elif candidate_id in self._relevant_ids:
                relevant = True
            else:
                relevant = self._default_relevant
            results.append(
                TriageItemResult(
                    candidate_id=candidate_id, relevant=relevant,
                    event_type=EventType.PARTNERSHIP if relevant else None,
                    category=TopicCategory.ENTERPRISE_AI if relevant else None,
                    confidence=0.8 if relevant else 0.2,
                )
            )
        return TriageBatchResponse(results=results)


def real_normalization_service() -> NormalizationService:
    return NormalizationService()


def real_event_service(config, clock=None) -> EventService:
    return EventService(config, clock=clock or FixedClock(D))


def real_digest_service(config, clock=None) -> DigestService:
    return DigestService(config, clock=clock or FixedClock(D))
