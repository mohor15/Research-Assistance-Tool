"""pipeline.nodes.make_retrieve_node must select CachePolicy.FRESH only for
DigestType.CURRENT, and CachePolicy.CACHE (the pre-existing behavior) for
every other digest_type -- Morning/Evening/manual must be completely
unaffected by the CURRENT freshness fix."""

from __future__ import annotations

from research_assistant.models.digest import DigestType
from research_assistant.pipeline.nodes import make_retrieve_node
from research_assistant.pipeline.state import PipelineState
from research_assistant.retrieval.models import CachePolicy

from tests.pipeline.conftest import jpmorgan_source
from tests.pipeline.test_graph import _build_services


class _RecordingRetrievalService:
    def __init__(self, sources):
        self._sources = sources
        self.received_cache_policies = []

    def list_sources(self):
        return list(self._sources)

    def retrieve_source(self, source, *, cache_policy=None):
        self.received_cache_policies.append(cache_policy)
        return []


def test_current_digest_type_uses_fresh_cache_policy(config):
    retrieval_service = _RecordingRetrievalService([jpmorgan_source()])
    services = _build_services(
        config, peer_llm_responses=[], report_llm_responses=[], retrieval_service=retrieval_service,
    )
    node = make_retrieve_node(services)

    node(PipelineState(digest_type=DigestType.CURRENT, sources=[jpmorgan_source()]))

    assert retrieval_service.received_cache_policies == [CachePolicy.FRESH]


def test_morning_digest_type_uses_cache_policy_unchanged(config):
    retrieval_service = _RecordingRetrievalService([jpmorgan_source()])
    services = _build_services(
        config, peer_llm_responses=[], report_llm_responses=[], retrieval_service=retrieval_service,
    )
    node = make_retrieve_node(services)

    node(PipelineState(digest_type=DigestType.MORNING, sources=[jpmorgan_source()]))

    assert retrieval_service.received_cache_policies == [CachePolicy.CACHE]


def test_evening_digest_type_uses_cache_policy_unchanged(config):
    retrieval_service = _RecordingRetrievalService([jpmorgan_source()])
    services = _build_services(
        config, peer_llm_responses=[], report_llm_responses=[], retrieval_service=retrieval_service,
    )
    node = make_retrieve_node(services)

    node(PipelineState(digest_type=DigestType.EVENING, sources=[jpmorgan_source()]))

    assert retrieval_service.received_cache_policies == [CachePolicy.CACHE]


def test_default_pipeline_state_digest_type_uses_cache_policy_unchanged(config):
    """PipelineState() with no digest_type specified defaults to MORNING --
    proves a caller that never heard of CURRENT gets exactly the old
    behavior, not a surprise."""
    retrieval_service = _RecordingRetrievalService([jpmorgan_source()])
    services = _build_services(
        config, peer_llm_responses=[], report_llm_responses=[], retrieval_service=retrieval_service,
    )
    node = make_retrieve_node(services)

    node(PipelineState(sources=[jpmorgan_source()]))

    assert retrieval_service.received_cache_policies == [CachePolicy.CACHE]
