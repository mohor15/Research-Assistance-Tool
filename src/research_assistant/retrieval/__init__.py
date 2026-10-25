"""Shared retrieval engine: source discovery, HTTP fetch, extraction, browser fallback.

`RetrievalService` is the intended public interface — agents and pipeline
code should depend on it rather than importing individual retrieval
submodules, so scraping/parsing details can change without breaking agents.
"""

from research_assistant.retrieval.models import (
    CandidateURL,
    ExtractedArticle,
    ExtractionMethod,
    ExtractionStatus,
    RetrievedPage,
    SourceCategory,
    SourceDefinition,
)
from research_assistant.retrieval.service import RetrievalService, build_source_definitions

__all__ = [
    "RetrievalService",
    "build_source_definitions",
    "SourceDefinition",
    "SourceCategory",
    "CandidateURL",
    "RetrievedPage",
    "ExtractedArticle",
    "ExtractionMethod",
    "ExtractionStatus",
]
