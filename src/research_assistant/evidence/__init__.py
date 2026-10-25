"""Article normalization and evidence generation — the layer between
retrieval and future agents:

    ExtractedArticle + RetrievedPage + SourceDefinition
                          |
                normalize_article()
                          |
                       Article
                          |
                deduplicate_articles()
                          |
                  deduplicated Articles
                          |
                  article_to_evidence()
                          |
                       Evidence

`NormalizationService` is the intended single entry point for later phases,
mirroring how `RetrievalService` is Phase 1's entry point. Nothing in this
package calls an LLM or makes a network request — normalization is
deterministic.
"""

from research_assistant.evidence.conversion import article_to_evidence, articles_to_evidence
from research_assistant.evidence.dedup import deduplicate_articles
from research_assistant.evidence.hashing import hash_content
from research_assistant.evidence.normalizer import normalize_article, normalize_from_page, normalize_many
from research_assistant.evidence.service import NormalizationService
from research_assistant.evidence.urls import canonicalize_url

__all__ = [
    "NormalizationService",
    "normalize_article",
    "normalize_from_page",
    "normalize_many",
    "deduplicate_articles",
    "article_to_evidence",
    "articles_to_evidence",
    "canonicalize_url",
    "hash_content",
]
