"""The single entry point later phases should use for normalization.

Mirrors `RetrievalService` from Phase 1: agents/pipeline code depends on
this class rather than the individual normalization/dedup/conversion
functions, so the internals can evolve without breaking callers.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from research_assistant.evidence.conversion import article_to_evidence, articles_to_evidence
from research_assistant.evidence.dedup import deduplicate_articles
from research_assistant.evidence.normalizer import normalize_article, normalize_from_page, normalize_many
from research_assistant.models.article import Article
from research_assistant.models.evidence import Evidence
from research_assistant.retrieval.models import ExtractedArticle, RetrievedPage, SourceDefinition


class NormalizationService:
    def normalize(
        self,
        extracted: ExtractedArticle,
        *,
        source: SourceDefinition,
        page: Optional[RetrievedPage] = None,
    ) -> Article:
        if page is not None:
            return normalize_from_page(extracted, source=source, page=page)
        return normalize_article(extracted, source=source)

    def normalize_many(self, items: List[Tuple[SourceDefinition, ExtractedArticle]]) -> List[Article]:
        return normalize_many(items)

    def deduplicate(self, articles: List[Article]) -> List[Article]:
        return deduplicate_articles(articles)

    def to_evidence(self, article: Article) -> Optional[Evidence]:
        return article_to_evidence(article)

    def to_evidence_many(self, articles: List[Article]) -> List[Evidence]:
        return articles_to_evidence(articles)

    def process(self, items: List[Tuple[SourceDefinition, ExtractedArticle]]) -> List[Evidence]:
        """Full pipeline: normalize -> deduplicate -> convert to Evidence."""
        articles = self.normalize_many(items)
        deduped = self.deduplicate(articles)
        return self.to_evidence_many(deduped)
