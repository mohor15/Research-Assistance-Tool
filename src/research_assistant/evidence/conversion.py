"""Article -> Evidence. Deterministic, no LLM, no summarization.

An Article with no usable text (failed/empty extraction) produces no
Evidence at all — there is nothing source-grounded to hand an agent.
"""

from __future__ import annotations

from typing import List, Optional

from research_assistant.evidence.ids import generate_evidence_id
from research_assistant.models.article import Article
from research_assistant.models.evidence import Evidence


def evidence_rejection_reason(article: Article) -> Optional[str]:
    """Explains why `article_to_evidence(article)` would return `None`, or
    `None` if it would succeed. Mirrors that function's gate exactly (the
    gate itself is unchanged) — this exists purely to make an otherwise
    silent rejection explicit and testable for pipeline-level diagnostics.
    """
    missing_clean_text = not article.clean_text
    missing_content_hash = not article.content_hash
    if missing_clean_text and missing_content_hash:
        return "missing_clean_text_and_content_hash"
    if missing_clean_text:
        return "missing_clean_text"
    if missing_content_hash:
        return "missing_content_hash"
    return None


def article_to_evidence(article: Article) -> Optional[Evidence]:
    if not article.clean_text or not article.content_hash:
        return None

    return Evidence(
        evidence_id=generate_evidence_id(article.id, article.content_hash),
        article_id=article.id,
        source_url=article.url,
        canonical_url=article.canonical_url,
        publisher=article.publisher,
        title=article.title,
        publication_date=article.publication_date,
        retrieved_at=article.retrieved_at,
        content=article.clean_text,
        content_hash=article.content_hash,
        source_type=article.source_type,
        source_priority=article.source_priority,
        extraction_method=article.extraction_method,
        metadata=dict(article.metadata),
    )


def articles_to_evidence(articles: List[Article]) -> List[Evidence]:
    evidence = []
    for article in articles:
        item = article_to_evidence(article)
        if item is not None:
            evidence.append(item)
    return evidence
