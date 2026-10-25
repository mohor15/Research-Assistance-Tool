"""End-to-end: retrieval-shaped ExtractedArticles -> normalized, deduplicated Evidence."""

from __future__ import annotations

from research_assistant.evidence.service import NormalizationService
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.models import (
    ExtractedArticle,
    ExtractionMethod,
    ExtractionStatus,
    SourceCategory,
    SourceDefinition,
)


def _source() -> SourceDefinition:
    return SourceDefinition(
        name="Example Bank", domain="example.com", urls=["https://example.com/newsroom"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )


def test_process_normalizes_dedupes_and_produces_evidence():
    source = _source()
    duplicate_variant = ExtractedArticle(
        url="https://example.com/news/a?utm_source=twitter", title="Story A",
        text="This is the article body for story A, long enough to be real content.",
        extraction_method=ExtractionMethod.BS4_FALLBACK, status=ExtractionStatus.SUCCESS,
    )
    same_article_trailing_slash = ExtractedArticle(
        url="https://example.com/news/a/", title="Story A",
        text="This is the article body for story A, long enough to be real content.",
        extraction_method=ExtractionMethod.BS4_FALLBACK, status=ExtractionStatus.SUCCESS,
    )
    distinct_article = ExtractedArticle(
        url="https://example.com/news/b", title="Story B",
        text="This is a completely different article body for story B.",
        extraction_method=ExtractionMethod.BS4_FALLBACK, status=ExtractionStatus.SUCCESS,
    )
    failed_extraction = ExtractedArticle(url="https://example.com/news/c", text=None, status=ExtractionStatus.FAILED)

    service = NormalizationService()
    evidence = service.process([
        (source, duplicate_variant),
        (source, same_article_trailing_slash),
        (source, distinct_article),
        (source, failed_extraction),
    ])

    # duplicate_variant + same_article_trailing_slash collapse to one; distinct_article stays separate;
    # failed_extraction never produces evidence at all.
    assert len(evidence) == 2
    titles = {e.title for e in evidence}
    assert titles == {"Story A", "Story B"}
    for item in evidence:
        assert item.source_type == SourceType.PRIMARY
        assert item.content  # never empty


def test_normalize_then_deduplicate_then_to_evidence_stepwise_matches_process():
    source = _source()
    items = [
        (source, ExtractedArticle(url="https://example.com/a", text="Body one.")),
        (source, ExtractedArticle(url="https://example.com/a/", text="Body one.")),
    ]
    service = NormalizationService()

    articles = service.normalize_many(items)
    deduped = service.deduplicate(articles)
    stepwise_evidence = service.to_evidence_many(deduped)

    process_evidence = service.process(items)

    assert len(stepwise_evidence) == len(process_evidence) == 1
