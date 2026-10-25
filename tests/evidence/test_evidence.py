"""Article -> Evidence conversion and provenance preservation.

Also covers the Evidence model directly (timezone handling, empty-content
rejection) and source-quality propagation (primary/secondary/press
distribution + priority).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from research_assistant.evidence.conversion import article_to_evidence, articles_to_evidence
from research_assistant.models.article import Article
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.models.evidence import Evidence
from research_assistant.retrieval.models import ExtractionMethod


def _article(**overrides) -> Article:
    kwargs = dict(
        id="art-1",
        url="https://example.com/a?utm_source=x",
        canonical_url="https://example.com/a",
        domain="example.com",
        source_name="Example Bank",
        source_type=SourceType.PRIMARY,
        source_priority=SourcePriority.HIGH,
        retrieved_at=datetime(2025, 9, 1, 8, 0, tzinfo=timezone.utc),
        publisher="Example Bank",
        title="A Title",
        author="Jane Doe",
        publication_date=datetime(2025, 8, 30, tzinfo=timezone.utc),
        clean_text="A full article body.",
        content_hash="deadbeef",
        extraction_method=ExtractionMethod.BS4_FALLBACK,
    )
    kwargs.update(overrides)
    return Article(**kwargs)


def test_article_with_content_produces_evidence():
    article = _article()
    evidence = article_to_evidence(article)

    assert evidence is not None
    assert isinstance(evidence, Evidence)
    assert evidence.content == "A full article body."


def test_article_without_content_produces_no_evidence():
    article = _article(clean_text=None, content_hash=None)
    assert article_to_evidence(article) is None


def test_evidence_preserves_full_provenance_chain():
    article = _article()
    evidence = article_to_evidence(article)

    # Where did this come from?
    assert evidence.source_url == article.url
    assert evidence.canonical_url == article.canonical_url
    assert evidence.publisher == article.publisher
    # When was it published, when did we retrieve it?
    assert evidence.publication_date == article.publication_date
    assert evidence.retrieved_at == article.retrieved_at
    # Was the source primary or secondary?
    assert evidence.source_type == article.source_type
    assert evidence.source_priority == article.source_priority
    # How was the content extracted?
    assert evidence.extraction_method == article.extraction_method
    # Traceable back to its Article
    assert evidence.article_id == article.id


def test_evidence_id_is_deterministic_for_the_same_article():
    article = _article()
    e1 = article_to_evidence(article)
    e2 = article_to_evidence(article)
    assert e1.evidence_id == e2.evidence_id


def test_evidence_id_differs_for_different_content():
    a1 = _article(content_hash="hash-one")
    a2 = _article(content_hash="hash-two")
    assert article_to_evidence(a1).evidence_id != article_to_evidence(a2).evidence_id


def test_articles_to_evidence_skips_articles_with_no_content():
    with_content = _article(id="with-content")
    without_content = _article(id="without-content", clean_text=None, content_hash=None)

    result = articles_to_evidence([with_content, without_content])

    assert len(result) == 1
    assert result[0].article_id == "with-content"


@pytest.mark.parametrize(
    "source_type,priority",
    [
        (SourceType.PRIMARY, SourcePriority.HIGH),
        (SourceType.SECONDARY, SourcePriority.HIGH),
        (SourceType.PRESS_DISTRIBUTION, SourcePriority.MEDIUM),
        (SourceType.ANALYST, SourcePriority.LOW),
    ],
)
def test_source_quality_propagates_from_article_to_evidence(source_type, priority):
    article = _article(source_type=source_type, source_priority=priority)
    evidence = article_to_evidence(article)
    assert evidence.source_type == source_type
    assert evidence.source_priority == priority


def test_evidence_content_cannot_be_empty():
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="e1", article_id="a1",
            source_url="https://example.com/a", canonical_url="https://example.com/a",
            retrieved_at=datetime(2025, 9, 1, tzinfo=timezone.utc),
            content="   ", content_hash="hash",
            source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
            extraction_method=ExtractionMethod.BS4_FALLBACK,
        )


def test_evidence_naive_datetime_is_rejected():
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="e1", article_id="a1",
            source_url="https://example.com/a", canonical_url="https://example.com/a",
            retrieved_at=datetime(2025, 9, 1),  # naive
            content="real content", content_hash="hash",
            source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
            extraction_method=ExtractionMethod.BS4_FALLBACK,
        )


def test_evidence_span_is_unset_in_phase_2():
    evidence = article_to_evidence(_article())
    assert evidence.evidence_span is None
