"""Article model: valid construction, missing optional metadata, invalid fields, timezone handling."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from research_assistant.models.article import Article, DatePrecision
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.models import ExtractionMethod, ExtractionStatus


def _base_kwargs(**overrides):
    kwargs = dict(
        id="abc123",
        url="https://example.com/a",
        canonical_url="https://example.com/a",
        domain="example.com",
        source_name="Example Bank",
        source_type=SourceType.PRIMARY,
        source_priority=SourcePriority.HIGH,
        retrieved_at=datetime(2025, 9, 1, tzinfo=timezone.utc),
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_article_constructs_with_all_fields():
    article = Article(
        **_base_kwargs(
            publisher="Example Bank",
            title="A Title",
            author="Jane Doe",
            publication_date=datetime(2025, 8, 1, 9, 0, tzinfo=timezone.utc),
            publication_date_precision=DatePrecision.EXACT,
            raw_text="raw",
            clean_text="clean",
            content_hash="deadbeef",
            extraction_method=ExtractionMethod.BS4_FALLBACK,
            extraction_status=ExtractionStatus.SUCCESS,
        )
    )
    assert article.title == "A Title"
    assert article.source_type == SourceType.PRIMARY


def test_missing_optional_metadata_defaults_cleanly():
    article = Article(**_base_kwargs())
    assert article.publisher is None
    assert article.title is None
    assert article.author is None
    assert article.publication_date is None
    assert article.publication_date_precision == DatePrecision.UNKNOWN
    assert article.content_hash is None
    assert article.metadata == {}


def test_naive_publication_date_is_rejected():
    with pytest.raises(ValidationError):
        Article(**_base_kwargs(publication_date=datetime(2025, 8, 1)))  # no tzinfo


def test_naive_retrieved_at_is_rejected():
    kwargs = _base_kwargs()
    kwargs["retrieved_at"] = datetime(2025, 8, 1)  # no tzinfo
    with pytest.raises(ValidationError):
        Article(**kwargs)


def test_missing_required_field_is_rejected():
    kwargs = _base_kwargs()
    del kwargs["url"]
    with pytest.raises(ValidationError):
        Article(**kwargs)


def test_invalid_source_type_is_rejected():
    kwargs = _base_kwargs()
    kwargs["source_type"] = "not-a-real-type"
    with pytest.raises(ValidationError):
        Article(**kwargs)


def test_timezone_aware_dates_in_non_utc_zone_are_accepted():
    from datetime import timedelta, timezone as tz

    ist = tz(timedelta(hours=5, minutes=30))
    article = Article(**_base_kwargs(publication_date=datetime(2025, 8, 1, 15, 0, tzinfo=ist)))
    assert article.publication_date.utcoffset() == timedelta(hours=5, minutes=30)
