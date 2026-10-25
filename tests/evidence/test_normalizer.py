"""ExtractedArticle -> Article normalization: metadata, dates, publisher, URL, content, hash, deterministic id."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.evidence.normalizer import normalize_article, normalize_from_page
from research_assistant.models.article import DatePrecision
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.models import (
    ExtractedArticle,
    ExtractionMethod,
    ExtractionStatus,
    RetrievalMethod,
    RetrievedPage,
    SourceCategory,
    SourceDefinition,
)


def _source(**overrides) -> SourceDefinition:
    kwargs = dict(
        name="Example Bank",
        domain="example.com",
        urls=["https://example.com/newsroom"],
        source_type=SourceType.PRIMARY,
        priority=SourcePriority.HIGH,
        category=SourceCategory.PEER,
    )
    kwargs.update(overrides)
    return SourceDefinition(**kwargs)


def test_publisher_falls_back_to_source_name_when_extraction_found_none():
    extracted = ExtractedArticle(
        url="https://example.com/a", text="Some article body text that is long enough to matter here.",
        extraction_method=ExtractionMethod.BS4_FALLBACK, status=ExtractionStatus.SUCCESS,
    )
    article = normalize_article(extracted, source=_source())
    assert article.publisher == "Example Bank"


def test_extracted_publisher_is_preserved_when_present():
    extracted = ExtractedArticle(
        url="https://example.com/a", publisher="Example Bank Newsroom",
        text="Body text.", extraction_method=ExtractionMethod.BS4_FALLBACK, status=ExtractionStatus.SUCCESS,
    )
    article = normalize_article(extracted, source=_source())
    assert article.publisher == "Example Bank Newsroom"


# ---------------------------------------------------------------------------
# Phase 14 date-precision fix: `ExtractedArticle.published_at_precision`
# (currently only ever set to "month" by the narrow BofA URL fallback in
# extractor.py) must be propagated into `Article.publication_date_precision`
# as `DatePrecision.MONTH`, distinct from the pre-existing day-exact/
# midnight-heuristic behavior.
# ---------------------------------------------------------------------------


def test_month_precision_hint_is_propagated_as_month():
    extracted = ExtractedArticle(
        url="https://example.com/a", text="Body text long enough.",
        published_at=datetime(2026, 9, 1, tzinfo=timezone.utc), published_at_precision="month",
    )
    article = normalize_article(extracted, source=_source())
    assert article.publication_date == datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert article.publication_date_precision == DatePrecision.MONTH


def test_no_precision_hint_falls_back_to_the_existing_midnight_heuristic():
    """Unaffected, pre-existing behavior: no hint at all (the default for
    every extraction path except the new BofA fallback) still uses the
    original day-exact-vs-midnight heuristic."""
    extracted = ExtractedArticle(
        url="https://example.com/a", text="Body text long enough.",
        published_at=datetime(2026, 9, 29, 14, 30, tzinfo=timezone.utc),
    )
    article = normalize_article(extracted, source=_source())
    assert article.publication_date_precision == DatePrecision.EXACT


def test_no_precision_hint_midnight_timestamp_is_still_date_only():
    extracted = ExtractedArticle(
        url="https://example.com/a", text="Body text long enough.",
        published_at=datetime(2026, 9, 29, 0, 0, tzinfo=timezone.utc),
    )
    article = normalize_article(extracted, source=_source())
    assert article.publication_date_precision == DatePrecision.DATE_ONLY


def test_url_is_canonicalized():
    extracted = ExtractedArticle(url="https://example.com/a/?utm_source=x#y", text="body")
    article = normalize_article(extracted, source=_source())
    assert article.canonical_url == "https://example.com/a"
    assert article.url == "https://example.com/a/?utm_source=x#y"  # original preserved separately


def test_content_is_cleaned_and_hashed():
    extracted = ExtractedArticle(url="https://example.com/a", text="Some   text   with  extra   spaces.")
    article = normalize_article(extracted, source=_source())
    assert article.clean_text == "Some text with extra spaces."
    assert article.raw_text == "Some   text   with  extra   spaces."
    assert article.content_hash is not None


def test_missing_content_produces_no_hash():
    extracted = ExtractedArticle(url="https://example.com/a", text=None, status=ExtractionStatus.FAILED)
    article = normalize_article(extracted, source=_source())
    assert article.clean_text is None
    assert article.content_hash is None


def test_missing_publication_date_is_preserved_as_missing_not_guessed():
    extracted = ExtractedArticle(url="https://example.com/a", text="body", published_at=None)
    article = normalize_article(extracted, source=_source())
    assert article.publication_date is None
    assert article.publication_date_precision == DatePrecision.UNKNOWN


def test_exact_timestamp_publication_date_is_marked_exact():
    extracted = ExtractedArticle(
        url="https://example.com/a", text="body",
        published_at=datetime(2025, 9, 1, 14, 30, tzinfo=timezone.utc),
    )
    article = normalize_article(extracted, source=_source())
    assert article.publication_date_precision == DatePrecision.EXACT


def test_midnight_publication_date_is_marked_date_only_heuristically():
    extracted = ExtractedArticle(
        url="https://example.com/a", text="body",
        published_at=datetime(2025, 9, 1, 0, 0, tzinfo=timezone.utc),
    )
    article = normalize_article(extracted, source=_source())
    assert article.publication_date_precision == DatePrecision.DATE_ONLY


def test_naive_published_at_is_coerced_to_utc_not_rejected():
    extracted = ExtractedArticle(url="https://example.com/a", text="body")
    extracted = extracted.model_copy(update={"published_at": datetime(2025, 9, 1, 10, 0)})  # naive
    article = normalize_article(extracted, source=_source())
    assert article.publication_date.tzinfo is not None


def test_deterministic_id_is_stable_for_same_canonical_url():
    extracted_a = ExtractedArticle(url="https://example.com/a?utm_source=twitter", text="body")
    extracted_b = ExtractedArticle(url="https://example.com/a/", text="different body text entirely")
    article_a = normalize_article(extracted_a, source=_source())
    article_b = normalize_article(extracted_b, source=_source())
    assert article_a.id == article_b.id  # same canonical_url -> same id, regardless of content


def test_different_canonical_urls_produce_different_ids():
    a = normalize_article(ExtractedArticle(url="https://example.com/a", text="body"), source=_source())
    b = normalize_article(ExtractedArticle(url="https://example.com/b", text="body"), source=_source())
    assert a.id != b.id


def test_title_and_author_whitespace_is_collapsed():
    extracted = ExtractedArticle(
        url="https://example.com/a", text="body",
        title="  A   Title  With   Spaces ", author="  Jane   Doe ",
    )
    article = normalize_article(extracted, source=_source())
    assert article.title == "A Title With Spaces"
    assert article.author == "Jane Doe"


def test_source_provenance_fields_are_copied_from_source_definition():
    extracted = ExtractedArticle(url="https://example.com/a", text="body")
    article = normalize_article(extracted, source=_source(name="Example Bank", domain="example.com"))
    assert article.source_name == "Example Bank"
    assert article.domain == "example.com"
    assert article.source_type == SourceType.PRIMARY
    assert article.source_priority == SourcePriority.HIGH


def test_extraction_method_and_status_are_carried_through():
    extracted = ExtractedArticle(
        url="https://example.com/a", text="body",
        extraction_method=ExtractionMethod.TRAFILATURA, status=ExtractionStatus.PARTIAL,
    )
    article = normalize_article(extracted, source=_source())
    assert article.extraction_method == ExtractionMethod.TRAFILATURA
    assert article.extraction_status == ExtractionStatus.PARTIAL


def test_normalize_from_page_uses_the_pages_retrieved_at_timestamp():
    extracted = ExtractedArticle(url="https://example.com/a", text="body")
    page = RetrievedPage(
        url="https://example.com/a", final_url="https://example.com/a",
        retrieved_at=datetime(2025, 1, 1, tzinfo=timezone.utc), method=RetrievalMethod.HTTP, success=True,
    )
    article = normalize_from_page(extracted, source=_source(), page=page)
    assert article.retrieved_at == datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_extracted_article_metadata_is_preserved_on_the_article():
    extracted = ExtractedArticle(url="https://example.com/a", text=None, status=ExtractionStatus.FAILED)
    extracted = extracted.model_copy(update={"metadata": {"reason": "empty_html"}})
    article = normalize_article(extracted, source=_source())
    assert article.metadata == {"reason": "empty_html"}
