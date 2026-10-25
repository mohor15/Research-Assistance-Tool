"""ExtractedArticle + RetrievedPage + SourceDefinition -> Article.

The one place that turns Phase 1's heterogeneous retrieval output into the
canonical, deterministic `Article`. Never invents missing metadata: a
missing publication date stays `None` rather than being guessed.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import List, Optional, Tuple

from research_assistant.evidence.hashing import hash_content
from research_assistant.evidence.ids import generate_article_id
from research_assistant.evidence.text import normalize_whitespace
from research_assistant.evidence.urls import canonicalize_url
from research_assistant.models.article import Article, DatePrecision
from research_assistant.retrieval.models import ExtractedArticle, RetrievedPage, SourceDefinition


def normalize_article(
    extracted: ExtractedArticle,
    *,
    source: SourceDefinition,
    retrieved_at: Optional[datetime] = None,
) -> Article:
    canonical_url = canonicalize_url(extracted.url, page_canonical_url=extracted.metadata.get("canonical_url"))
    clean_text = normalize_whitespace(extracted.text)
    publication_date, precision = _normalize_publication_date(
        extracted.published_at, precision_hint=extracted.published_at_precision,
    )

    return Article(
        id=generate_article_id(canonical_url),
        url=extracted.url,
        canonical_url=canonical_url,
        domain=source.domain,
        source_name=source.name,
        source_type=source.source_type,
        source_priority=source.priority,
        publisher=_normalize_scalar(extracted.publisher) or source.name,
        title=_normalize_scalar(extracted.title),
        author=_normalize_scalar(extracted.author),
        publication_date=publication_date,
        publication_date_precision=precision,
        retrieved_at=retrieved_at or datetime.now(timezone.utc),
        raw_text=extracted.text,
        clean_text=clean_text,
        content_hash=hash_content(clean_text),
        extraction_method=extracted.extraction_method,
        extraction_status=extracted.status,
        metadata=dict(extracted.metadata),
    )


def normalize_from_page(extracted: ExtractedArticle, *, source: SourceDefinition, page: RetrievedPage) -> Article:
    """Convenience wrapper for callers that still have the `RetrievedPage` on hand."""
    return normalize_article(extracted, source=source, retrieved_at=page.retrieved_at)


def normalize_many(items: List[Tuple[SourceDefinition, ExtractedArticle]]) -> List[Article]:
    return [normalize_article(extracted, source=source) for source, extracted in items]


def _normalize_scalar(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    collapsed = " ".join(value.split())
    return collapsed or None


def _normalize_publication_date(
    value: Optional[datetime], *, precision_hint: Optional[str] = None,
) -> Tuple[Optional[datetime], DatePrecision]:
    if value is None:
        return None, DatePrecision.UNKNOWN
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    if precision_hint == "month":
        # Explicit, not inferred: a source-structural fallback (see
        # extractor.py's narrow BofA URL rule) only ever knows the month --
        # the day is a storage placeholder, never a real fact, and must
        # never be mistaken for the ordinary day-exact midnight heuristic
        # below.
        return value, DatePrecision.MONTH
    if precision_hint == "exact":
        # Explicit signal from the extractor (forensic fix, PR Newswire
        # CURRENT run `e818e08adc2e4dd1`): the RAW source date string
        # itself carried a time-of-day component -- authoritative, and
        # deliberately never re-derived from whether the parsed UTC instant
        # happens to land on a midnight boundary (an explicit
        # "...T00:00:00Z" source string is still a real, exact timestamp,
        # not a bare date that merely parsed to the same value).
        return value, DatePrecision.EXACT
    if precision_hint == "date_only":
        # Same explicit signal, the other direction: the extractor already
        # determined the raw source string carried no time component at
        # all, so there is no need to re-derive it from the midnight
        # heuristic below.
        return value, DatePrecision.DATE_ONLY
    # Heuristic fallback (unchanged, pre-existing): no explicit hint from
    # the extractor (every extraction path except the ones above) -- a
    # timestamp landing exactly at midnight UTC is indistinguishable, after
    # Phase 1's date parsing, from a source that only ever gave a bare
    # date. We cannot recover the original raw string here, so this is a
    # best-effort signal, not a certainty — see DatePrecision's docstring.
    if value.astimezone(timezone.utc).time() == time(0, 0, 0):
        return value, DatePrecision.DATE_ONLY
    return value, DatePrecision.EXACT
