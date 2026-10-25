"""`evidence_rejection_reason()` -- the pure, explicit explanation of why
`article_to_evidence()` would reject an article. Additive only: does not
change `article_to_evidence`'s existing gate/behavior (see
test_evidence.py for that contract, unchanged).
"""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.evidence.conversion import article_to_evidence, evidence_rejection_reason
from research_assistant.models.article import Article
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.models import ExtractionMethod, ExtractionStatus


def _article(**overrides) -> Article:
    kwargs = dict(
        id="art-1", url="https://example.com/a", canonical_url="https://example.com/a",
        domain="example.com", source_name="Example Bank", source_type=SourceType.PRIMARY,
        source_priority=SourcePriority.HIGH, retrieved_at=datetime(2025, 9, 1, tzinfo=timezone.utc),
        title="A Title", clean_text="A full article body.", content_hash="deadbeef",
        extraction_method=ExtractionMethod.BS4_FALLBACK,
    )
    kwargs.update(overrides)
    return Article(**kwargs)


def test_1_valid_article_has_no_rejection_reason_and_produces_evidence():
    article = _article()
    assert evidence_rejection_reason(article) is None
    assert article_to_evidence(article) is not None


def test_2_missing_clean_text_is_rejected_with_explicit_reason():
    # content_hash present with no clean_text is a state that can't arise
    # from normalize_article() but is directly constructible -- exercises
    # the "missing_clean_text" branch in isolation from "missing both".
    article = _article(clean_text=None, content_hash="deadbeef")
    assert evidence_rejection_reason(article) == "missing_clean_text"
    assert article_to_evidence(article) is None


def test_3_missing_content_hash_is_rejected_with_explicit_reason():
    article = _article(content_hash=None)
    assert evidence_rejection_reason(article) == "missing_content_hash"
    assert article_to_evidence(article) is None


def test_4_missing_both_is_rejected_with_explicit_combined_reason():
    article = _article(clean_text=None, content_hash=None)
    assert evidence_rejection_reason(article) == "missing_clean_text_and_content_hash"
    assert article_to_evidence(article) is None


def test_8_existing_evidence_behavior_is_unchanged():
    """The pure gate article_to_evidence() itself was not touched."""
    valid = _article()
    assert article_to_evidence(valid) is not None
    rejected = _article(clean_text=None, content_hash=None)
    assert article_to_evidence(rejected) is None


def test_not_article_extraction_status_with_non_empty_text_still_passes_the_gate():
    """Retrieval-correctness review finding (not a bug fix): the evidence
    gate checks only presence of clean_text/content_hash, never
    `extraction_status`. An article the extractor itself judged as
    NOT_ARTICLE (e.g. a thin boilerplate/nav scrap below the "real
    article" character threshold) can still carry a short non-empty
    clean_text and therefore still produces Evidence -- "retrieval/
    extraction succeeded" is not the same thing as "this is a genuine
    article", and the current gate does not distinguish them. Documented
    here as existing, unchanged behavior; not altered."""
    article = _article(
        clean_text="Menu Contact Careers", content_hash="deadbeef",
        extraction_status=ExtractionStatus.NOT_ARTICLE,
    )
    assert evidence_rejection_reason(article) is None
    assert article_to_evidence(article) is not None


def test_thin_but_non_empty_clean_text_currently_passes_the_gate():
    """Documents current, unchanged behavior for the "valid title/URL/date
    but thin content" case raised in review: the gate is presence-only
    (`not clean_text`), not a minimum-length/quality check. A 6-character
    body with a fully valid title/URL/publication_date is NOT rejected --
    it produces Evidence with that thin content. This is not changed here
    (no quality gate was requested), only made explicit: a future,
    separately-decided minimum-content-length policy would be a real
    behavior change, not a bug fix."""
    thin = _article(title="Real Title", clean_text="Menu.", content_hash="deadbeef")
    assert evidence_rejection_reason(thin) is None
    evidence = article_to_evidence(thin)
    assert evidence is not None
    assert evidence.content == "Menu."
