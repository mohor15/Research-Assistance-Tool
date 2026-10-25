"""Phase-9-hotfix evidence-stage diagnostics: deterministic, testable
counts and per-article log detail explaining exactly why an in-period
article did or didn't produce Evidence -- added purely as observability,
the evidence gate itself (`article_to_evidence`) is unchanged.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace

from research_assistant.config import get_config
from research_assistant.evidence.service import NormalizationService
from research_assistant.models.article import Article
from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.pipeline.nodes import make_evidence_node
from research_assistant.pipeline.state import PipelineState
from research_assistant.retrieval.models import ExtractionMethod

CONFIG = get_config()


def _article(**overrides) -> Article:
    kwargs = dict(
        id=f"id-{overrides.get('url', 'x')}", url="https://jpmorganchase.com/a",
        canonical_url="https://jpmorganchase.com/a", domain="jpmorganchase.com",
        source_name="JPMorgan Chase", source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH,
        retrieved_at=datetime(2025, 9, 1, tzinfo=timezone.utc), title="A Title",
        publication_date=datetime(2025, 9, 1, tzinfo=timezone.utc),
        clean_text="Real, usable article body text.", content_hash="deadbeef",
        extraction_method=ExtractionMethod.BS4_FALLBACK,
    )
    kwargs.update(overrides)
    return Article(**kwargs)


def _services():
    return SimpleNamespace(config=CONFIG, normalization_service=NormalizationService())


def _state(*articles: Article) -> PipelineState:
    return PipelineState(articles=list(articles))


def test_5_diagnostics_count_each_rejection_reason():
    valid = _article(url="v", id="v")
    missing_clean_text = _article(url="a", id="a", clean_text=None, content_hash="deadbeef")
    missing_content_hash = _article(url="b", id="b", content_hash=None)
    missing_both = _article(url="c", id="c", clean_text=None, content_hash=None)

    node = make_evidence_node(_services())
    result = node(_state(valid, missing_clean_text, missing_content_hash, missing_both))

    assert result["metrics"]["evidence_created"] == 1
    assert result["metrics"]["evidence_rejected_missing_clean_text"] == 1
    assert result["metrics"]["evidence_rejected_missing_content_hash"] == 1
    assert result["metrics"]["evidence_rejected_missing_both"] == 1
    assert result["metrics"]["evidence_rejected_other"] == 0


def test_6_source_name_and_source_type_survive_into_the_rejection_log(caplog):
    rejected = _article(
        url="r", id="r", clean_text=None, content_hash=None,
        source_name="Deutsche Bank", source_type=SourceType.PRIMARY,
    )
    node = make_evidence_node(_services())

    with caplog.at_level(logging.INFO, logger="research_assistant.pipeline.nodes"):
        node(_state(rejected))

    rejection_lines = [r.message for r in caplog.records if "evidence: rejected" in r.message]
    assert len(rejection_lines) == 1
    assert "source_name=Deutsche Bank" in rejection_lines[0]
    assert "source_type=primary" in rejection_lines[0]
    assert "reason=missing_clean_text_and_content_hash" in rejection_lines[0]
    assert "article_id=r" in rejection_lines[0]


def test_6b_by_source_category_breakdown_is_present_for_created_evidence():
    peer = _article(url="p", id="p", source_name="JPMorgan Chase", source_type=SourceType.PRIMARY)
    ai_provider = _article(url="q", id="q", source_name="OpenAI", source_type=SourceType.PRIMARY)

    node = make_evidence_node(_services())
    result = node(_state(peer, ai_provider))

    assert result["metrics"]["evidence_by_category_peers"] == 1
    assert result["metrics"]["evidence_by_category_ai_providers"] == 1


def test_7_no_article_body_text_is_emitted_into_logs(caplog):
    secret_marker = "SECRET-ARTICLE-BODY-MARKER-XYZ"
    accepted = _article(url="accepted", id="accepted", clean_text=secret_marker, content_hash="deadbeef")
    rejected = _article(url="rejected", id="rejected", clean_text=None, content_hash=None)
    # A rejected article can still carry raw_text (pre-whitespace-cleanup
    # content) even though clean_text ended up empty/None -- confirm that
    # never leaks into logs either.
    rejected = rejected.model_copy(update={"raw_text": secret_marker})

    node = make_evidence_node(_services())
    with caplog.at_level(logging.INFO, logger="research_assistant.pipeline.nodes"):
        node(_state(accepted, rejected))

    full_log_text = "\n".join(r.message for r in caplog.records)
    assert secret_marker not in full_log_text


def test_diagnostics_list_gets_one_summary_entry_when_rejections_occur():
    rejected = _article(url="d", id="d", clean_text=None, content_hash=None)
    node = make_evidence_node(_services())

    result = node(_state(rejected))

    evidence_diagnostics = [d for d in result["diagnostics"] if d.stage == "evidence"]
    assert len(evidence_diagnostics) == 1
    assert "rejected_missing_both=1" in evidence_diagnostics[0].message


def test_no_diagnostics_entry_when_everything_produces_evidence():
    valid = _article(url="ok", id="ok")
    node = make_evidence_node(_services())

    result = node(_state(valid))

    assert not [d for d in result["diagnostics"] if d.stage == "evidence"]


def test_8_existing_evidence_behavior_unchanged_valid_articles_still_produce_evidence():
    valid_one = _article(url="e1", id="e1")
    valid_two = _article(url="e2", id="e2", clean_text="A different body of text entirely.", content_hash="beefdead")

    node = make_evidence_node(_services())
    result = node(_state(valid_one, valid_two))

    assert len(result["evidence"]) == 2
    assert result["metrics"]["evidence_created"] == 2
