"""Regression tests for `scripts/reprocess_research_item.py` — the
recovery path for ResearchItem/Event/Digest rows persisted by an OLDER
version of the `_resolve_dated_development` guardrail.

Seeds a tmp DB with exactly the STALE shape the real production row for
CURRENT run `7aab105954d38497` (Bits In Glass) had: ResearchItem with
`is_dated_development=False` and the old "not corroborated" override
rationale, an Event with `event_date=None`, a `pipeline_runs` row, and an
empty persisted digest (`digest_persistence_status=no_qualifying_events`)
-- then proves the script recovers all three, idempotently, without
touching any other run, entirely offline (FakeLLMClient — no Ollama, no
network).
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from research_assistant.agents.base import _generate_research_item_id
from research_assistant.config import get_config
from research_assistant.evidence.conversion import article_to_evidence
from research_assistant.evidence.normalizer import normalize_article
from research_assistant.events.builder import generate_event_id
from research_assistant.events.models import ClusterMetadata, Event, EventScoringBreakdown, SourceRef
from research_assistant.events.scoring import ConfidenceBreakdown, RelevanceBreakdown
from research_assistant.models.digest import Digest, DigestType
from research_assistant.models.enums import (
    AgentName,
    DigestPersistenceStatus,
    GroundingType,
    SourcePriority,
    SourceType,
)
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    ResearchItem,
    SourceQuality,
    TopicCategory,
)
from research_assistant.pipeline.runner import PipelineRunResult, RunStatus
from research_assistant.retrieval.extractor import extract_article
from research_assistant.retrieval.models import SourceCategory, SourceDefinition
from research_assistant.storage.database import Database
from research_assistant.storage.persistence_service import PersistenceService

from tests.agents.conftest import FakeLLMClient
from tests.retrieval.test_prnewswire_bits_in_glass_and_iquest_q1 import (
    _BITS_IN_GLASS_HTML,
    _BITS_IN_GLASS_URL,
)

_SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "reprocess_research_item.py"
_spec = importlib.util.spec_from_file_location("reprocess_research_item", _SCRIPT_PATH)
reprocess_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _spec.loader.exec_module(reprocess_module) or reprocess_module  # type: ignore

_RUN_ID = "7aab105954d38497"  # same run id as the real forensic case, fixture-only
_PERIOD_START = datetime(2026, 10, 8, 4, 34, tzinfo=timezone.utc)
_PERIOD_END = datetime(2026, 10, 9, 4, 34, tzinfo=timezone.utc)
_RESEARCH_ITEM_ID = "dc6927a203709f02"[:16]  # deterministic id shape, overwritten below with the real one

_STALE_RATIONALE = (
    "Deterministic override: the evidence text contains no explicit "
    "announcement/release language (e.g. 'today announced', 'has "
    "launched'), so is_dated_development=true was not corroborated and "
    "is not trusted on the model's judgment alone."
)


def _bits_in_glass_source() -> SourceDefinition:
    return SourceDefinition(
        name="PR Newswire",
        domain="prnewswire.com",
        urls=[],
        source_type=SourceType.PRESS_DISTRIBUTION,
        priority=SourcePriority.MEDIUM,
        category=SourceCategory.AI_PROVIDER,
    )


def _real_assessment():
    from research_assistant.agents.schemas import AgentAssessment

    return AgentAssessment(
        relevance=RelevanceDecision(
            label=RelevanceLabel.RELEVANT,
            reason="Describes a specific action (achieved Select partner status) with announcement language.",
        ),
        category=TopicCategory.ENTERPRISE_AI,
        summary="Bits In Glass achieved Select partner status in the Claude Partner Network.",
        is_dated_development=True,
        entities=[],
        people=[],
        products=[],
        technologies=[],
        geography=["Denver"],
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="x"),
    )


@pytest.fixture
def seeded_database(tmp_path: Path):
    """Builds a tmp DB holding the real Bits In Glass Article/Evidence plus
    a STALE ResearchItem/Event/pipeline_runs/digest exactly matching the
    shape the pre-fix guardrail would have persisted."""
    config = get_config()
    database = Database(tmp_path / "stale_bits_in_glass.db")
    database.initialize()
    service = PersistenceService(database)

    source = _bits_in_glass_source()
    extracted = extract_article(_BITS_IN_GLASS_HTML, _BITS_IN_GLASS_URL)
    article = normalize_article(extracted, source=source, retrieved_at=_PERIOD_END)
    evidence = article_to_evidence(article)

    item_id = _generate_research_item_id(AgentName.AI_PRODUCT, article.id)
    stale_item = ResearchItem(
        id=item_id,
        agent=AgentName.AI_PRODUCT,
        article_id=article.id,
        publisher=article.publisher,
        title=article.title,
        url=article.url,
        publication_date=article.publication_date,
        company=article.source_name,
        category=TopicCategory.ENTERPRISE_AI,
        is_dated_development=False,  # the stale, pre-fix narrowed value
        development_date_rationale=_STALE_RATIONALE,
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="x"),
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="x"),
        nomura_relevance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.HIGH, rationale="x"),
        evidence=[evidence.evidence_id],
        source_quality=SourceQuality(source_type=article.source_type, source_priority=article.source_priority),
        grounding_type=GroundingType.CONTENT,
        run_id=_RUN_ID,
    )

    event_id = generate_event_id([stale_item.id])
    stale_event = Event(
        id=event_id,
        canonical_title=stale_item.title,
        event_type=None,
        primary_company=stale_item.company,
        related_companies=[],
        event_date=None,  # the stale, pre-fix consequence of is_dated_development=False
        description=stale_item.summary,
        research_items=[stale_item.id],
        articles=[article.id],
        primary_source=SourceRef(
            research_item_id=stale_item.id, article_id=article.id, url=stale_item.url,
            publisher=stale_item.publisher, source_type=article.source_type, source_priority=article.source_priority,
        ),
        secondary_sources=[],
        relevance_score=0.7375,
        confidence_score=0.35,
        scoring_breakdown=EventScoringBreakdown(
            relevance=RelevanceBreakdown(
                ai_relevance=1.0, financial_services_relevance=0.75, strategic_significance=0.75,
                recency=0.0, novelty=1.0, weighted_total=0.7375,
            ),
            confidence=ConfidenceBreakdown(
                primary_source_confirmation=0.3, multiple_source_corroboration=0.3,
                publication_date_certainty=0.0, entity_extraction_confidence=1.0,
                event_classification_confidence=0.5, content_completeness=0.4, weighted_total=0.35,
            ),
        ),
        cluster_metadata=ClusterMetadata(member_count=1, matches=[], algorithm_version="v1"),
        grounding_type=GroundingType.CONTENT,
        run_id=_RUN_ID,
    )

    empty_digest = Digest(
        id="stale0000000001", digest_type=DigestType.CURRENT, generated_at=_PERIOD_END,
        period_start=_PERIOD_START, period_end=_PERIOD_END, events=[], themes=[],
    )

    service.persist_pipeline_result(
        articles=[article], evidence=[evidence], research_items=[stale_item], events=[stale_event],
        digest=empty_digest,
    )
    service.persist_run(
        PipelineRunResult(
            run_id=_RUN_ID, digest_type=DigestType.CURRENT, started_at=_PERIOD_END, finished_at=_PERIOD_END,
            status=RunStatus.PARTIAL_SUCCESS, period_start=_PERIOD_START, period_end=_PERIOD_END,
            digest_attempted=True, digest_persisted=True,
            digest_persistence_status=DigestPersistenceStatus.NO_QUALIFYING_EVENTS, digest_id=empty_digest.id,
        )
    )

    return database, stale_item.id, event_id, empty_digest.id


def test_stale_research_item_has_is_dated_development_false(seeded_database):
    database, item_id, _event_id, _digest_id = seeded_database
    conn = database.connect()
    try:
        row = conn.execute("SELECT is_dated_development FROM research_items WHERE id = ?", (item_id,)).fetchone()
    finally:
        conn.close()
    assert bool(row[0]) is False  # the fixture faithfully reproduces the stale shape


def test_dry_run_does_not_mutate_anything(seeded_database):
    database, item_id, _event_id, _digest_id = seeded_database
    fake_llm = FakeLLMClient([_real_assessment()])

    audit = reprocess_module.reprocess(
        database.path, item_id, commit=False, llm_client=fake_llm,
    )
    assert audit["before"]["is_dated_development"] is False
    assert audit["after"]["is_dated_development"] is True
    assert audit["event_after"]["event_date"] is not None
    assert audit["committed"] is False

    conn = database.connect()
    try:
        row = conn.execute("SELECT is_dated_development FROM research_items WHERE id = ?", (item_id,)).fetchone()
    finally:
        conn.close()
    assert bool(row[0]) is False  # rolled back -- nothing persisted


def test_commit_recovers_the_dated_development_and_event_date(seeded_database):
    database, item_id, event_id, _digest_id = seeded_database
    fake_llm = FakeLLMClient([_real_assessment()])

    audit = reprocess_module.reprocess(database.path, item_id, commit=True, llm_client=fake_llm)
    assert audit["after"]["is_dated_development"] is True
    assert audit["event_after"]["event_date"] == "2026-10-09T00:40:00+00:00"

    conn = database.connect()
    try:
        item_row = conn.execute(
            "SELECT is_dated_development, run_id FROM research_items WHERE id = ?", (item_id,)
        ).fetchone()
        event_row = conn.execute("SELECT event_date, run_id FROM events WHERE id = ?", (event_id,)).fetchone()
        item_count = conn.execute("SELECT COUNT(*) FROM research_items").fetchone()[0]
        event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    finally:
        conn.close()

    assert bool(item_row[0]) is True
    assert item_row[1] == _RUN_ID  # run ownership preserved, never reassigned
    assert event_row[0] is not None
    assert event_row[1] == _RUN_ID
    assert item_count == 1  # no duplicate ResearchItem row
    assert event_count == 1  # no duplicate Event row


def test_reprocessing_twice_is_idempotent(seeded_database):
    database, item_id, event_id, _digest_id = seeded_database
    fake_llm_1 = FakeLLMClient([_real_assessment()])
    fake_llm_2 = FakeLLMClient([_real_assessment()])

    reprocess_module.reprocess(database.path, item_id, commit=True, llm_client=fake_llm_1)
    audit_2 = reprocess_module.reprocess(database.path, item_id, commit=True, llm_client=fake_llm_2)

    assert audit_2["before"]["is_dated_development"] is True  # already corrected by the first run
    assert audit_2["after"]["is_dated_development"] is True

    conn = database.connect()
    try:
        item_count = conn.execute("SELECT COUNT(*) FROM research_items WHERE id = ?", (item_id,)).fetchone()[0]
        event_count = conn.execute("SELECT COUNT(*) FROM events WHERE id = ?", (event_id,)).fetchone()[0]
    finally:
        conn.close()
    assert item_count == 1
    assert event_count == 1


def test_rebuild_digest_repoints_the_run_at_a_persisted_non_empty_digest(seeded_database):
    database, item_id, _event_id, old_digest_id = seeded_database
    fake_llm = FakeLLMClient([_real_assessment()])

    audit = reprocess_module.reprocess(
        database.path, item_id, commit=True, rebuild_digest=True, llm_client=fake_llm,
    )
    assert audit["digest_rebuilt"] is True
    assert audit["digest"]["new_digest_persistence_status"] == "persisted"
    assert audit["digest"]["new_digest_event_count"] == 1
    assert audit["digest"]["new_digest_id"] != old_digest_id

    conn = database.connect()
    try:
        run_row = conn.execute(
            "SELECT digest_id, digest_persistence_status FROM pipeline_runs WHERE run_id = ?", (_RUN_ID,)
        ).fetchone()
        new_digest_events = conn.execute(
            "SELECT event_id FROM digest_events WHERE digest_id = ?", (run_row[0],)
        ).fetchall()
        old_digest_still_exists = conn.execute(
            "SELECT 1 FROM digests WHERE id = ?", (old_digest_id,)
        ).fetchone()
    finally:
        conn.close()

    assert run_row[1] == "persisted"
    assert len(new_digest_events) == 1  # Bits In Glass now qualifies for its own run's digest
    assert old_digest_still_exists is not None  # old (empty) digest kept as an audit trail, not deleted


def test_reprocessing_does_not_touch_an_unrelated_run(seeded_database):
    """A second, unrelated run's ResearchItem/Event must never be touched
    by reprocessing a different run's item -- run ownership isolation."""
    database, item_id, _event_id, _digest_id = seeded_database
    service = PersistenceService(database)

    from research_assistant.retrieval.models import ExtractedArticle

    other_extracted = ExtractedArticle(
        url="https://example.com/other", title="Unrelated other-run development", text="Unrelated content.",
    )
    other_article = normalize_article(other_extracted, source=_bits_in_glass_source(), retrieved_at=_PERIOD_END)

    other_run_id = "deadbeefcafef00d"
    other_item = ResearchItem(
        id="other0000000001",
        agent=AgentName.AI_PRODUCT,
        article_id=other_article.id,
        publisher="Other Publisher",
        title="Unrelated other-run development",
        url="https://example.com/other",
        publication_date=None,
        company="Other Co",
        category=TopicCategory.ENTERPRISE_AI,
        is_dated_development=False,
        development_date_rationale=_STALE_RATIONALE,
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="x"),
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
        nomura_relevance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
        evidence=[],
        source_quality=SourceQuality(source_type=SourceType.SECONDARY, source_priority=SourcePriority.LOW),
        grounding_type=GroundingType.LINK,
        run_id=other_run_id,
    )
    conn = database.connect()
    try:
        service.articles.upsert(conn, other_article)
        service.research_items.upsert(conn, other_item)
        conn.commit()
    finally:
        conn.close()

    fake_llm = FakeLLMClient([_real_assessment()])
    reprocess_module.reprocess(database.path, item_id, commit=True, llm_client=fake_llm)

    conn = database.connect()
    try:
        row = conn.execute(
            "SELECT is_dated_development, run_id FROM research_items WHERE id = ?", (other_item.id,)
        ).fetchone()
    finally:
        conn.close()
    assert bool(row[0]) is False  # untouched
    assert row[1] == other_run_id  # untouched


# ---------------------------------------------------------------------------
# --audit (read-only stale-record discovery, generalized beyond Bits In Glass)
# ---------------------------------------------------------------------------


def test_audit_flags_the_seeded_stale_record(seeded_database):
    """The fixture's stale ResearchItem (is_dated_development=False, the
    real override rationale, real evidence text the CURRENT pattern set
    now corroborates) must be found by a read-only audit scan -- the exact
    general shape the Bits In Glass and JumpCloud production cases shared,
    without naming either company anywhere in the check itself."""
    database, item_id, _event_id, _digest_id = seeded_database

    findings = reprocess_module.audit_stale_records(database.path)

    assert any(f["research_item_id"] == item_id for f in findings)

    conn = database.connect()
    try:
        row = conn.execute("SELECT is_dated_development FROM research_items WHERE id = ?", (item_id,)).fetchone()
    finally:
        conn.close()
    assert bool(row[0]) is False  # read-only -- the audit itself never mutates anything


def test_audit_does_not_flag_a_model_only_false_with_coincidental_text(seeded_database):
    """Negative control (real production false-positive this check must
    avoid): when `development_date_rationale` is None, the guardrail never
    fired -- the MODEL's own judgment was already False, and
    `_resolve_dated_development` returns before even reading the evidence
    text in that branch. A text-pattern match against such an item's
    evidence proves nothing about staleness and must never be flagged,
    even when the body happens to contain unrelated announcement-shaped
    wording (e.g. a quoted press release inside a retrospective feature)."""
    database, _item_id, _event_id, _digest_id = seeded_database
    service = PersistenceService(database)
    config = get_config()

    extracted = extract_article(
        "<html><head><title>x</title></head><body><article>"
        "<h1>A retrospective feature</h1>"
        "<p>Years ago the company today announced a product that reshaped the market, "
        "and commentators are still discussing its long-term impact.</p>"
        "</article></body></html>",
        "https://example.com/retrospective-feature",
    )
    source = _bits_in_glass_source()
    article = normalize_article(extracted, source=source, retrieved_at=_PERIOD_END)
    evidence = article_to_evidence(article)
    model_false_item_id = _generate_research_item_id(AgentName.AI_PRODUCT, article.id)
    model_false_item = ResearchItem(
        id=model_false_item_id,
        agent=AgentName.AI_PRODUCT,
        article_id=article.id,
        publisher=article.publisher,
        title=article.title,
        url=article.url,
        publication_date=article.publication_date,
        company=article.source_name,
        category=TopicCategory.ENTERPRISE_AI,
        is_dated_development=False,
        development_date_rationale=None,  # the model's OWN judgment -- guardrail never fired
        relevance=RelevanceDecision(label=RelevanceLabel.RELEVANT, reason="x"),
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
        nomura_relevance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
        evidence=[evidence.evidence_id],
        source_quality=SourceQuality(source_type=article.source_type, source_priority=article.source_priority),
        grounding_type=GroundingType.CONTENT,
        run_id="unrelated0000001",
    )
    conn = database.connect()
    try:
        service.articles.upsert(conn, article)
        service.evidence.upsert(conn, evidence)
        service.research_items.upsert(conn, model_false_item)
        conn.commit()
    finally:
        conn.close()

    findings = reprocess_module.audit_stale_records(database.path)

    assert not any(f["research_item_id"] == model_false_item_id for f in findings)
