"""Phase 1 fix: entity-centric monitoring pages (Peers/AI Providers/
Analysts) must always show every CONFIGURED entity, with an explicit,
distinguishable status, even when that entity has zero qualifying
developments -- never silently empty, never fabricated.
"""

from __future__ import annotations

from datetime import timedelta

from research_assistant.config import get_config
from research_assistant.evidence import article_to_evidence
from research_assistant.models.enums import AgentName, EventType
from research_assistant.pipeline.runner import PipelineRunResult, RunStatus
from research_assistant.pipeline.state import PipelineDiagnostic
from research_assistant.storage.persistence_service import PersistenceService
from research_assistant.ui import dashboard_data as dd

from tests.storage.conftest import D, make_article, make_event, make_research_item

CONFIG = get_config()


def _seed_relevant_development(service: PersistenceService, *, company: str, event_id="ev1"):
    article = make_article("art-" + event_id, company=company, content_hash="hash-" + event_id, publication_date=D)
    evidence = article_to_evidence(article)
    item = make_research_item(
        "ri-" + event_id, article_id=article.id, company=company, agent=AgentName.PEER,
        event_type=EventType.PRODUCT_LAUNCH, evidence_ids=[evidence.evidence_id], publication_date=D,
    )
    event = make_event(event_id, research_item_ids=[item.id], article_ids=[article.id], company=company, event_date=D)
    service.persist_pipeline_result(articles=[article], evidence=[evidence], research_items=[item], events=[event])
    return event


def _mark_a_run_as_completed(service: PersistenceService, *, diagnostics=None):
    """Entity statuses other than NOT_YET_ANALYZED require at least one
    completed pipeline run to exist (see `has_any_pipeline_run_completed`)."""
    service.persist_run(
        PipelineRunResult(
            run_id="run1", digest_type="morning", started_at=D, finished_at=D,
            status=RunStatus.SUCCESS, diagnostics=diagnostics or [], counts={},
        )
    )


# ---------------------------------------------------------------------------
# A. Entity registry visibility
# ---------------------------------------------------------------------------


def test_all_eight_peers_appear_even_with_zero_events(persistence_service: PersistenceService):
    statuses = dd.get_peer_entities(persistence_service, CONFIG)

    assert len(statuses) == 9
    assert {s.name for s in statuses} == {s.name for s in CONFIG.sources.peers}
    assert all(s.event_count == 0 for s in statuses)


def test_all_eight_ai_providers_appear_even_with_zero_events(persistence_service: PersistenceService):
    statuses = dd.get_ai_provider_entities(persistence_service, CONFIG)

    assert len(statuses) == 8
    assert {s.name for s in statuses} == {s.name for s in CONFIG.sources.ai_providers}


def test_all_five_analysts_appear_even_with_zero_events(persistence_service: PersistenceService):
    statuses = dd.get_analyst_entities(persistence_service, CONFIG)

    assert len(statuses) == 5
    assert {s.name for s in statuses} == {s.name for s in CONFIG.sources.analysts}


def test_entity_with_a_real_event_is_still_listed_alongside_entities_with_none(persistence_service: PersistenceService):
    _mark_a_run_as_completed(persistence_service)
    _seed_relevant_development(persistence_service, company="JPMorgan Chase")

    statuses = dd.get_peer_entities(persistence_service, CONFIG, since=D - timedelta(hours=1), until=D + timedelta(hours=1))

    assert len(statuses) == 9  # every peer still appears
    jpmorgan = next(s for s in statuses if s.name == "JPMorgan Chase")
    morgan_stanley = next(s for s in statuses if s.name == "Morgan Stanley")
    assert jpmorgan.status == dd.STATUS_RECENT_DEVELOPMENTS
    assert jpmorgan.event_count == 1
    assert morgan_stanley.status == dd.STATUS_NO_ARTICLES_RETRIEVED
    assert morgan_stanley.event_count == 0


# ---------------------------------------------------------------------------
# B. The five distinguishable statuses
# ---------------------------------------------------------------------------


def test_status_not_yet_analyzed_when_no_pipeline_run_has_ever_completed(persistence_service: PersistenceService):
    statuses = dd.get_peer_entities(persistence_service, CONFIG)

    assert all(s.status == dd.STATUS_NOT_YET_ANALYZED for s in statuses)


def test_status_source_retrieval_failed_is_distinguished_from_no_articles(persistence_service: PersistenceService):
    _mark_a_run_as_completed(
        persistence_service,
        diagnostics=[PipelineDiagnostic(stage="retrieve", message="Morgan Stanley: connection_error")],
    )

    statuses = dd.get_peer_entities(persistence_service, CONFIG)

    morgan_stanley = next(s for s in statuses if s.name == "Morgan Stanley")
    jpmorgan = next(s for s in statuses if s.name == "JPMorgan Chase")
    assert morgan_stanley.status == dd.STATUS_RETRIEVAL_FAILED
    assert morgan_stanley.source_failure_reason == "connection_error"
    assert jpmorgan.status == dd.STATUS_NO_ARTICLES_RETRIEVED  # unaffected by another source's failure
    assert jpmorgan.source_failure_reason is None


def test_status_no_articles_retrieved(persistence_service: PersistenceService):
    _mark_a_run_as_completed(persistence_service)

    statuses = dd.get_peer_entities(persistence_service, CONFIG)

    assert all(s.status == dd.STATUS_NO_ARTICLES_RETRIEVED for s in statuses)
    assert all(s.articles_retrieved == 0 for s in statuses)


def test_status_articles_retrieved_but_none_passed_relevance_filtering(persistence_service: PersistenceService):
    _mark_a_run_as_completed(persistence_service)
    article = make_article("art1", company="JPMorgan Chase", publication_date=D)
    conn = persistence_service.database.connect()
    persistence_service.articles.upsert(conn, article)
    conn.commit()
    conn.close()

    statuses = dd.get_peer_entities(persistence_service, CONFIG, since=D - timedelta(hours=1), until=D + timedelta(hours=1))

    jpmorgan = next(s for s in statuses if s.name == "JPMorgan Chase")
    assert jpmorgan.status == dd.STATUS_RETRIEVED_NOT_RELEVANT
    assert jpmorgan.articles_retrieved == 1
    assert jpmorgan.relevant_items == 0


def test_status_recent_developments_found_includes_latest_development_and_source_link(persistence_service: PersistenceService):
    _mark_a_run_as_completed(persistence_service)
    event = _seed_relevant_development(persistence_service, company="JPMorgan Chase")

    statuses = dd.get_peer_entities(persistence_service, CONFIG, since=D - timedelta(hours=1), until=D + timedelta(hours=1))

    jpmorgan = next(s for s in statuses if s.name == "JPMorgan Chase")
    assert jpmorgan.status == dd.STATUS_RECENT_DEVELOPMENTS
    assert jpmorgan.latest_development is not None
    assert jpmorgan.latest_development.id == event.id
    assert jpmorgan.latest_article_url == event.primary_source.url
    assert jpmorgan.latest_timestamp == D


# ---------------------------------------------------------------------------
# G. No fabrication
# ---------------------------------------------------------------------------


def test_zero_qualifying_articles_yields_an_explicit_zero_state_never_a_fabricated_event(persistence_service: PersistenceService):
    _mark_a_run_as_completed(persistence_service)

    statuses = dd.get_peer_entities(persistence_service, CONFIG)

    for status in statuses:
        assert status.event_count == 0
        assert status.latest_development is None
        assert status.status != dd.STATUS_RECENT_DEVELOPMENTS


def test_official_source_url_is_populated_from_the_real_registry_never_fabricated(persistence_service: PersistenceService):
    statuses = dd.get_peer_entities(persistence_service, CONFIG)

    jpmorgan = next(s for s in statuses if s.name == "JPMorgan Chase")
    configured_url = next(s.urls[0] for s in CONFIG.sources.peers if s.name == "JPMorgan Chase")
    assert jpmorgan.official_url == configured_url


# ---------------------------------------------------------------------------
# H. Article visibility (Phase 13D) -- the full surviving article set must
# be inspectable, not collapsed down to "the latest one".
# ---------------------------------------------------------------------------


def test_get_entity_articles_exposes_every_article_not_only_the_latest(persistence_service: PersistenceService):
    event = _seed_relevant_development(persistence_service, company="JPMorgan Chase", event_id="ev1")

    # A second, older article for the same company that was retrieved and
    # analyzed (a ResearchItem exists) but never clustered into an event.
    older_article = make_article("art-ri-only", company="JPMorgan Chase", content_hash="hash-ri-only", publication_date=D - timedelta(days=1))
    older_evidence = article_to_evidence(older_article)
    older_item = make_research_item(
        "ri-only", article_id=older_article.id, company="JPMorgan Chase", agent=AgentName.PEER,
        evidence_ids=[older_evidence.evidence_id], publication_date=D - timedelta(days=1),
    )
    persistence_service.persist_pipeline_result(
        articles=[older_article], evidence=[older_evidence], research_items=[older_item],
    )

    # A third, oldest article that was retrieved but never became a
    # ResearchItem at all (irrelevant, or never reached the LLM).
    unanalyzed_article = make_article(
        "art-unanalyzed", company="JPMorgan Chase", content_hash="hash-unanalyzed", publication_date=D - timedelta(days=2),
    )
    persistence_service.persist_pipeline_result(articles=[unanalyzed_article])

    rows = dd.get_entity_articles(persistence_service, "JPMorgan Chase")

    assert len(rows) == 3  # never collapsed to just the latest
    by_id = {r.article_id: r for r in rows}
    assert by_id[event.primary_source.article_id].status == dd.ARTICLE_STATUS_EVENT_FORMED
    assert by_id[event.primary_source.article_id].event_relevance_score == event.relevance_score
    assert by_id["art-ri-only"].status == dd.ARTICLE_STATUS_RELEVANT_NO_EVENT
    assert by_id["art-ri-only"].event_relevance_score is None  # never fabricated
    assert by_id["art-unanalyzed"].status == dd.ARTICLE_STATUS_NOT_RELEVANT_OR_UNANALYZED
    assert by_id["art-unanalyzed"].relevance_label is None

    # newest first
    assert [r.article_id for r in rows] == [event.primary_source.article_id, "art-ri-only", "art-unanalyzed"]


def test_get_entity_articles_source_links_and_titles_are_always_present(persistence_service: PersistenceService):
    article = make_article("art-x", company="JPMorgan Chase", content_hash="hash-x", publication_date=D)
    persistence_service.persist_pipeline_result(articles=[article])

    rows = dd.get_entity_articles(persistence_service, "JPMorgan Chase")

    assert len(rows) == 1
    assert rows[0].url == article.url
    assert rows[0].title == article.title
