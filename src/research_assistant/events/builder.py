"""Cluster -> Event (everything except the relevance/confidence scores,
which `events.scoring` fills in afterward — scoring an Event that doesn't
exist yet isn't meaningful, so construction happens in two steps).

Primary/secondary source selection is fully deterministic, using only the
existing `ResearchItem.source_quality` (itself copied from `Article`,
never LLM-decided): source_type rank, then source_priority, then earliest
publication date, then research_item id as a final tiebreak. Press
distribution sources rank below ordinary secondary sources, never above —
per config/sources.yaml's own classification, they are never treated as
independent journalism, let alone promoted to primary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Set

from research_assistant.events.clustering import Cluster
from research_assistant.events.models import ClusterMetadata, SourceRef
from research_assistant.models.enums import EntityType, EventType, GroundingType, SourcePriority, SourceType
from research_assistant.models.research_item import ResearchItem

_SOURCE_TYPE_RANK = {
    SourceType.PRIMARY: 3,
    SourceType.ANALYST: 2,
    SourceType.SECONDARY: 1,
    SourceType.PRESS_DISTRIBUTION: 0,
}
_PRIORITY_RANK = {SourcePriority.HIGH: 2, SourcePriority.MEDIUM: 1, SourcePriority.LOW: 0}
_ID_LENGTH = 16

# Sentinel so a missing publication_date always sorts last (least-preferred
# primary-source candidate) without needing another item's tzinfo to build it.
_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


@dataclass
class EventDraft:
    """Every Event field except the scores, which are filled in by events.scoring."""

    id: str
    canonical_title: str
    event_type: Optional[EventType]
    primary_company: str
    related_companies: List[str]
    event_date: Optional[datetime]
    description: Optional[str]
    research_items: List[str]
    articles: List[str]
    primary_source: SourceRef
    secondary_sources: List[SourceRef]
    cluster_metadata: ClusterMetadata
    grounding_type: GroundingType = GroundingType.CONTENT


def generate_event_id(research_item_ids: List[str]) -> str:
    """Deterministic: the same set of member research items always produces
    the same Event id, regardless of clustering/iteration order."""
    basis = ",".join(sorted(research_item_ids))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:_ID_LENGTH]


def build_event_draft(cluster: Cluster, *, excluded_terms: Optional[Set[str]] = None) -> EventDraft:
    """`excluded_terms` (optional, case-insensitive): a vocabulary of known
    non-company concepts/technologies/topics (see
    `flatten_non_company_vocabulary`) that must never be reported as a
    "related company" even if the LLM listed one under `entities` or
    `partnerships` -- e.g. "Agentic AI" is a technology, not an
    organization. Defaults to no filtering, for backward compatibility with
    callers that don't have a `TopicsRegistry` on hand (e.g. existing unit
    tests); `EventService` always supplies the real configured vocabulary."""
    items = cluster.items
    sorted_by_source_quality = sorted(items, key=_primary_sort_key)
    primary_item = sorted_by_source_quality[0]
    primary_source = _to_source_ref(primary_item)
    secondary_sources = [_to_source_ref(item) for item in sorted_by_source_quality[1:]]
    canonical_title = _select_canonical_title(items, primary_item)
    primary_company = _select_primary_company(items, primary_item, canonical_title)

    return EventDraft(
        id=generate_event_id([item.id for item in items]),
        canonical_title=canonical_title,
        event_type=_select_event_type(items),
        primary_company=primary_company,
        related_companies=_collect_related_companies(items, primary_company, excluded_terms=excluded_terms),
        event_date=_select_event_date(items),
        description=_select_description(items, primary_item),
        research_items=[item.id for item in items],
        # Order-preserving dedup: the same article can be routed to more
        # than one agent, so two distinct ResearchItems in this cluster may
        # share an article_id -- `event_articles` has PRIMARY KEY
        # (event_id, article_id), so a duplicate here would fail persistence.
        articles=list(dict.fromkeys(item.article_id for item in items)),
        primary_source=primary_source,
        secondary_sources=secondary_sources,
        cluster_metadata=cluster.metadata,
        grounding_type=_select_grounding_type(items),
    )


def _select_grounding_type(items: List[ResearchItem]) -> GroundingType:
    """CONTENT if any member actually has retrieved-article evidence behind
    it -- real content backing even part of a cluster means the Event as a
    whole is not purely metadata-only. Only when EVERY member is
    link-grounded does the Event itself become LINK."""
    if any(item.grounding_type == GroundingType.CONTENT for item in items):
        return GroundingType.CONTENT
    return GroundingType.LINK


def _primary_sort_key(item: ResearchItem):
    return (
        -_SOURCE_TYPE_RANK.get(item.source_quality.source_type, 0),
        -_PRIORITY_RANK.get(item.source_quality.source_priority, 0),
        item.publication_date or _FAR_FUTURE,
        item.id,
    )


def _to_source_ref(item: ResearchItem) -> SourceRef:
    return SourceRef(
        research_item_id=item.id,
        article_id=item.article_id,
        url=item.url,
        publisher=item.publisher,
        source_type=item.source_quality.source_type,
        source_priority=item.source_quality.source_priority,
    )


def _select_canonical_title(items: List[ResearchItem], primary_item: ResearchItem) -> str:
    if primary_item.title:
        return primary_item.title
    for item in sorted(items, key=lambda i: i.id):
        if item.title:
            return item.title
    return f"{primary_item.company} development"


def _select_primary_company(
    items: List[ResearchItem], primary_item: ResearchItem, canonical_title: str,
) -> str:
    """`ResearchItem.company` is copied from `Article.source_name` -- the
    SOURCE REGISTRY's own name (see `agents.base._build_research_item`),
    which is the real operating company only for a source that publishes
    its own content (e.g. "Cohere" for cohere.com). For a press-distribution
    WIRE SERVICE (PR Newswire, Business Wire, GlobeNewswire, ...), that name
    is the DISTRIBUTOR, never the company the release is actually about --
    confirmed real-production defect: CaixaBank/KONST/Senao/PUFsecurity
    events all persisted `primary_company="PR Newswire"` despite their own
    `related_companies`/entities correctly naming the real subject.

    Only ever engages for `SourceType.PRESS_DISTRIBUTION` (an existing,
    already-classified source property -- never a hardcoded publisher
    name), and only ever promotes a COMPANY/ORGANIZATION-typed entity
    (never a free-text/partnership-derived name) that the release's own
    canonical title actually contains -- the deterministic, source-agnostic
    convention that a press release's own headline names its subject
    company first ("CaixaBank Expands...", "KONST Raises...", "Senao
    Networks Launches..."). Falls back to the distributor's name, honestly,
    when no such entity can be identified -- never fabricates a company
    that isn't in the evidence."""
    if primary_item.source_quality.source_type != SourceType.PRESS_DISTRIBUTION:
        return primary_item.company

    company_entity_types = {EntityType.COMPANY, EntityType.ORGANIZATION}
    title_lower = canonical_title.lower()
    candidates = [
        entity.name
        for item in items
        for entity in item.entities
        if entity.type in company_entity_types and entity.name and entity.name.lower() in title_lower
    ]
    if not candidates:
        return primary_item.company
    for name in candidates:
        if title_lower.startswith(name.lower()):
            return name
    return min(candidates, key=lambda name: title_lower.index(name.lower()))


def _select_event_type(items: List[ResearchItem]) -> Optional[EventType]:
    # Clustering already vetoes conflicting non-None event types, so any
    # non-None value found here is guaranteed consistent across the cluster.
    for item in items:
        if item.event_type is not None:
            return item.event_type
    return None


def _select_event_date(items: List[ResearchItem]) -> Optional[datetime]:
    """Phase 13B: only a member explicitly judged a dated development
    (`is_dated_development`) may contribute its `publication_date` as this
    Event's date -- an evergreen/reference item's `publication_date` (page
    retrieval/modification metadata, not a confirmed announcement date)
    must never be presented as the event's date. If no member qualifies,
    the event's date is honestly unknown (None) rather than invented from
    whichever date happened to be available."""
    dates = [
        item.publication_date
        for item in items
        if item.publication_date is not None and item.is_dated_development
    ]
    return min(dates) if dates else None


def _select_description(items: List[ResearchItem], primary_item: ResearchItem) -> Optional[str]:
    if primary_item.summary:
        return primary_item.summary
    for item in sorted(items, key=lambda i: i.id):
        if item.summary:
            return item.summary
    return None


def _collect_related_companies(
    items: List[ResearchItem], primary_company: str, *, excluded_terms: Optional[Set[str]] = None
) -> List[str]:
    """Two complementary, independent guards keep a non-company name out of
    "related companies":

    1. `item.entities` is now a TYPED (name, type) list (Phase 13B) -- only
       entries the agent itself classified as `COMPANY`/`ORGANIZATION`
       contribute a name here, e.g. "Claude.ai" (type `product`) or
       "Agentic AI" (type `technology`) are excluded by construction,
       never by name-matching.
    2. `item.partnerships` is still a free-text list (unchanged -- the
       agent is instructed, via prompt, to only name a genuine partner
       there), so `excluded_terms` (case-insensitive exact match against
       the configured, non-LLM topic vocabulary) remains the deterministic
       backstop for it, exactly as before this phase.
    """
    excluded_lower = {t.lower() for t in (excluded_terms or ())}
    company_entity_types = {EntityType.COMPANY, EntityType.ORGANIZATION}
    names = set()
    for item in items:
        names.add(item.company)
        names.update(e.name for e in item.entities if e.type in company_entity_types)
        names.update(item.partnerships)
    names.discard(primary_company)
    return sorted(name for name in names if name.lower() not in excluded_lower)


def flatten_non_company_vocabulary(topics) -> Set[str]:
    """Every configured AI/financial-services topic, strategic theme, and
    event-type identifier from `config/topics.yaml` -- the deterministic
    "this is a concept/technology/topic, not a company" vocabulary used by
    `_collect_related_companies`. Takes a `TopicsRegistry` (typed loosely
    here to avoid an import-cycle risk; `EventService` passes the real
    `config.topics`)."""
    buckets = [
        topics.ai_topics.core, topics.ai_topics.models, topics.ai_topics.infrastructure,
        topics.ai_topics.enterprise, topics.ai_topics.governance,
        topics.financial_services.core, topics.financial_services.markets,
        topics.financial_services.wealth, topics.financial_services.risk, topics.financial_services.operations,
        topics.strategic_themes, topics.event_types,
    ]
    return {term for bucket in buckets for term in bucket}
