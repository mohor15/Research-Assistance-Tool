"""Deterministic, local, no-embedding similarity signals used by clustering.

Everything here is pure text/set arithmetic (stdlib only — no external
embedding API, no FAISS). If semantic similarity beyond this is ever
needed, structured signals (entity overlap, event type, date proximity)
are deliberately tried first, per Phase 5's brief: prefer structured
signals over a bare similarity threshold.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional, Set

from research_assistant.events.models import ClusterSignal
from research_assistant.models.enums import EventType
from research_assistant.models.research_item import ResearchItem

# Combined-score weights. These are internal clustering constants, distinct
# from config/settings.yaml's ranking weights (which score already-built
# Events, not raw text similarity) — deliberately not user-configurable.
ENTITY_OVERLAP_WEIGHT = 0.35
EVENT_TYPE_WEIGHT = 0.15
DATE_PROXIMITY_WEIGHT = 0.15
TITLE_SIMILARITY_WEIGHT = 0.20
CONTENT_SIMILARITY_WEIGHT = 0.15

CLUSTER_THRESHOLD = 0.55
DATE_PROXIMITY_WINDOW_DAYS = 5.0

# Beyond this gap, two items are never the same event regardless of how
# similar their text looks — real coverage of one event clusters within
# days, not months (distinguishes "different dates" from "same breaking
# story reported over a few days"). Deliberately wider than
# DATE_PROXIMITY_WINDOW_DAYS, which governs the smooth in-window decay.
DATE_GAP_VETO_DAYS = 14.0

# "OpenAI releases Model A" vs "OpenAI releases Model B": same company, same
# event_type, near-identical phrasing — text similarity alone cannot tell
# these apart. When both items name disjoint products under one of these
# event types, that is treated as decisive evidence of separate events.
RELEASE_LIKE_EVENT_TYPES = {
    EventType.PRODUCT_LAUNCH,
    EventType.MODEL_RELEASE,
    EventType.PLATFORM_RELEASE,
}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with",
    "new", "its", "at", "is", "are", "as", "by",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: Optional[str]) -> Set[str]:
    if not text:
        return set()
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _entity_set(item: ResearchItem) -> Set[str]:
    # `entities` is a typed (name, type) list (Phase 13B) -- only the name
    # participates in text-overlap matching, exactly as a plain string did
    # before; the type itself is irrelevant to similarity matching.
    values: List[str] = [item.company, *(e.name for e in item.entities), *item.partnerships, *item.products]
    return {v.strip().lower() for v in values if v and v.strip()}


def entity_overlap(a: ResearchItem, b: ResearchItem) -> float:
    """Jaccard similarity over {company, entities, partnerships, products}."""
    set_a, set_b = _entity_set(a), _entity_set(b)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def event_type_score(a: ResearchItem, b: ResearchItem) -> float:
    """1.0 if both set and equal, 0.5 (neutral) if either is unknown, 0.0 if
    both set and different. Callers additionally veto a hard mismatch — see
    `is_event_type_conflict`."""
    if a.event_type is None or b.event_type is None:
        return 0.5
    return 1.0 if a.event_type == b.event_type else 0.0


def is_event_type_conflict(a: ResearchItem, b: ResearchItem) -> bool:
    """A hard veto: two known, different event types are never the same event
    (e.g. an acquisition is never the same event as a partnership)."""
    return a.event_type is not None and b.event_type is not None and a.event_type != b.event_type


def is_disjoint_release_products(a: ResearchItem, b: ResearchItem) -> bool:
    """A second hard veto for "Model A" vs "Model B": same release-like event
    type, both name at least one product, and the named products share no
    overlap at all."""
    if a.event_type != b.event_type or a.event_type not in RELEASE_LIKE_EVENT_TYPES:
        return False
    products_a = {p.strip().lower() for p in a.products if p and p.strip()}
    products_b = {p.strip().lower() for p in b.products if p and p.strip()}
    if not products_a or not products_b:
        return False
    return products_a.isdisjoint(products_b)


def is_date_gap_conflict(a: ResearchItem, b: ResearchItem) -> bool:
    """A hard veto: publication dates more than `DATE_GAP_VETO_DAYS` apart are
    never the same event, no matter how similar the text looks (e.g. an
    identical recurring headline reused months later)."""
    if a.publication_date is None or b.publication_date is None:
        return False
    delta_days = abs((a.publication_date - b.publication_date).total_seconds()) / 86400.0
    return delta_days > DATE_GAP_VETO_DAYS


def is_disjoint_entities(a: ResearchItem, b: ResearchItem) -> bool:
    """A hard veto: zero shared company/entity/partnership/product name means
    there is no anchor connecting the two items to the same real-world
    subject — prevents clustering purely on generic topical similarity
    ("DO NOT simply cluster everything by topic")."""
    set_a, set_b = _entity_set(a), _entity_set(b)
    if not set_a or not set_b:
        return False  # missing metadata alone should not force a veto
    return set_a.isdisjoint(set_b)


def date_proximity(a: ResearchItem, b: ResearchItem) -> float:
    """1.0 for the same day, linearly decaying to 0.0 over
    `DATE_PROXIMITY_WINDOW_DAYS`. Neutral (0.5) when either date is missing —
    missing metadata should not by itself prevent clustering."""
    if a.publication_date is None or b.publication_date is None:
        return 0.5
    delta_days = abs((a.publication_date - b.publication_date).total_seconds()) / 86400.0
    if delta_days >= DATE_PROXIMITY_WINDOW_DAYS:
        return 0.0
    return 1.0 - (delta_days / DATE_PROXIMITY_WINDOW_DAYS)


def title_similarity(a: ResearchItem, b: ResearchItem) -> float:
    """Token-set Jaccard over normalized (lowercased, stopword-stripped) titles."""
    tokens_a, tokens_b = _tokenize(a.title), _tokenize(b.title)
    if not tokens_a and not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def content_similarity(a: ResearchItem, b: ResearchItem) -> float:
    """`difflib.SequenceMatcher` ratio over each item's agent-authored summary
    (the only free-text content a ResearchItem carries) — a fallback signal,
    not the primary decision-maker."""
    text_a, text_b = (a.summary or "").strip().lower(), (b.summary or "").strip().lower()
    if not text_a or not text_b:
        return 0.0
    return SequenceMatcher(None, text_a, text_b).ratio()


def combined_score(a: ResearchItem, b: ResearchItem) -> float:
    return (
        ENTITY_OVERLAP_WEIGHT * entity_overlap(a, b)
        + EVENT_TYPE_WEIGHT * event_type_score(a, b)
        + DATE_PROXIMITY_WEIGHT * date_proximity(a, b)
        + TITLE_SIMILARITY_WEIGHT * title_similarity(a, b)
        + CONTENT_SIMILARITY_WEIGHT * content_similarity(a, b)
    )


_NEAR_DUPLICATE_TITLE_THRESHOLD = 0.9
_NEAR_DUPLICATE_MAX_DATE_GAP_DAYS = 1.0


def is_near_duplicate_title_same_day(a: ResearchItem, b: ResearchItem) -> bool:
    """Confirmed forensic finding (CURRENT run `5bac1b659bc8cceb`): the
    same PR Newswire release, syndicated under two distinct URLs (two
    separate numeric release IDs, two different content hashes -- so
    Phase 2's article-level dedup correctly never merges them), produced
    two ResearchItems with an IDENTICAL title and the SAME exact
    publication timestamp, yet the weighted `combined_score` (0.51) fell
    just under `CLUSTER_THRESHOLD` (0.55) -- both items' `summary` was
    `None` (content_similarity=0.0, a real signal lost entirely) and the
    two independent Qwen calls extracted noticeably different entity/
    product lists from near-identical text (entity_overlap=0.25). An
    (almost) verbatim title match on the same day is, on its own, already
    overwhelming evidence of the same real-world release -- stronger
    than the weighted score needs it to be -- so it is treated as a
    direct override (same role as the existing hard vetoes, just in the
    opposite direction): it can only ever ADD a cluster membership that
    the weighted score would have missed, never override one of the
    existing hard vetoes above (checked after all of them), and never
    lowers the bar for any pair with a materially different title."""
    if a.publication_date is None or b.publication_date is None:
        return False
    delta_days = abs((a.publication_date - b.publication_date).total_seconds()) / 86400.0
    if delta_days > _NEAR_DUPLICATE_MAX_DATE_GAP_DAYS:
        return False
    return title_similarity(a, b) >= _NEAR_DUPLICATE_TITLE_THRESHOLD


def should_cluster(a: ResearchItem, b: ResearchItem) -> bool:
    """The full pairwise decision: hard vetoes first, then the weighted
    combined score against `CLUSTER_THRESHOLD`, with one narrow override
    (`is_near_duplicate_title_same_day`) for a near-verbatim same-day
    title match that the weighted score alone would otherwise miss."""
    if is_event_type_conflict(a, b):
        return False
    if is_disjoint_release_products(a, b):
        return False
    if is_date_gap_conflict(a, b):
        return False
    if is_disjoint_entities(a, b):
        return False
    if is_near_duplicate_title_same_day(a, b):
        return True
    return combined_score(a, b) >= CLUSTER_THRESHOLD


def matched_signals(a: ResearchItem, b: ResearchItem) -> List[ClusterSignal]:
    """Which individual signals meaningfully contributed, for explainability."""
    signals: List[ClusterSignal] = []
    if entity_overlap(a, b) > 0:
        signals.append(ClusterSignal.ENTITY_OVERLAP)
    if event_type_score(a, b) == 1.0:
        signals.append(ClusterSignal.EVENT_TYPE_MATCH)
    if date_proximity(a, b) > 0:
        signals.append(ClusterSignal.DATE_PROXIMITY)
    if title_similarity(a, b) > 0:
        signals.append(ClusterSignal.TITLE_SIMILARITY)
    if content_similarity(a, b) > 0:
        signals.append(ClusterSignal.CONTENT_SIMILARITY)
    return signals
