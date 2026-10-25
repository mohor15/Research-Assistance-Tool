"""Multi-signal, deterministic, explainable event clustering.

A cluster represents the SAME underlying real-world event, possibly
described by several `ResearchItem`s. This is deliberately separate from
Phase 2's article-level deduplication (same URL/content hash = same
Article); here, DIFFERENT articles/research items may describe the SAME
event and get grouped, which Phase 2 never does.

Grouping is transitive (union-find, mirroring `evidence.dedup`): if A~B and
B~C pass `similarity.should_cluster`, all three land in one cluster even if
A~C alone would not — this is what makes 3+ outlets covering one
announcement group together.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from research_assistant.events.models import ClusterMetadata, PairwiseMatch
from research_assistant.events.similarity import combined_score, matched_signals, should_cluster
from research_assistant.models.research_item import ResearchItem


@dataclass
class Cluster:
    items: List[ResearchItem]
    metadata: ClusterMetadata = field(default_factory=lambda: ClusterMetadata(member_count=0))


def cluster_research_items(items: List[ResearchItem]) -> List[Cluster]:
    n = len(items)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    all_matches: List[PairwiseMatch] = []
    for i in range(n):
        for j in range(i + 1, n):
            if should_cluster(items[i], items[j]):
                union(i, j)
                all_matches.append(
                    PairwiseMatch(
                        research_item_id_a=items[i].id,
                        research_item_id_b=items[j].id,
                        combined_score=round(combined_score(items[i], items[j]), 4),
                        matched_signals=matched_signals(items[i], items[j]),
                    )
                )

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    matches_by_root: Dict[int, List[PairwiseMatch]] = defaultdict(list)
    for match in all_matches:
        # Any member of the pair identifies which cluster this match belongs to.
        idx_a = next(i for i, item in enumerate(items) if item.id == match.research_item_id_a)
        matches_by_root[find(idx_a)].append(match)

    clusters: List[Cluster] = []
    for root in sorted(groups, key=lambda r: min(groups[r])):  # preserve first-seen order
        member_indices = groups[root]
        member_items = [items[i] for i in member_indices]
        metadata = ClusterMetadata(member_count=len(member_items), matches=matches_by_root.get(root, []))
        clusters.append(Cluster(items=member_items, metadata=metadata))

    return clusters
