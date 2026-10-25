"""Article-level duplicate detection.

Deliberately narrow: two Articles are duplicates only if they share a
canonical URL, or share a non-null content hash. This is identity
detection, not event clustering — two distinct articles that both cover
the same real-world announcement (different publishers, different wording)
are NOT duplicates here and both survive into Phase 5's event clustering.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from research_assistant.models.article import Article


def deduplicate_articles(articles: List[Article]) -> List[Article]:
    """Collapse duplicate Articles, keeping the most complete copy of each group.

    Grouping is transitive: if A and B share a canonical_url, and B and C
    share a content_hash, all three are treated as one duplicate group.
    Order of the returned list follows the first occurrence of each kept
    article in the input.
    """
    n = len(articles)
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

    first_seen_by_url: Dict[str, int] = {}
    first_seen_by_hash: Dict[str, int] = {}
    for i, article in enumerate(articles):
        if article.canonical_url in first_seen_by_url:
            union(i, first_seen_by_url[article.canonical_url])
        else:
            first_seen_by_url[article.canonical_url] = i

        if article.content_hash:
            if article.content_hash in first_seen_by_hash:
                union(i, first_seen_by_hash[article.content_hash])
            else:
                first_seen_by_hash[article.content_hash] = i

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    kept_indices = [max(group, key=lambda idx: _completeness_key(articles[idx])) for group in groups.values()]
    kept_indices.sort()
    return [articles[i] for i in kept_indices]


def _completeness_key(article: Article):
    text_length = len(article.clean_text) if article.clean_text else 0
    return (
        article.publication_date is not None,
        article.author is not None,
        article.title is not None,
        text_length,
    )
