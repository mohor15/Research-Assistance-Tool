"""Event builder: canonical title/date/description selection, related companies,
primary/secondary source tie-breaking, and provenance (source URLs never lost)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.events.builder import build_event_draft, flatten_non_company_vocabulary
from research_assistant.events.clustering import Cluster
from research_assistant.events.models import ClusterMetadata
from research_assistant.models.enums import EventType, SourcePriority, SourceType

from tests.events.conftest import make_item

D = datetime(2025, 9, 10, 12, 0, tzinfo=timezone.utc)


def _draft_for(items, **kwargs):
    return build_event_draft(Cluster(items=items, metadata=ClusterMetadata(member_count=len(items))), **kwargs)


def test_canonical_title_prefers_primary_source_title():
    primary = make_item("a1", company="JPMorgan Chase", title="JPMorgan launches AI assistant",
                         event_type=EventType.PRODUCT_LAUNCH, publication_date=D, source_type=SourceType.PRIMARY)
    secondary = make_item("a2", company="Reuters", title="Reuters covers JPMorgan AI news",
                           event_type=EventType.PRODUCT_LAUNCH, publication_date=D, source_type=SourceType.SECONDARY,
                           entities=["JPMorgan Chase"])

    draft = _draft_for([secondary, primary])

    assert draft.canonical_title == "JPMorgan launches AI assistant"


def test_canonical_title_falls_back_when_primary_has_no_title():
    primary = make_item("a1", company="JPMorgan Chase", title=None,
                         event_type=EventType.PRODUCT_LAUNCH, publication_date=D, source_type=SourceType.PRIMARY)
    other = make_item("a2", company="JPMorgan Chase", title="JPMorgan introduces new AI assistant",
                       event_type=EventType.PRODUCT_LAUNCH, publication_date=D, source_type=SourceType.PRIMARY)

    draft = _draft_for([primary, other])

    assert draft.canonical_title == "JPMorgan introduces new AI assistant"


def test_canonical_title_generic_fallback_when_nothing_has_a_title():
    a = make_item("a1", company="JPMorgan Chase", title=None, event_type=None, publication_date=None)

    draft = _draft_for([a])

    assert draft.canonical_title == "JPMorgan Chase development"


def test_event_date_is_earliest_among_members():
    earlier = make_item("a1", company="X", title="t", publication_date=D)
    later = make_item("a2", company="X", title="t", publication_date=D + timedelta(days=2))

    draft = _draft_for([later, earlier])

    assert draft.event_date == D


def test_event_date_is_none_when_no_member_has_one():
    a = make_item("a1", company="X", title="t", publication_date=None)
    b = make_item("a2", company="X", title="t", publication_date=None)

    draft = _draft_for([a, b])

    assert draft.event_date is None


def test_related_companies_excludes_primary_company_and_dedupes():
    a = make_item("a1", company="JPMorgan Chase", title="t", entities=["Acme AI", "JPMorgan Chase"], partnerships=["Acme AI"])
    b = make_item("a2", company="JPMorgan Chase", title="t", entities=["Acme AI"])

    draft = _draft_for([a, b])

    assert "JPMorgan Chase" not in draft.related_companies
    assert draft.related_companies == ["Acme AI"]


def test_press_distribution_primary_company_uses_the_subject_entity_not_the_distributor():
    """Forensic finding (2026-10-12): `ResearchItem.company` is copied from
    the SOURCE REGISTRY's name (see `agents.base._build_research_item`),
    which for a press-distribution wire is the DISTRIBUTOR ("PR Newswire"),
    never the company the release is actually about. Real regression:
    CaixaBank/KONST/Senao/PUFsecurity events all persisted
    `primary_company="PR Newswire"` despite a correct COMPANY-typed entity
    naming the real subject, matching the title's own subject-first
    convention."""
    item = make_item(
        "a1", company="PR Newswire", title="CaixaBank Expands Strategic Collaboration with Google Cloud",
        entities=["CaixaBank", "Google Cloud"], source_type=SourceType.PRESS_DISTRIBUTION,
    )

    draft = _draft_for([item])

    assert draft.primary_company == "CaixaBank"
    assert "Google Cloud" in draft.related_companies
    assert "CaixaBank" not in draft.related_companies


def test_non_press_distribution_primary_company_is_unaffected():
    """A source that publishes its own content (not a wire distributor)
    keeps the existing, unmodified behavior -- `item.company` is already
    the real company."""
    item = make_item(
        "a1", company="Cohere", title="Shared vs. Dedicated AI Inference: Embed & Rerank",
        entities=["Cohere"], source_type=SourceType.PRIMARY,
    )

    draft = _draft_for([item])

    assert draft.primary_company == "Cohere"


def test_press_distribution_falls_back_to_distributor_when_no_matching_entity():
    """No COMPANY/ORGANIZATION-typed entity in the title -- the function
    must never fabricate a subject company, so it honestly falls back to
    the distributor's name rather than guessing."""
    item = make_item(
        "a1", company="PR Newswire", title="Industry Report on Data Center Growth",
        entities=["Unrelated Analytics Corp"], source_type=SourceType.PRESS_DISTRIBUTION,
    )

    draft = _draft_for([item])

    assert draft.primary_company == "PR Newswire"


def test_real_world_regression_agentic_ai_is_not_treated_as_a_related_company():
    """Reproduces the exact real-run defect: the LLM listed "Agentic AI" (a
    technology/concept) under `partnerships` alongside genuine companies
    (CoreWeave, Ennoble Care). With the configured topic vocabulary as
    `excluded_terms`, "Agentic AI" must be filtered out while the genuine
    companies survive."""
    item = make_item(
        "a1", company="NVIDIA", title="t",
        partnerships=["NVIDIA", "CoreWeave", "Agentic AI", "Ennoble Care"],
    )
    from research_assistant.config import get_config

    excluded_terms = flatten_non_company_vocabulary(get_config().topics)

    draft = _draft_for([item], excluded_terms=excluded_terms)

    assert "Agentic AI" not in draft.related_companies
    assert "CoreWeave" in draft.related_companies
    assert "Ennoble Care" in draft.related_companies


def test_without_excluded_terms_behavior_is_unchanged_backward_compatible():
    """Default (no excluded_terms supplied) preserves the original,
    unfiltered behavior -- existing callers with no TopicsRegistry on hand
    are unaffected."""
    item = make_item("a1", company="NVIDIA", title="t", partnerships=["Agentic AI"])

    draft = _draft_for([item])

    assert "Agentic AI" in draft.related_companies


def test_flatten_non_company_vocabulary_includes_agentic_ai_from_real_config():
    from research_assistant.config import get_config

    vocabulary_lower = {t.lower() for t in flatten_non_company_vocabulary(get_config().topics)}
    assert "agentic ai" in vocabulary_lower


def test_description_uses_primary_source_summary():
    primary = make_item("a1", company="X", title="t", source_type=SourceType.PRIMARY, summary="Primary's own summary.")
    secondary = make_item("a2", company="X", title="t", source_type=SourceType.SECONDARY, summary="Secondary's summary.")

    draft = _draft_for([secondary, primary])

    assert draft.description == "Primary's own summary."


def test_provenance_never_loses_original_source_urls():
    a = make_item("a1", company="X", title="t", url="https://a.example.com/story")
    b = make_item("a2", company="X", title="t", url="https://b.example.com/story")

    draft = _draft_for([a, b])

    all_urls = {draft.primary_source.url} | {ref.url for ref in draft.secondary_sources}
    assert all_urls == {"https://a.example.com/story", "https://b.example.com/story"}
    assert set(draft.articles) == {"art-a1", "art-a2"}
    assert set(draft.research_items) == {"a1", "a2"}


def test_higher_priority_breaks_ties_between_equal_source_type():
    high = make_item("a1", company="X", title="t", source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH)
    medium = make_item("a2", company="X", title="t", source_type=SourceType.PRIMARY, source_priority=SourcePriority.MEDIUM)

    draft = _draft_for([medium, high])

    assert draft.primary_source.research_item_id == "a1"


def test_earlier_date_breaks_ties_when_type_and_priority_equal():
    earlier = make_item("a1", company="X", title="t", source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH, publication_date=D)
    later = make_item("a2", company="X", title="t", source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH, publication_date=D + timedelta(days=1))

    draft = _draft_for([later, earlier])

    assert draft.primary_source.research_item_id == "a1"


def test_duplicate_article_ids_within_one_event_are_deduplicated():
    """Two ResearchItems (e.g. the same article routed to two agents) can
    share an article_id; `event_articles` has PRIMARY KEY (event_id,
    article_id), so the draft must never list that article_id twice."""
    a = make_item("a1", company="X", title="t", article_id="shared-article")
    b = make_item("a2", company="X", title="t", article_id="shared-article")

    draft = _draft_for([a, b])

    assert draft.articles == ["shared-article"]


def test_duplicate_article_ids_preserve_first_seen_order():
    a = make_item("a1", company="X", title="t", article_id="art-1")
    b = make_item("a2", company="X", title="t", article_id="art-2")
    c = make_item("a3", company="X", title="t", article_id="art-1")  # duplicate, later position

    draft = _draft_for([a, b, c])

    assert draft.articles == ["art-1", "art-2"]


def test_distinct_article_ids_are_unaffected_by_dedup():
    a = make_item("a1", company="X", title="t", article_id="art-1")
    b = make_item("a2", company="X", title="t", article_id="art-2")
    c = make_item("a3", company="X", title="t", article_id="art-3")

    draft = _draft_for([a, b, c])

    assert draft.articles == ["art-1", "art-2", "art-3"]


def test_research_item_id_breaks_final_tie_deterministically():
    a = make_item("z-item", company="X", title="t", source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH)
    b = make_item("a-item", company="X", title="t", source_type=SourceType.PRIMARY, source_priority=SourcePriority.HIGH)

    draft = _draft_for([a, b])

    assert draft.primary_source.research_item_id == "a-item"  # lexicographically smaller id wins the tie
