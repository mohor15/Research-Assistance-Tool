"""Phase 13B: date grounding (an evergreen/reference item's `publication_date`
must never become a manufactured Event date) and entity typing (only
`company`/`organization`-typed entities are ever surfaced as "related
companies"). Fully deterministic -- no LLM involved; `is_dated_development`
and typed `entities` are ordinary ResearchItem fields set directly here,
exactly as a real agent assessment would populate them.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.events.builder import build_event_draft
from research_assistant.events.clustering import Cluster
from research_assistant.events.models import ClusterMetadata
from research_assistant.models.enums import EntityType
from research_assistant.models.research_item import TypedEntity

from tests.events.conftest import make_item

D = datetime(2026, 9, 28, tzinfo=timezone.utc)


def _draft_for(items, **kwargs):
    return build_event_draft(Cluster(items=items, metadata=ClusterMetadata(member_count=len(items))), **kwargs)


# ---------------------------------------------------------------------------
# Date grounding (Problem 3/4, tests 4-7)
# ---------------------------------------------------------------------------


def test_an_evergreen_items_publication_date_never_becomes_the_event_date():
    """Test 4: a page with retrieval/modification metadata but no actual
    dated announcement (is_dated_development=False) must not have that
    date presented as the Event's own date."""
    item = make_item(
        "a1", company="Anthropic", title="Claude Sonnet", publication_date=D,
        is_dated_development=False,
    )

    draft = _draft_for([item])

    assert draft.event_date is None


def test_a_dated_announcement_keeps_its_correct_date():
    """Test 5: the normal, common case -- a real dated announcement is
    unaffected by this change."""
    item = make_item(
        "a1", company="NVIDIA", title="NVIDIA Launches Open Agent Safety Platform",
        publication_date=D, is_dated_development=True,
    )

    draft = _draft_for([item])

    assert draft.event_date == D


def test_evergreen_page_still_forms_an_event_just_without_a_fabricated_date():
    """Test 6: an evergreen/reference page is NOT suppressed into no event
    at all (the architecture must not simply hide it) -- it still produces
    an Event, with every other field intact, just an honest unknown date."""
    item = make_item(
        "a1", company="Anthropic", title="Claude Sonnet", publication_date=D,
        is_dated_development=False,
    )

    draft = _draft_for([item])

    assert draft.canonical_title == "Claude Sonnet"
    assert draft.primary_company == "Anthropic"
    assert draft.event_date is None
    assert draft.primary_source.url == item.url


def test_explicit_dated_announcement_remains_eligible_for_a_real_date():
    """Test 7: a genuine announcement page is unaffected -- it is not
    penalized or treated with suspicion just because evergreen pages exist
    elsewhere in the system."""
    item = make_item(
        "a1", company="Cohere", title="Cohere Embed 5", publication_date=D,
        is_dated_development=True,
    )

    draft = _draft_for([item])

    assert draft.event_date == D


def test_mixed_cluster_uses_only_the_dated_members_date():
    """A dated item sharing a cluster with an evergreen item contributes its
    date; the evergreen item's date is simply not considered."""
    dated = make_item(
        "a1", company="NVIDIA", title="t", publication_date=D, is_dated_development=True,
    )
    evergreen = make_item(
        "a2", company="NVIDIA", title="t", publication_date=D - timedelta(days=30), is_dated_development=False,
    )

    draft = _draft_for([dated, evergreen])

    assert draft.event_date == D  # not the evergreen item's much-earlier date


# ---------------------------------------------------------------------------
# Entity typing (Problem 6, tests 8-11)
# ---------------------------------------------------------------------------


def test_claude_ai_typed_as_product_is_not_a_related_company():
    """Test 8."""
    item = make_item(
        "a1", company="Anthropic", title="t",
        entities=[TypedEntity(name="Claude.ai", type=EntityType.PRODUCT)],
    )

    draft = _draft_for([item])

    assert "Claude.ai" not in draft.related_companies


def test_claude_sonnet_typed_as_model_is_not_a_related_company():
    """Test 9."""
    item = make_item(
        "a1", company="Anthropic", title="t",
        entities=[TypedEntity(name="Claude Sonnet", type=EntityType.MODEL)],
    )

    draft = _draft_for([item])

    assert "Claude Sonnet" not in draft.related_companies


def test_agentic_ai_typed_as_technology_is_not_a_related_company():
    """Test 10 -- the structural (typed-entity) counterpart to the existing
    vocabulary-blacklist regression test for the `partnerships` field."""
    item = make_item(
        "a1", company="NVIDIA", title="t",
        entities=[TypedEntity(name="Agentic AI", type=EntityType.TECHNOLOGY)],
    )

    draft = _draft_for([item])

    assert "Agentic AI" not in draft.related_companies


def test_a_genuine_company_typed_as_company_is_a_related_company():
    """Test 11."""
    item = make_item(
        "a1", company="NVIDIA", title="t",
        entities=[TypedEntity(name="CoreWeave", type=EntityType.COMPANY)],
    )

    draft = _draft_for([item])

    assert "CoreWeave" in draft.related_companies


def test_mixed_typed_entities_only_company_and_organization_survive():
    item = make_item(
        "a1", company="Anthropic", title="t",
        entities=[
            TypedEntity(name="Claude.ai", type=EntityType.PRODUCT),
            TypedEntity(name="Claude Platform", type=EntityType.PLATFORM),
            TypedEntity(name="Agentic AI", type=EntityType.TECHNOLOGY),
            TypedEntity(name="Amazon Web Services", type=EntityType.COMPANY),
            TypedEntity(name="Partnership on AI", type=EntityType.ORGANIZATION),
        ],
    )

    draft = _draft_for([item])

    assert draft.related_companies == ["Amazon Web Services", "Partnership on AI"]
