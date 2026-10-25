"""Deterministic digest theme candidates.

Theme labels come exclusively from config/topics.yaml's `strategic_themes`
— a closed, configured vocabulary. The LLM is never asked to invent a theme
label or decide which events belong to it; matching is a plain keyword
search over each event's own title/description/company fields. An LLM
could optionally be used later to write a theme's `description` prose, but
that is deferred (see the module docstring in `reporting.digest_service`)
— themes are fully deterministic in Phase 6.
"""

from __future__ import annotations

from typing import List

from research_assistant.config.schema import TopicsRegistry
from research_assistant.events.models import Event
from research_assistant.models.digest import DigestTheme


def _event_haystack(event: Event) -> str:
    parts = [event.canonical_title or "", event.description or "", event.primary_company, *event.related_companies]
    return " ".join(parts).lower()


def build_theme_candidates(events: List[Event], topics: TopicsRegistry) -> List[DigestTheme]:
    """One `DigestTheme` per configured strategic theme that matches at least
    one supplied event — never a theme unrelated to the given events."""
    themes: List[DigestTheme] = []
    for theme_label in topics.strategic_themes:
        keyword = theme_label.lower()
        matching_ids = [event.id for event in events if keyword in _event_haystack(event)]
        if matching_ids:
            themes.append(
                DigestTheme(
                    label=theme_label,
                    description=f"{len(matching_ids)} event(s) related to {theme_label.lower()}.",
                    event_ids=matching_ids,
                )
            )
    return themes
