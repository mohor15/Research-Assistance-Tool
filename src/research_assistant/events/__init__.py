"""Event clustering and deterministic relevance/confidence scoring.

    ResearchItem (Phase 3/4)
          |
    cluster_research_items()   -- events.clustering  (multi-signal, deterministic)
          |
       Cluster
          |
    build_event()               -- events.builder     (canonical fields, provenance)
          |
       Event (unscored)
          |
    EventService.score()        -- events.scoring     (relevance + confidence, config-driven)
          |
       scored Events
          |
    rank_events()                -- events.ranking     (deterministic sort)
          |
      Ranked Events

`EventService` (events.service) is the intended single entry point. Nothing
in this package calls an LLM, and nothing imports
`research_assistant.retrieval` — it only ever reasons over the
`ResearchItem` objects agents already produced.
"""

from research_assistant.events.clock import Clock, FixedClock, SystemClock
from research_assistant.events.models import (
    ClusterMetadata,
    ConfidenceBreakdown,
    Event,
    EventScoringBreakdown,
    PairwiseMatch,
    RelevanceBreakdown,
    SourceRef,
)
from research_assistant.events.service import EventService

__all__ = [
    "EventService",
    "Clock",
    "SystemClock",
    "FixedClock",
    "Event",
    "SourceRef",
    "ClusterMetadata",
    "PairwiseMatch",
    "RelevanceBreakdown",
    "ConfidenceBreakdown",
    "EventScoringBreakdown",
]
