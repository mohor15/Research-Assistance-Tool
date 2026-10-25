"""Deterministic (no LLM, no Ollama call) construction of a LINK-grounded
ResearchItem -- the second lifecycle path alongside the existing
ResearchAgent.analyze() (which requires real retrieved article content/
Evidence). This is for an article whose body could not be retrieved
(robots.txt, anti-bot, HTTP error, browser-extraction failure) but whose
discovery metadata (title, source, publication date) is itself a reliable,
substantive signal.

HARD RULE: every field produced here must be literally derivable from
`article.title`/`article.source_name`/`article.publication_date` -- never
invented. There is no body text to summarize, no entities to extract, no
quotes, no financial figures, no partnership details beyond what the title
itself states. This is enforced by construction: the functions below never
read `article.clean_text` (there isn't any -- that's the whole point) and
never call an LLM.

Mirrors `ResearchAgent.analyze()`'s OUTPUT shape (a ResearchItem) exactly,
so every downstream stage (event clustering/scoring/digest qualification)
needs no special-casing -- only `grounding_type=LINK` distinguishes it.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from research_assistant.config.schema import AppConfig
from research_assistant.evidence.ids import generate_article_id
from research_assistant.models.article import Article, DatePrecision
from research_assistant.models.enums import AgentName, EventType, GroundingType
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    ResearchItem,
    SourceQuality,
    TopicCategory,
)

logger = logging.getLogger(__name__)

# A title this short (and with no concrete action-type phrase match below)
# reads as a generic/evergreen mention rather than a specific development --
# the same threshold `pipeline.nodes._candidate_rank_key` already uses for
# ranking demotion, reused here as a hard reject (no evidence exists to
# "demote and still keep" for a link-only item -- it must clear this bar
# outright or not become an item at all).
_GENERIC_TITLE_MAX_WORDS = 3


def _flatten_topic_keywords(topics) -> List[str]:
    """Deliberately duplicated (not imported) from
    `pipeline.nodes._flatten_topic_keywords`: `agents.*` must never import
    `pipeline.nodes` (nodes.py imports `agents.base`, so the reverse
    direction would be circular) -- see `tests/agents/test_base.py`'s
    existing layering test. Same source vocabulary, same logic."""
    buckets = [
        topics.ai_topics.core, topics.ai_topics.models, topics.ai_topics.infrastructure,
        topics.ai_topics.enterprise, topics.ai_topics.governance,
        topics.financial_services.core, topics.financial_services.markets,
        topics.financial_services.wealth, topics.financial_services.risk, topics.financial_services.operations,
    ]
    return [keyword.lower() for bucket in buckets for keyword in bucket]


def _match_event_type(title_lower: str, event_types: List[str]) -> Optional[EventType]:
    """The ONLY action-signal check used here: a configured event-type
    phrase (e.g. "partnership", "acquisition", "funding", "model release")
    literally present in the title. Deliberately NOT
    `agents.base._contains_dated_announcement_language` (tuned for body
    prose like "today announced", which a terse headline rarely contains)
    -- a title either names a concrete development type or it doesn't."""
    for event_type_value in event_types:
        phrase = event_type_value.replace("_", " ").lower()
        if phrase and phrase in title_lower:
            try:
                return EventType(event_type_value)
            except ValueError:
                continue
    return None


def build_link_grounded_research_item(
    article: Article, *, config: AppConfig, agent: AgentName,
) -> Optional[ResearchItem]:
    """Returns `None` (no item, exactly like an agent judging something
    IRRELEVANT) unless the title clears two bars: it is not too short/
    generic, AND it names a configured event-type phrase (a concrete
    development, e.g. "partnership"/"funding"/"acquisition" -- rejects
    generic commentary like "AI is changing the world").

    The topical-relevance bar itself is NOT re-checked here: this function
    is only ever called (see `pipeline.nodes.agent_analysis`) for an
    article that the existing deterministic router (`agents.router`)
    already routed to an agent via its own strong-signal check (source-name
    registry membership, or a configured company name/alias actually
    appearing in the title) -- re-requiring a separate topic-keyword hit
    here would be redundant and, worse, would reject the task's own
    worked example ("OpenAI signs major enterprise partnership with
    Microsoft" contains no bare config/topics.yaml phrase, only company
    names already validated by routing). A topic-keyword hit, when
    present, still raises the metadata-confidence level below from LOW to
    MEDIUM."""
    title = article.title
    if not title or not title.strip():
        return None
    title_lower = title.lower()

    topic_keywords = _flatten_topic_keywords(config.topics)
    keyword_hit = any(keyword in title_lower for keyword in topic_keywords)
    event_type = _match_event_type(title_lower, config.topics.event_types)

    if event_type is None:
        logger.info("link_grounded: rejecting %r (no concrete event-type phrase in title)", title)
        return None
    if len(title.split()) <= _GENERIC_TITLE_MAX_WORDS:
        logger.info("link_grounded: rejecting %r (title too short/generic)", title)
        return None

    # Date semantics (unchanged rules, reused exactly): only an
    # EXACT/DATE_ONLY precision date is trusted as a real dated-development
    # signal -- MONTH/UNKNOWN never promote `is_dated_development`, exactly
    # like the content-grounded path's own guard (agents.base's
    # `_resolve_dated_development` / `_build_research_item`). Never invented
    # here if absent.
    is_dated_development = (
        article.publication_date is not None
        and article.publication_date_precision in (DatePrecision.EXACT, DatePrecision.DATE_ONLY)
    )
    publication_date = (
        article.publication_date
        if article.publication_date_precision != DatePrecision.MONTH
        else None
    )

    summary = (
        f"{article.source_name} has published an article titled ‘{title}’. "
        "The article was identified as relevant based on its title and publication metadata, "
        "but automated access to the article content was unavailable. "
        "Open the original article for full details."
    )
    development_date_rationale = (
        "Date taken directly from the source's own publication metadata; article content was "
        "not retrieved, so no body text corroborates this beyond the title itself."
        if is_dated_development else
        "No reliably precise publication date is available for this link-grounded item."
    )

    # LOW/MEDIUM only -- metadata alone can never earn HIGH on these
    # qualitative assessments (there is no article body to substantiate a
    # strong claim); mirrors the explicit confidence ceiling applied at
    # event-scoring time (events.scoring._LINK_GROUNDED_CONFIDENCE_CEILING).
    metadata_level = RelevanceLevel.MEDIUM if keyword_hit else RelevanceLevel.LOW

    item_id = f"link-{generate_article_id(article.canonical_url)}"
    return ResearchItem(
        id=item_id, agent=agent, article_id=article.id, publisher=article.publisher or article.source_name,
        title=title, url=article.url, publication_date=publication_date,
        # The publisher (e.g. "Financial Times") is NOT the subject company --
        # it is already captured above in `publisher`. No subject company is
        # reliably derivable from title/metadata alone without an LLM call,
        # so `company` is left empty (no known subject) rather than falsely
        # feeding the publisher into clustering's entity-overlap signal
        # (events.similarity `_entity_set`, weight 0.35), which caused
        # unrelated same-publisher stories to incorrectly cluster.
        # `ResearchItem.company` is a required `str` (NOT NULL in the DB
        # schema, which this change does not touch) -- "" is the sentinel
        # for "no known company", exactly equivalent to None for every
        # downstream consumer, since `_entity_set()` already filters out
        # falsy/blank values (`if v and v.strip()`).
        company="",
        relevance=RelevanceDecision(
            label=RelevanceLabel.RELEVANT,
            reason=(
                f"Title names a specific {event_type.value.replace('_', ' ')} and matches configured "
                "AI/financial-services topic vocabulary; article content itself was unavailable."
            ),
        ),
        category=TopicCategory.OTHER, event_type=event_type, summary=summary,
        is_dated_development=is_dated_development, development_date_rationale=development_date_rationale,
        entities=[], people=[], products=[], deal_value=None, partnerships=[], technologies=[], geography=[],
        financial_services_relevance=RelevanceAssessment(
            level=metadata_level, rationale="Based on title/topic-keyword match only; no article body available.",
        ),
        nomura_relevance=RelevanceAssessment(
            level=RelevanceLevel.LOW, rationale="Metadata-only item; strategic relevance cannot be assessed without article content.",
        ),
        strategic_significance=RelevanceAssessment(
            level=RelevanceLevel.LOW, rationale="Metadata-only item; strategic significance cannot be assessed without article content.",
        ),
        evidence=[], source_quality=SourceQuality(source_type=article.source_type, source_priority=article.source_priority),
        grounding_type=GroundingType.LINK,
    )
