"""Pydantic schemas for the three configuration registries.

These models validate structure and internal consistency (e.g. ranking
weights summing to 1.0) only. They never perform I/O, network calls, or
LLM/database access — that is left to later phases.
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional
from zoneinfo import ZoneInfoNotFoundError, available_timezones

from pydantic import BaseModel, Field, field_validator, model_validator

from research_assistant.models.enums import SourcePriority, SourceType

_WEIGHT_SUM_TOLERANCE = 1e-6


def _check_weights_sum_to_one(weights: Dict[str, float], label: str) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"{label} weights must sum to 1.0, got {total!r}: {weights!r}")


# ---------------------------------------------------------------------------
# settings.yaml
# ---------------------------------------------------------------------------

SUPPORTED_LLM_PROVIDERS = ("ollama",)


class LLMSettings(BaseModel):
    provider: Literal["ollama"]
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0)

    @field_validator("base_url")
    @classmethod
    def _base_url_looks_like_a_url(cls, value: str) -> str:
        if not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError(f"llm.base_url must start with http:// or https://, got {value!r}")
        return value


class SystemSettings(BaseModel):
    timezone: str
    llm: LLMSettings

    @field_validator("timezone")
    @classmethod
    def _timezone_must_be_known(cls, value: str) -> str:
        if value not in available_timezones():
            raise ZoneInfoNotFoundError(f"Unknown IANA timezone: {value!r}")
        return value


class ResearchSettings(BaseModel):
    lookback_hours: int = Field(gt=0)
    top_k_per_agent: int = Field(gt=0)
    morning_digest_hour: int = Field(ge=0, le=23)
    evening_digest_hour: int = Field(ge=0, le=23)
    use_primary_sources_first: bool
    use_secondary_sources: bool
    # CURRENT mode's rolling window width (see `scheduling.period.
    # compute_current_period`) -- independent of `lookback_hours`, which is
    # a separate, wider article-HISTORY retention floor (Phase 9) never
    # used to define CURRENT's own reporting period. Defaults to 48 (per
    # management's requirement to widen CURRENT from its original 24h) so
    # a settings.yaml predating this field still loads unchanged.
    current_window_hours: int = Field(default=48, gt=0)
    # Automated CURRENT scheduling (every-other-day scan, Section 1): the
    # hour-of-day (configured `system.timezone`) the scheduler's CURRENT
    # job fires on its scheduled days -- see `scheduling.scheduler`'s
    # `IntervalTrigger(days=2, ...)` registration. Independent of

class RetrievalSettings(BaseModel):
    request_timeout_seconds: int = Field(gt=0)
    max_retries: int = Field(ge=0)
    retry_delay_seconds: float = Field(ge=0)
    max_articles_per_source: int = Field(gt=0)
    max_candidates_per_source: int = Field(gt=0)
    browser_fallback: bool
    respect_robots_txt: bool
    respect_rate_limits: bool
    cache_pages: bool


class RelevanceWeights(BaseModel):
    ai_relevance: float
    financial_services_relevance: float
    strategic_significance: float
    recency: float
    novelty: float

    @model_validator(mode="after")
    def _sum_to_one(self) -> "RelevanceWeights":
        _check_weights_sum_to_one(self.model_dump(), "ranking.relevance")
        return self


class ConfidenceWeights(BaseModel):
    primary_source_confirmation: float
    multiple_source_corroboration: float
    publication_date_certainty: float
    entity_extraction_confidence: float
    event_classification_confidence: float
    content_completeness: float

    @model_validator(mode="after")
    def _sum_to_one(self) -> "ConfidenceWeights":
        _check_weights_sum_to_one(self.model_dump(), "ranking.confidence")
        return self


class RankingSettings(BaseModel):
    relevance: RelevanceWeights
    confidence: ConfidenceWeights


class DeduplicationSettings(BaseModel):
    enabled: bool
    similarity_threshold: float = Field(ge=0.0, le=1.0)
    cluster_related_events: bool


class AgentToggle(BaseModel):
    enabled: bool
    top_k: Optional[int] = Field(default=None, gt=0)
    # CURRENT-mode-only budget override (Part B3): None means "use `top_k`
    # unchanged", so a source registry/settings file with no
    # `current_top_k` entries at all reproduces Morning/Evening's existing
    # behavior exactly, including for CURRENT runs.
    current_top_k: Optional[int] = Field(default=None, gt=0)


class AgentsSettings(BaseModel):
    peer: AgentToggle
    ai_product: AgentToggle
    startup: AgentToggle
    analyst: AgentToggle
    financial_news: AgentToggle
    report: AgentToggle


class CurrentLLMSettings(BaseModel):
    """CURRENT-only bounded LLM funnel budget (see `pipeline.nodes`'s
    CURRENT-mode `agent_analysis`): caps total Ollama calls independent of
    how many candidates were discovered, instead of every specialist agent
    independently calling the LLM on every one of its routed candidates.
    Never applied to Morning/Evening, which keep their existing per-agent
    `top_k`/`current_top_k` behavior unchanged."""

    triage_batch_size: int = Field(default=3, gt=0)
    max_triage_batches: int = Field(default=8, gt=0)
    max_deep_analysis: int = Field(default=8, gt=0)
    triage_timeout_seconds: float = Field(default=90.0, gt=0)
    deep_analysis_timeout_seconds: float = Field(default=90.0, gt=0)

    # Candidate-backlog settings (eligible-but-not-yet-processed candidates
    # carried forward across runs instead of being lost to a per-run
    # budget -- see `storage.repositories.backlog` and `pipeline.nodes`'s
    # CURRENT-mode `agent_analysis`). All bounded and configurable, per the
    # same "never simply raise every budget" principle as the settings above.
    backlog_claim_limit: int = Field(
        default=8, ge=0,
        description="Max pending backlog candidates merged into one CURRENT run's candidate pool.",
    )
    backlog_max_attempts: int = Field(
        default=5, gt=0,
        description="Attempts (deep-analysis failures/timeouts) before a backlog candidate is marked "
                     "rejected_retry_exhausted instead of remaining pending_retry forever.",
    )
    backlog_retention_days: int = Field(
        default=14, gt=0,
        description="Age (days, by last update) after which a TERMINAL backlog row becomes eligible for "
                     "cleanup -- pending rows are never removed by this regardless of age.",
    )
    tier_b_excerpt_chars: int = Field(
        default=400, gt=0,
        description="Max characters of already-retrieved article text shown verbatim in a Tier B "
                     "('awaiting specialist analysis') card -- never an LLM-generated summary.",
    )

    # On-demand Tier B newsletter report generation (see
    # `reporting.tier_b_reports`) -- deliberately separate from the Tier B
    # CARD settings above, which govern the always-free excerpt/key-points
    # display and are unaffected by this feature.
    tier_b_report_min_sufficient_chars: int = Field(
        default=200, gt=0,
        description="Min characters of the article's own effective text (Evidence.content -> "
                     "Article.clean_text -> Article.raw_text) required, alongside a 'success' "
                     "extraction_status, before a Tier B newsletter report is generated for it -- "
                     "mirrors retrieval.extractor.MIN_SUCCESS_CHARS's existing real-extraction "
                     "threshold rather than inventing an independent one.",
    )
    max_tier_b_reports_per_newsletter_request: int = Field(
        default=5, gt=0,
        description="Bounded cap on how many NEW (not-already-cached) Tier B reports one "
                     "'Generate Newsletter' click will generate, sequentially, through the local "
                     "CPU-only qwen2.5:3b model -- keeps the action's worst-case synchronous "
                     "latency bounded (this click runs inside the same Streamlit request/response "
                     "cycle, the same kind of long synchronous block a prior fix already had to "
                     "move a full CURRENT scan OUT of -- see ui/app.py's own note on that). A "
                     "conservative default given each deep-analysis-class call can take up to "
                     "~90s on this CPU-only model.",
    )

    # Explicit, user-triggered single-article live re-fetch for a Tier B
    # article whose persisted/network-free-re-extracted text remains
    # insufficient (see `reporting.tier_b_newsletter_resolution.
    # fetch_full_article_for_tier_b`) -- deliberately NEVER invoked
    # automatically by resolution/rendering/selection; it is its own
    # explicit UI action, bounded to exactly one request for exactly one
    # article, with no retry loop.
    tier_b_manual_fetch_timeout_seconds: float = Field(
        default=20.0, gt=0,
        description="Timeout for the single, explicit, user-triggered live re-fetch of ONE Tier B "
                     "article's own URL. Deliberately a dedicated, separately-configurable bound "
                     "(not reused from the CURRENT scan's retrieval.request_timeout_seconds) "
                     "because this runs synchronously inside a Streamlit button click, not a "
                     "background scan.",
    )
    tier_b_manual_fetch_max_html_chars: int = Field(
        default=3_000_000, gt=0,
        description="Bounded response-size limit (characters of raw HTML) for the manual Tier B "
                     "re-fetch action. A response at or beyond this size is treated as a failure "
                     "(never truncated mid-tag and fed to the extractor) -- protects the "
                     "synchronous UI action from an unexpectedly huge page, not a real-world "
                     "article size.",
    )


class ReportSettings(BaseModel):
    include_relevance_score: bool
    include_confidence_score: bool
    include_source_evidence: bool
    require_source_for_claim: bool
    include_nomura_opportunity: bool


class StorageSettings(BaseModel):
    database: Literal["sqlite"]
    database_path: str = Field(min_length=1)


_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate_email_address(value: str, *, field_label: str) -> str:
    if not _EMAIL_PATTERN.match(value):
        raise ValueError(f"{field_label} must be a valid email address, got {value!r}")
    return value


class EmailSettings(BaseModel):
    """Email-delivery configuration only -- never credentials. SMTP
    authentication secrets (the password/app-password) are read exclusively
    from the `SMTP_PASSWORD` environment variable (see
    `config.loader.load_smtp_password`), never from this model or any YAML
    file, so they can never be accidentally logged or displayed alongside
    the rest of a loaded `Settings` object. `enabled=False` and empty
    `smtp_host`/`sender_address` are the safe defaults: with no `email:`
    section in settings.yaml at all, delivery stays disabled rather than
    guessing at SMTP settings."""

    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = Field(default=587, gt=0, le=65535)
    smtp_use_tls: bool = True
    smtp_username: str = ""
    sender_address: str = ""
    default_recipient: str = "ghosalmohor@gmail.com"

    @field_validator("default_recipient")
    @classmethod
    def _default_recipient_is_a_valid_email(cls, value: str) -> str:
        return _validate_email_address(value, field_label="email.default_recipient")

    @field_validator("sender_address")
    @classmethod
    def _sender_address_is_valid_when_present(cls, value: str) -> str:
        if not value:
            return value
        return _validate_email_address(value, field_label="email.sender_address")


class AuthSettings(BaseModel):
    """A single shared login gate for the Streamlit UI -- not a
    multi-user/per-account system. The username is not a secret (default
    `"admin"`, overridable here); the password is NEVER configured here --
    it is read only from the `APP_LOGIN_PASSWORD` environment variable
    (see `config.loader.load_app_login_password`), mirroring how the SMTP/
    Resend secrets are handled. `enabled=True` is the default: with no
    `auth:` section in settings.yaml at all, the login gate is still on."""

    enabled: bool = True
    username: str = "admin"


class Settings(BaseModel):
    system: SystemSettings
    research: ResearchSettings
    retrieval: RetrievalSettings
    ranking: RankingSettings
    deduplication: DeduplicationSettings
    agents: AgentsSettings
    current_llm: CurrentLLMSettings = Field(default_factory=CurrentLLMSettings)
    report: ReportSettings
    storage: StorageSettings
    email: EmailSettings = Field(default_factory=EmailSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)


# ---------------------------------------------------------------------------
# sources.yaml
# ---------------------------------------------------------------------------


class SourceSearchConfig(BaseModel):
    """Optional interactive-search discovery, additive to a source's normal
    hub/RSS discovery (see `retrieval.ft_search`). Generic at the schema
    level -- any source COULD configure this -- but only Financial Times
    does today, and the one implementation that reads it
    (`retrieval.ft_search`) is deliberately narrow/FT-specific, not a
    generic site-search framework."""

    enabled: bool = False
    search_terms: List[str] = Field(default_factory=list)
    top_k_per_query: int = Field(default=5, gt=0)


class SourceEntry(BaseModel):
    name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    urls: List[str] = Field(default_factory=list)
    source_type: SourceType
    priority: SourcePriority
    search: Optional[SourceSearchConfig] = None
    research_domain: Optional[str] = None
    """Config-driven specialist-agent routing override for a `secondary_sources`
    (or `press_distribution`) entry whose source-type membership would
    otherwise always route to `AgentName.FINANCIAL_NEWS` (see
    `agents.router.ArticleRouter`'s case 3). One of "financial_institutions"
    ("peer"), "ai_technology" ("ai_product"), "startup_funding" ("startup"),
    "analyst", or "financial_news" -- any other value (including `None`,
    the default) is treated identically to "financial_news", i.e. unchanged
    pre-existing behavior. Never consulted for `peers`/`ai_providers`/
    `analysts` entries, whose own registry membership already determines
    routing unambiguously."""


class StartupsDiscovery(BaseModel):
    enabled: bool


class StartupsConfig(BaseModel):
    discovery: StartupsDiscovery
    categories: List[str] = Field(default_factory=list)
    funding_events: List[str] = Field(default_factory=list)
    event_types: List[str] = Field(default_factory=list)


class RetrievalPolicy(BaseModel):
    primary_source_priority: List[str] = Field(default_factory=list)
    secondary_source_priority: List[str] = Field(default_factory=list)
    blocked_behavior: Dict[str, str] = Field(default_factory=dict)


class SourcesRegistry(BaseModel):
    peers: List[SourceEntry]
    ai_providers: List[SourceEntry]
    analysts: List[SourceEntry]
    secondary_sources: List[SourceEntry]
    startups: StartupsConfig
    retrieval_policy: RetrievalPolicy


# ---------------------------------------------------------------------------
# topics.yaml
# ---------------------------------------------------------------------------


class AITopics(BaseModel):
    core: List[str]
    models: List[str]
    infrastructure: List[str]
    enterprise: List[str]
    governance: List[str]


class FinancialServicesTopics(BaseModel):
    core: List[str]
    markets: List[str]
    wealth: List[str]
    risk: List[str]
    operations: List[str]


class TopicsRegistry(BaseModel):
    ai_topics: AITopics
    financial_services: FinancialServicesTopics
    event_types: List[str]
    strategic_themes: List[str]


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Everything a later phase needs from configuration, already validated."""

    settings: Settings
    sources: SourcesRegistry
    topics: TopicsRegistry
