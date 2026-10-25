"""Typed results for every stage of the retrieval pipeline.

These are retrieval-layer concepts only. `ExtractedArticle` is deliberately
*not* the canonical `Article`/`Evidence` model that Phase 2 will introduce —
it just carries what extraction produced, before normalization,
deduplication, and provenance/source-quality classification are applied.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from research_assistant.config.schema import SourceSearchConfig
from research_assistant.models.enums import SourcePriority, SourceType


class SourceCategory(str, Enum):
    """Which section of config/sources.yaml a source came from."""

    PEER = "peers"
    AI_PROVIDER = "ai_providers"
    STARTUP = "startups"
    ANALYST = "analysts"
    SECONDARY = "secondary_sources"


class DiscoveryMethod(str, Enum):
    RSS = "rss"
    ATOM = "atom"
    HTML_LINKS = "html_links"
    SITEMAP = "sitemap"
    SEARCH = "search"
    CONTENT_HUB_API = "content_hub_api"


class RetrievalMethod(str, Enum):
    HTTP = "http"
    BROWSER = "browser"


class CachePolicy(str, Enum):
    """Controls whether `HTTPClient.get()` may serve a cached `RetrievedPage`
    instead of issuing a live request, for the exact same URL/cache-key
    space (`PageCache` has no TTL -- see its own docstring -- so without
    this, a page fetched once is reused forever).

    CACHE: existing behavior, unchanged -- read from cache if present,
    write to cache on success. Used by Morning/Evening/manual runs, and the
    default everywhere, so no existing caller's behavior changes.

    FRESH: never read from the cache -- always issue a live request -- but
    still WRITE the live result to cache on success afterward, so a later
    CACHE-policy run benefits from the now-fresher snapshot too. This is
    CURRENT mode's policy (see `pipeline.nodes.make_retrieve_node`): a
    stale cached source/discovery page can make genuinely current articles
    structurally invisible before ranking, agents, or Ollama ever see them,
    which is incompatible with CURRENT's `[now-24h, now)` contract.
    Browser-rendered fetches (`retrieval.browser.fetch_with_browser`) are
    never cached at all regardless of this policy, so they're already
    "fresh" by construction.
    """

    CACHE = "cache"
    FRESH = "fresh"


class ExtractionMethod(str, Enum):
    TRAFILATURA = "trafilatura"
    BS4_FALLBACK = "bs4_fallback"
    NONE = "none"


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NOT_ARTICLE = "not_article"
    FAILED = "failed"


class SourceDefinition(BaseModel):
    """The retrieval engine's view of one configured source.

    Thinner than `config.schema.SourceEntry` only in that it also records
    which registry category the source came from, so retrieval code never
    needs to reach back into raw config.
    """

    name: str
    domain: str
    urls: List[str] = Field(default_factory=list)
    source_type: SourceType
    priority: SourcePriority
    category: SourceCategory
    search: Optional[SourceSearchConfig] = None


class CandidateURL(BaseModel):
    """A discovered, not-yet-fetched article/page candidate."""

    url: str
    source_name: str
    domain: str
    discovery_method: DiscoveryMethod
    discovered_at: datetime
    title: Optional[str] = None
    published_at: Optional[datetime] = None
    feed_summary: Optional[str] = None
    feed_author: Optional[str] = None


class RetrievalFailure(BaseModel):
    reason: str
    detail: Optional[str] = None


class RetrievedPage(BaseModel):
    """The raw result of fetching a URL, before article extraction."""

    url: str
    final_url: str
    status_code: Optional[int] = None
    retrieved_at: datetime
    method: RetrievalMethod
    content_type: Optional[str] = None
    html: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    from_cache: bool = False
    success: bool = True
    failure: Optional[RetrievalFailure] = None


class ExtractedArticle(BaseModel):
    """Article-shaped content pulled out of a `RetrievedPage`."""

    url: str
    title: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    # Phase 14 date-precision fix: explicit precision hint for `published_at`
    # when extraction knows it is coarser than day-level (currently the
    # ONLY value this is ever set to is "month", from the narrow BofA
    # URL-structure fallback in `extractor.py` -- see that module for why).
    # `None` (the default, and every other extraction path) means "defer to
    # `evidence.normalizer`'s existing exact/date-only heuristic", exactly
    # as before this field existed -- fully backward compatible.
    published_at_precision: Optional[str] = None
    modified_at: Optional[datetime] = None
    publisher: Optional[str] = None
    text: Optional[str] = None
    extraction_method: ExtractionMethod = ExtractionMethod.NONE
    status: ExtractionStatus = ExtractionStatus.FAILED
    metadata: Dict[str, Any] = Field(default_factory=dict)
