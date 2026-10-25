"""The canonical Evidence model — source-grounded content derived from an Article.

Every material claim a future agent (Phase 3+) produces must ultimately be
traceable back to an `Evidence` object. This model therefore holds only
what was actually retrieved and extracted, plus its provenance — never an
AI-generated summary or interpretation. Evidence is never constructed for
an Article with no usable text (see
`research_assistant.evidence.article_to_evidence`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from research_assistant.models.enums import SourcePriority, SourceType
from research_assistant.retrieval.models import ExtractionMethod


class Evidence(BaseModel):
    evidence_id: str
    article_id: str

    source_url: str
    """The URL this evidence was actually retrieved from (`Article.url`)."""
    canonical_url: str
    """The normalized identity used for duplicate detection (`Article.canonical_url`)."""

    publisher: Optional[str] = None
    title: Optional[str] = None

    publication_date: Optional[datetime] = None
    retrieved_at: datetime

    content: str
    """Non-empty, cleaned article text — the source-grounded material itself."""
    content_hash: str

    source_type: SourceType
    source_priority: SourcePriority
    extraction_method: ExtractionMethod

    evidence_span: Optional[str] = None
    """Reserved for a future phase's claim-level spans/quotes; always None in Phase 2."""

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("publication_date", "retrieved_at")
    @classmethod
    def _require_timezone_aware(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is not None and value.tzinfo is None:
            raise ValueError("datetime fields must be timezone-aware")
        return value

    @field_validator("content")
    @classmethod
    def _content_must_be_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Evidence.content must be non-empty — construct it only from an Article with usable text")
        return value
