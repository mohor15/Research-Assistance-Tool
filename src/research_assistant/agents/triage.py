"""Batched LLM triage -- CURRENT's bounded funnel (see `pipeline.nodes`'s
CURRENT-mode `agent_analysis`): a cheap, single Ollama call screens SEVERAL
candidates at once ("is this substantive and relevant, and what broad kind
of development is it?") instead of every specialist agent independently
running a full deep-analysis call on every one of its routed candidates.

Deliberately NOT a specialist analysis: no entities/people/products/deal
value/strategic implications are asked for or produced here -- only a
relevant/not-relevant screen plus a broad event-type/category guess, used
purely to decide which small set of candidates earns a real deep-analysis
call (`agents.base.ResearchAgent.analyze`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from research_assistant.agents.prompt_loader import load_prompt
from research_assistant.llm.client import LLMClient
from research_assistant.models.article import Article
from research_assistant.models.enums import AgentName, EventType
from research_assistant.models.evidence import Evidence
from research_assistant.models.research_item import TopicCategory

TRIAGE_SYSTEM_PROMPT_FILE = "triage_system.md"

# Mirrors `agents.base.MAX_EVIDENCE_CONTENT_CHARS`'s reasoning, scaled down
# hard: this prompt carries MULTIPLE candidates at once, so each one's
# snippet must stay small or a full batch would itself become as expensive
# as several single deep-analysis calls, defeating the point of batching.
# A title plus opening paragraphs is enough for a fast relevant/not screen
# (the full body is only read later, during deep analysis of the winners).
MAX_TRIAGE_SNIPPET_CHARS = 600


@dataclass(frozen=True)
class TriageCandidate:
    """One pool entry offered to a triage batch -- `candidate_id` is the
    Article's own id, reused unchanged so the result maps back without any
    extra bookkeeping."""

    candidate_id: str
    article: Article
    evidence: Evidence
    primary_agent: AgentName


class TriageItemResult(BaseModel):
    """No free-text `reason` field (Phase 17 triage performance fix):
    confirmed unused by every consumer -- `_run_current_triage`
    (pipeline/nodes.py) only reads `.relevant` to decide which candidates
    advance, and `_run_current_deep_analysis` discards the whole triage
    result (`_triage_result`, unused) once a candidate wins. Dropping it
    shrinks the per-candidate generation the LLM must produce under
    grammar-constrained decoding, directly reducing triage latency."""

    candidate_id: str
    relevant: bool
    event_type: Optional[EventType] = None
    category: Optional[TopicCategory] = None
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_percentage_confidence(cls, value):
        """Confirmed real production failure (2026-10-07 CURRENT run
        `2a55bf02ff1b7d1f`): qwen2.5:3b sometimes returns `confidence` on a
        0-100 scale (e.g. 90, 95, 75, 25, 5) instead of the documented
        0.0-1.0 one, which previously failed this field's `ge=0.0, le=1.0`
        constraint outright and discarded the ENTIRE triage batch (13/24
        real candidates lost to this in that run) -- not just the one
        malformed item. A numeric value strictly greater than 1 and at most
        100 is interpreted as a percentage and divided by 100 (90 -> 0.90,
        5 -> 0.05); a value already in [0, 1] passes through unchanged.
        Anything else (negative, or still >100 after this check) is left
        alone and still rejected by the field's own `ge=0.0, le=1.0`
        constraint below -- this never silently accepts an out-of-range
        value, it only rescales the one specific, observed misinterpretation."""
        if isinstance(value, (int, float)) and 1 < value <= 100:
            return value / 100
        return value


class TriageBatchResponse(BaseModel):
    results: List[TriageItemResult] = Field(default_factory=list)


def _snippet(article: Article, evidence: Evidence) -> str:
    body = evidence.content or ""
    if len(body) > MAX_TRIAGE_SNIPPET_CHARS:
        body = body[:MAX_TRIAGE_SNIPPET_CHARS] + " [...]"
    return f"Title: {article.title or 'unknown'}\nSource: {article.source_name}\nSnippet: {body}"


def build_triage_prompt(batch: List[TriageCandidate], *, domain_hint: Optional[str] = None) -> str:
    """`domain_hint` (source-aware-routing refactor): every candidate in
    `batch` already shares the same `primary_agent` (see
    `pipeline.nodes._run_current_triage_by_domain`, the only real caller
    that ever passes batches this way) -- a one-line description of that
    specialist domain, prepended to the prompt, so this cheap triage call
    reasons about a single, named domain instead of a domain-blind mixed
    pool. `None` (the default, and every pre-existing call site) omits it
    entirely, preserving the original prompt text exactly."""
    entries = "\n\n".join(
        f"--- Candidate {c.candidate_id} ---\n{_snippet(c.article, c.evidence)}" for c in batch
    )
    candidate_ids = ", ".join(c.candidate_id for c in batch)
    domain_line = f"Domain focus for this batch: {domain_hint}\n\n" if domain_hint else ""
    return (
        f"{domain_line}"
        "Assess EACH of the following candidates. Return a `results` array "
        f"with exactly one entry per candidate_id ({candidate_ids}), in any "
        "order, each matching the provided response schema.\n\n"
        f"{entries}\n"
    )


def run_triage_batch(
    llm_client: LLMClient, batch: List[TriageCandidate], *, domain_hint: Optional[str] = None,
) -> Dict[str, TriageItemResult]:
    """Raises on LLM failure (timeout, connection error, schema validation)
    -- the caller decides how to charge that against the triage budget and
    move on; this function never retries and never silently drops the
    batch itself."""
    system_prompt = load_prompt(TRIAGE_SYSTEM_PROMPT_FILE)
    prompt = build_triage_prompt(batch, domain_hint=domain_hint)
    response = llm_client.generate_structured(system=system_prompt, prompt=prompt, schema=TriageBatchResponse)
    return {result.candidate_id: result for result in response.results}
