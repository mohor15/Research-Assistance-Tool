"""Digest generation — a collection of ranked Events -> a structured Digest.

`DigestService` (Phase 6) is implemented; report/report-agent code itself
lives in `research_assistant.agents.report` alongside the other agents,
consistent with Phase 3/4's convention that agents live in `agents/`.
"""

from research_assistant.reporting.digest_service import DigestService
from research_assistant.reporting.themes import build_theme_candidates

__all__ = ["DigestService", "build_theme_candidates"]
