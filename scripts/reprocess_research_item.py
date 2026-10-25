"""Recovery path for persisted ResearchItem/Event rows that predate a
deterministic-guardrail code fix (e.g. the `_DATED_ANNOUNCEMENT_PATTERNS`
"has achieved" gap fixed for CURRENT run `7aab105954d38497`'s Bits In Glass
candidate — see `agents/base.py` and `tests/agents/test_grounding_guardrails.py`).

A code fix to `agents.base`/`events.builder` only changes what a FUTURE
pipeline run computes. It does not, by itself, rewrite a ResearchItem/Event
already written to `data/research.db` by an OLDER version of that code —
those rows keep whatever `is_dated_development`/`event_date` the OLD
guardrail logic produced until something explicitly re-derives them. This
script is that explicit, supported "something":

    python scripts/reprocess_research_item.py <research_item_id> [--commit]

What it does, in order, inside one transaction (dry-run by default):
  1. Loads the ResearchItem, its Article, and its Evidence exactly as
     persisted (no live retrieval, no network call — the already-retrieved
     evidence content is reused as-is).
  2. Re-runs the SAME agent (by the item's own stored `agent` field) via
     `ResearchAgent.analyze()` against the CURRENT code — a real Qwen call
     through the already-running local Ollama server, not a hardcoded
     "assume true" shortcut. This naturally re-applies whichever guardrail
     code is live today, whether that fixes this item, leaves it alone, or
     (if a future guardrail become stricter) narrows it further.
  3. Reloads every ResearchItem belonging to the same persisted Event
     (via `event_research_items`), substitutes the freshly reprocessed one,
     and rebuilds the Event (`EventService.build_events`) — so a corrected
     `event_date` is recomputed from the real member set, not guessed.
  4. Upserts the ResearchItem and Event back with their ORIGINAL ids and
     ORIGINAL `run_id` preserved (both ids are content-derived — the
     ResearchItem id from `agent+article_id`, the Event id from the sorted
     member ResearchItem id set — so re-running this against unchanged
     members is idempotent: same ids in, same ids out, no duplicate rows,
     no other run's data touched).
  5. Prints an explicit before/after audit (dated-development, rationale,
     event date) and, unless `--commit` is passed, rolls back instead of
     committing — a dry run by default.

By default this does NOT touch the digest row (`digests`/`digest_events`)
or the owning `pipeline_runs` row — only the ResearchItem/Event tables.
Pass `--rebuild-digest` to go one step further and also rebuild THIS RUN's
own digest from its own (now-corrected) events, still scoped strictly to
`research_item.run_id`/`event.run_id`, still only reachable with
`--commit`: it re-derives the digest exactly the way `digest_generation`
normally would (same period bounds the run itself recorded, same in-period
filter), upserts it under its own content-derived id (a different event
set deterministically produces a different digest id — this is a NEW
digest row, not a mutation of the old one's content), and repoints
`pipeline_runs.digest_id`/`digest_persistence_status` at it. The OLD digest
row is left in place (unreferenced, but not deleted) as an audit trail of
what the run originally produced before this reprocessing.

## Finding OTHER stale records (not just a known id)

    python scripts/reprocess_research_item.py --audit [--database PATH]

Read-only, deterministic, no LLM call: scans every persisted ResearchItem
that has evidence and flags one as SUSPECTED STALE only when a check that
needs no model call -- re-running the current, already-deployed
`_contains_dated_announcement_language` pattern set against the item's own
stored `Evidence.content` -- disagrees with what was persisted. Because
`_resolve_dated_development` only ever NARROWS the LLM's True to False
(never promotes False to True, see its own docstring), a stored
`is_dated_development=False` whose own evidence text the CURRENT pattern
set would now corroborate is exactly the shape of staleness this project
has already hit once (Bits In Glass, run `7aab105954d38497`) and may hit
again whenever `_DATED_ANNOUNCEMENT_PATTERNS` grows. This can only ever
suggest a record for human review and `reprocess(..., commit=True)` --
it never re-analyzes or mutates anything itself, so it is safe to run
against the live production database at any time, including from a
read path (it opens no write transaction).

A second, independent check flags a persisted Event whose `event_date` is
None while one of its OWN member ResearchItems has both a
`publication_date` and `is_dated_development=True` on file -- that
combination can only arise if the Event was built/persisted before its
member's ResearchItem was last corrected (e.g. a partial/interrupted
reprocessing), never from current, consistent code.

Neither check touches, infers, or re-litigates an agent's RELEVANCE
judgment (`relevance_label`) -- only the deterministic, evidence-grounded
dated-development/event-date consistency a pure function can already
verify without calling Qwen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from research_assistant.agents.ai_product import AIProductAgent
from research_assistant.agents.analyst import AnalystAgent
from research_assistant.agents.base import _contains_dated_announcement_language
from research_assistant.agents.financial_news import FinancialNewsAgent
from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.startup import StartupAgent
from research_assistant.config import get_config
from research_assistant.events.service import EventService
from research_assistant.llm.client import create_ollama_client
from research_assistant.models.digest import DigestType
from research_assistant.models.enums import AgentName, DigestPersistenceStatus
from research_assistant.reporting.digest_service import DigestService
from research_assistant.storage.database import Database
from research_assistant.storage.repositories.article import ArticleRepository
from research_assistant.storage.repositories.digest import DigestRepository
from research_assistant.storage.repositories.event import EventRepository
from research_assistant.storage.repositories.evidence import EvidenceRepository
from research_assistant.storage.repositories.research_item import ResearchItemRepository
from research_assistant.storage.serde import dt_to_str

_AGENT_CLASSES = {
    AgentName.AI_PRODUCT: AIProductAgent,
    AgentName.STARTUP: StartupAgent,
    AgentName.PEER: PeerAgent,
    AgentName.ANALYST: AnalystAgent,
    AgentName.FINANCIAL_NEWS: FinancialNewsAgent,
}


def _snapshot(item) -> dict:
    return {
        "is_dated_development": item.is_dated_development,
        "development_date_rationale": item.development_date_rationale,
        "relevance_label": item.relevance.label.value,
    }


def _event_snapshot(event) -> Optional[dict]:
    if event is None:
        return None
    return {
        "event_date": event.event_date.isoformat() if event.event_date else None,
        "relevance_score": event.relevance_score,
        "confidence_score": event.confidence_score,
    }


def _rebuild_run_digest(conn, config, run_id: str) -> dict:
    """Rebuilds and repoints THIS RUN's own digest from its own (now
    corrected) events, using the exact same period bounds and in-period
    filter `digest_generation` applies during a normal run -- never a new
    period, never another run's events."""
    run_row = conn.execute(
        "SELECT digest_type, period_start, period_end, digest_id AS old_digest_id "
        "FROM pipeline_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if run_row is None:
        raise SystemExit(f"No pipeline_runs row found for run_id={run_id!r}")

    event_repo = EventRepository()
    events = event_repo.get_by_run_id(conn, run_id)

    from research_assistant.storage.serde import dt_from_str

    period_start = dt_from_str(run_row["period_start"])
    period_end = dt_from_str(run_row["period_end"])
    in_period_events = [
        event for event in events
        if event.event_date is not None and period_start <= event.event_date < period_end
    ]

    digest_type = DigestType(run_row["digest_type"])
    digest = DigestService(config).build_digest(
        in_period_events, digest_type=digest_type, period_start=period_start, period_end=period_end
    )

    DigestRepository().upsert(conn, digest)
    new_status = (
        DigestPersistenceStatus.PERSISTED.value
        if in_period_events
        else DigestPersistenceStatus.NO_QUALIFYING_EVENTS.value
    )
    conn.execute(
        "UPDATE pipeline_runs SET digest_id = ?, digest_persistence_status = ? WHERE run_id = ?",
        (digest.id, new_status, run_id),
    )
    return {
        "old_digest_id": run_row["old_digest_id"],
        "new_digest_id": digest.id,
        "new_digest_event_count": len(in_period_events),
        "new_digest_persistence_status": new_status,
    }


def audit_stale_records(database_path: Path) -> list:
    """Read-only scan for suspected-stale ResearchItem/Event rows -- see
    the module docstring's "Finding OTHER stale records" section. Returns a
    list of dicts, each `{"research_item_id", "event_id", "reason"}`; never
    opens a write transaction, never calls an LLM."""
    database = Database(database_path)
    item_repo = ResearchItemRepository()
    evidence_repo = EvidenceRepository()
    event_repo = EventRepository()

    findings: list = []
    conn = database.connect()
    try:
        item_rows = conn.execute("SELECT id FROM research_items").fetchall()
        for row in item_rows:
            item = item_repo.get(conn, row["id"])
            if item is None or not item.evidence:
                continue
            if item.is_dated_development:
                continue  # only a stored False can be stale (narrowing is one-directional)
            if not item.development_date_rationale or "Deterministic override" not in item.development_date_rationale:
                # No override ever fired for this item -- the MODEL's own
                # judgment was already False (`_resolve_dated_development`
                # returns immediately in that branch without even reading
                # the evidence text, see its first `if`), so a text-pattern
                # match here proves nothing about staleness; re-flagging it
                # would be a false positive (confirmed real case: "Satya
                # Nadella reinvented Microsoft..." -- a retrospective piece
                # whose body happens to contain unrelated announcement-
                # shaped language, but whose False came from the model, not
                # a guardrail that might now disagree with itself).
                continue
            evidence = evidence_repo.get(conn, item.evidence[0])
            if evidence is None:
                continue
            if _contains_dated_announcement_language(evidence.content or ""):
                findings.append({
                    "research_item_id": item.id,
                    "run_id": item.run_id,
                    "reason": (
                        "is_dated_development=False is stored, but the CURRENT deterministic "
                        "pattern set now finds explicit announcement language in this item's own "
                        "stored evidence text -- re-running analyze() may recover a dated event "
                        "(exactly the Bits In Glass shape). Review, then "
                        "`reprocess(..., commit=True)` if appropriate."
                    ),
                })

        event_rows = conn.execute("SELECT id FROM events WHERE event_date IS NULL").fetchall()
        for row in event_rows:
            event_id = row["id"]
            member_ids = event_repo.get_research_item_ids(conn, event_id)
            for member_id in member_ids:
                member = item_repo.get(conn, member_id)
                if member is not None and member.is_dated_development and member.publication_date is not None:
                    findings.append({
                        "research_item_id": member_id,
                        "event_id": event_id,
                        "reason": (
                            f"Event {event_id!r} has event_date=None, but its own member "
                            f"ResearchItem {member_id!r} is_dated_development=True with a known "
                            "publication_date -- the Event is out of sync with its member and "
                            "should be rebuilt (reprocess that ResearchItem)."
                        ),
                    })
                    break
    finally:
        conn.close()
    return findings


def reprocess(
    database_path: Path,
    research_item_id: str,
    *,
    commit: bool,
    rebuild_digest: bool = False,
    llm_client=None,
) -> dict:
    """`llm_client`, when supplied, replaces the real Ollama call for this
    one reprocessing (tests inject a fake here so they never need a live
    model); production callers (the CLI below) always omit it and get the
    real `create_ollama_client(config)`."""
    config = get_config()
    database = Database(database_path)
    item_repo = ResearchItemRepository()
    article_repo = ArticleRepository()
    evidence_repo = EvidenceRepository()
    event_repo = EventRepository()

    conn = database.connect()
    try:
        before_item = item_repo.get(conn, research_item_id)
        if before_item is None:
            raise SystemExit(f"No ResearchItem found with id={research_item_id!r}")

        article = article_repo.get(conn, before_item.article_id)
        if article is None:
            raise SystemExit(f"No Article found with id={before_item.article_id!r} (FK integrity broken)")
        if not before_item.evidence:
            raise SystemExit(f"ResearchItem {research_item_id!r} has no evidence references")
        evidence = evidence_repo.get(conn, before_item.evidence[0])
        if evidence is None:
            raise SystemExit(f"No Evidence found with id={before_item.evidence[0]!r} (FK integrity broken)")

        agent_cls = _AGENT_CLASSES.get(before_item.agent)
        if agent_cls is None:
            raise SystemExit(f"No agent class mapped for AgentName={before_item.agent!r}")

        active_llm_client = llm_client if llm_client is not None else create_ollama_client(config)
        agent = agent_cls(active_llm_client, config)

        reanalyzed = agent.analyze(article, evidence)
        if reanalyzed is None:
            raise SystemExit(
                f"Re-analysis rejected {research_item_id!r} as IRRELEVANT under current code — "
                "this is a real outcome change, not a bug in this script. Inspect manually before "
                "deciding whether the Event/digest should drop this item."
            )
        if reanalyzed.id != before_item.id:
            raise SystemExit(
                f"Re-analysis produced a different ResearchItem id ({reanalyzed.id!r} != "
                f"{before_item.id!r}) — article/agent identity changed unexpectedly; aborting "
                "rather than risk creating a duplicate."
            )

        # Preserve ownership: analyze() never sets run_id (it's attached by
        # the pipeline, not the agent) — carry the ORIGINAL run forward
        # explicitly so this reprocessed row still belongs to the same run.
        after_item = reanalyzed.model_copy(update={"run_id": before_item.run_id})

        owning_event_row = conn.execute(
            "SELECT event_id FROM event_research_items WHERE research_item_id = ?",
            (research_item_id,),
        ).fetchone()

        before_event = event_repo.get(conn, owning_event_row["event_id"]) if owning_event_row else None
        after_event = None
        if owning_event_row is not None:
            member_ids = event_repo.get_research_item_ids(conn, owning_event_row["event_id"])
            member_items = []
            for member_id in member_ids:
                member_items.append(after_item if member_id == research_item_id else item_repo.get(conn, member_id))
            missing = [mid for mid, it in zip(member_ids, member_items) if it is None]
            if missing:
                raise SystemExit(f"Event {owning_event_row['event_id']!r} has dangling member ids: {missing}")

            rebuilt_events = EventService(config).build_events(member_items)
            matching = [e for e in rebuilt_events if e.id == owning_event_row["event_id"]]
            if not matching:
                raise SystemExit(
                    f"Rebuilt Event id ({[e.id for e in rebuilt_events]}) does not match the original "
                    f"({owning_event_row['event_id']!r}) — member set identity changed unexpectedly; aborting."
                )
            after_event = matching[0].model_copy(update={"run_id": before_event.run_id if before_event else None})

        item_repo.upsert(conn, after_item)
        if after_event is not None:
            event_repo.upsert(conn, after_event)

        audit = {
            "research_item_id": research_item_id,
            "before": _snapshot(before_item),
            "after": _snapshot(after_item),
            "event_id": owning_event_row["event_id"] if owning_event_row else None,
            "event_before": _event_snapshot(before_event),
            "event_after": _event_snapshot(after_event),
            "committed": commit,
            "digest_rebuilt": False,
        }

        if rebuild_digest and before_item.run_id is not None:
            audit["digest"] = _rebuild_run_digest(conn, config, before_item.run_id)
            audit["digest_rebuilt"] = True

        if commit:
            conn.commit()
        else:
            conn.rollback()
        return audit
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("research_item_id", nargs="?", help="Omit when using --audit")
    parser.add_argument(
        "--database", type=Path, default=Path("data/research.db"),
        help="Path to the production SQLite database (default: data/research.db)",
    )
    parser.add_argument(
        "--audit", action="store_true",
        help="Read-only: list suspected-stale records instead of reprocessing one by id.",
    )
    parser.add_argument(
        "--commit", action="store_true",
        help="Actually persist the change. Without this flag, the transaction is rolled back (dry run).",
    )
    parser.add_argument(
        "--rebuild-digest", action="store_true",
        help="Also rebuild this run's own digest from its corrected events and repoint pipeline_runs at it.",
    )
    args = parser.parse_args()

    if args.audit:
        findings = audit_stale_records(args.database)
        print(json.dumps(findings, indent=2))
        print(f"\n{len(findings)} suspected-stale record(s) found (read-only scan)", file=sys.stderr)
        return

    if not args.research_item_id:
        parser.error("research_item_id is required unless --audit is given")

    audit = reprocess(args.database, args.research_item_id, commit=args.commit, rebuild_digest=args.rebuild_digest)
    print(json.dumps(audit, indent=2))
    if not args.commit:
        print("\n(dry run — no changes were committed; re-run with --commit to persist)", file=sys.stderr)


if __name__ == "__main__":
    main()
