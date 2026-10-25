"""SQLite connection management and deterministic, idempotent schema init.

`initialize()` only ever uses `CREATE TABLE/INDEX IF NOT EXISTS` — running it
against an existing database never drops or recreates existing tables or
data. A tiny `schema_meta` table records a schema version; `initialize()`
bumps it forward (never backward) after applying the schema, which is all
the "migration" this project needs since every change so far has been
purely additive (new tables/columns via `IF NOT EXISTS`, never a drop or
rename). Phase 8 bumped this from version 1 (Phase 7) to version 2 by
adding the `pipeline_runs` table — still no separate migration framework.
Phase 13B bumped it to version 3: `CREATE TABLE IF NOT EXISTS` alone can't
add a column to a table that already exists (it's a no-op there), so
`initialize()` additionally runs a small guarded `_add_column_if_missing`
step for the two new `research_items` columns -- still no drop/rename,
still safe to run repeatedly against an already-populated database. Phase
14 bumped it to version 4: new `scheduler_state` table (additive, `CREATE
TABLE IF NOT EXISTS`) plus 7 new `_add_column_if_missing` columns on
`pipeline_runs` (execution_mode/period_start/period_end/digest_attempted/
digest_persisted/digest_persistence_status/digest_id) -- existing rows get
safe, documented defaults ('manual', 'not_attempted') rather than a guess.
Bumped to version 9: email delivery (two new, purely additive tables --
`email_deliveries` and `email_settings` -- see their own comments below).
Bumped to version 10: the CURRENT candidate backlog (one new, purely
additive table, `candidate_backlog` -- see its own comment below) so a
candidate that runs out of triage/deep-analysis budget in one run is
durably retried by a later run instead of being silently lost.
Bumped to version 11 (pre-live-scan reliability patch): one new, additive
`candidate_backlog.defer_count` column (existing rows default to 0) --
distinguishes a genuine lost-processing-opportunity-to-budget deferral
from a real analysis attempt, so the backlog's fairness/aging mechanism
actually strengthens across repeated pure-budget deferrals, not only
repeated analysis failures.
Bumped to version 13 (on-demand Tier B newsletter reports): one new,
additive `tier_b_reports` table. A Tier B article has no `Event`, so the
existing `reports` table (`event_id TEXT PRIMARY KEY REFERENCES
events(id)`) genuinely cannot hold one -- that FK is live and real
(confirmed by reading this schema directly, not assumed), unlike
`newsletter_selections`'s now-removed one -- so this is a small, separate,
purely additive table rather than an unsafe reuse of `reports`. See the
table's own comment below.

Junction tables (`event_research_items`, `event_articles`, `digest_events`)
normalize what are otherwise flat lists of foreign-key ids on the Pydantic
models — genuinely nested sub-objects (metadata dicts, source references,
scoring breakdowns, cluster metadata, report details/sources, digest
themes) are stored as JSON columns instead, per Phase 7's own guidance to
reserve JSON for cases where normalization would add unnecessary
complexity.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

CURRENT_SCHEMA_VERSION = 13

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    domain TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_priority TEXT NOT NULL,
    publisher TEXT,
    title TEXT,
    author TEXT,
    publication_date TEXT,
    publication_date_precision TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    raw_text TEXT,
    clean_text TEXT,
    content_hash TEXT,
    extraction_method TEXT NOT NULL,
    extraction_status TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_articles_canonical_url ON articles(canonical_url);
CREATE INDEX IF NOT EXISTS idx_articles_content_hash ON articles(content_hash);
CREATE INDEX IF NOT EXISTS idx_articles_publication_date ON articles(publication_date);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    article_id TEXT NOT NULL REFERENCES articles(id),
    source_url TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    publisher TEXT,
    title TEXT,
    publication_date TEXT,
    retrieved_at TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_priority TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    evidence_span TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_evidence_article_id ON evidence(article_id);

CREATE TABLE IF NOT EXISTS research_items (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    article_id TEXT NOT NULL REFERENCES articles(id),
    publisher TEXT,
    title TEXT,
    url TEXT NOT NULL,
    publication_date TEXT,
    company TEXT NOT NULL,
    category TEXT NOT NULL,
    event_type TEXT,
    summary TEXT,
    is_dated_development INTEGER NOT NULL DEFAULT 1,
    development_date_rationale TEXT,
    entities_json TEXT NOT NULL DEFAULT '[]',
    people_json TEXT NOT NULL DEFAULT '[]',
    products_json TEXT NOT NULL DEFAULT '[]',
    deal_value TEXT,
    partnerships_json TEXT NOT NULL DEFAULT '[]',
    technologies_json TEXT NOT NULL DEFAULT '[]',
    geography_json TEXT NOT NULL DEFAULT '[]',
    relevance_label TEXT NOT NULL,
    relevance_reason TEXT NOT NULL,
    financial_services_relevance_level TEXT NOT NULL,
    financial_services_relevance_rationale TEXT NOT NULL,
    nomura_relevance_level TEXT NOT NULL,
    nomura_relevance_rationale TEXT NOT NULL,
    strategic_significance_level TEXT NOT NULL,
    strategic_significance_rationale TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
    source_type TEXT NOT NULL,
    source_priority TEXT NOT NULL,
    analyst_findings_json TEXT,
    grounding_type TEXT NOT NULL DEFAULT 'content',
    relevance_override TEXT
);
CREATE INDEX IF NOT EXISTS idx_research_items_article_id ON research_items(article_id);
CREATE INDEX IF NOT EXISTS idx_research_items_agent ON research_items(agent);
CREATE INDEX IF NOT EXISTS idx_research_items_company ON research_items(company);
CREATE INDEX IF NOT EXISTS idx_research_items_event_type ON research_items(event_type);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    canonical_title TEXT NOT NULL,
    event_type TEXT,
    primary_company TEXT NOT NULL,
    related_companies_json TEXT NOT NULL DEFAULT '[]',
    event_date TEXT,
    description TEXT,
    primary_source_json TEXT NOT NULL,
    secondary_sources_json TEXT NOT NULL DEFAULT '[]',
    relevance_score REAL NOT NULL,
    confidence_score REAL NOT NULL,
    scoring_breakdown_json TEXT NOT NULL,
    cluster_metadata_json TEXT NOT NULL,
    grounding_type TEXT NOT NULL DEFAULT 'content'
);
CREATE INDEX IF NOT EXISTS idx_events_event_date ON events(event_date);
CREATE INDEX IF NOT EXISTS idx_events_primary_company ON events(primary_company);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_relevance_score ON events(relevance_score);
CREATE INDEX IF NOT EXISTS idx_events_confidence_score ON events(confidence_score);

CREATE TABLE IF NOT EXISTS event_research_items (
    event_id TEXT NOT NULL REFERENCES events(id),
    research_item_id TEXT NOT NULL REFERENCES research_items(id),
    position INTEGER NOT NULL,
    PRIMARY KEY (event_id, research_item_id)
);
CREATE INDEX IF NOT EXISTS idx_eri_research_item_id ON event_research_items(research_item_id);

CREATE TABLE IF NOT EXISTS event_articles (
    event_id TEXT NOT NULL REFERENCES events(id),
    article_id TEXT NOT NULL REFERENCES articles(id),
    position INTEGER NOT NULL,
    PRIMARY KEY (event_id, article_id)
);
CREATE INDEX IF NOT EXISTS idx_ea_article_id ON event_articles(article_id);

CREATE TABLE IF NOT EXISTS reports (
    event_id TEXT PRIMARY KEY REFERENCES events(id),
    status TEXT NOT NULL,
    title TEXT,
    opening_context TEXT,
    details_json TEXT,
    strategic_implications TEXT,
    nomura_opportunity TEXT,
    sources_json TEXT,
    relevance_score REAL,
    confidence_score REAL,
    failure_reason TEXT,
    failure_detail TEXT
);

CREATE TABLE IF NOT EXISTS digests (
    id TEXT PRIMARY KEY,
    digest_type TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    themes_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_digests_digest_type ON digests(digest_type);
CREATE INDEX IF NOT EXISTS idx_digests_generated_at ON digests(generated_at);

CREATE TABLE IF NOT EXISTS digest_events (
    digest_id TEXT NOT NULL REFERENCES digests(id),
    event_id TEXT NOT NULL REFERENCES events(id),
    position INTEGER NOT NULL,
    PRIMARY KEY (digest_id, event_id)
);
CREATE INDEX IF NOT EXISTS idx_de_event_id ON digest_events(event_id);

-- Phase 8: lightweight run observability. run_id is deterministic (derived
-- from digest_type + period bounds — see pipeline.runner.generate_run_id),
-- so a scheduler retry/restart upserts the same row rather than duplicating it.
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    digest_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    status TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL DEFAULT '[]',
    counts_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_digest_type ON pipeline_runs(digest_type);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started_at ON pipeline_runs(started_at);

-- Phase 14: a single-row table recording the persistent scheduler
-- process's own observable state -- never inferred from `pipeline_runs`
-- alone (a run can be hours old while the scheduler itself is actually
-- fine, or vice versa: the scheduler can be dead while an old run sits in
-- the table looking superficially healthy). `id` is always the literal
-- string 'singleton'; there is exactly one scheduler per deployment.
CREATE TABLE IF NOT EXISTS scheduler_state (
    id TEXT PRIMARY KEY,
    started_at TEXT,
    last_heartbeat_at TEXT,
    next_morning_run_at TEXT,
    next_evening_run_at TEXT,
    last_scheduled_run_id TEXT,
    last_scheduled_execution_mode TEXT,
    last_scheduled_digest_type TEXT,
    last_scheduled_started_at TEXT,
    last_scheduled_finished_at TEXT,
    last_scheduled_status TEXT
);

-- Phase 16 (user-facing newsletter curation): which CURRENT events the
-- user has selected for the newsletter. Deliberately a bare id + timestamp
-- pointer table -- it never stores a copy of the Event itself, so the
-- underlying intelligence is never mutated/duplicated by a user's
-- selection. PRIMARY KEY on event_id makes a duplicate selection
-- structurally impossible (an "add" is an upsert, never a second row).
--
-- Two-tier CURRENT digest: `event_id` now also holds a Tier B selection
-- (a `tb:`-prefixed article id -- see `ui.dashboard_data.
-- TIER_B_SELECTION_PREFIX`), which is NOT a row in `events` -- so the
-- original `REFERENCES events(id)` foreign key (present through schema
-- version 11) is intentionally absent from a freshly-created database.
-- An EXISTING database created before this change still has that FK;
-- `_drop_newsletter_selections_event_fk` below rebuilds the table without
-- it, exactly once, idempotently, without touching any existing selection.
CREATE TABLE IF NOT EXISTS newsletter_selections (
    event_id TEXT PRIMARY KEY,
    selected_at TEXT NOT NULL
);

-- Phase 17: explicit application-level mutex for CURRENT (Section 3) --
-- CURRENT has no scheduled variant, so it is the one digest type an
-- operator can accidentally trigger twice (e.g. a double click on "Run
-- Live Current Scan now"), overwhelming the single local Ollama instance.
-- Exactly one row ever exists (id='singleton'); its presence means a
-- CURRENT run is currently in flight and names which run_id holds it.
CREATE TABLE IF NOT EXISTS current_run_lock (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    owner_pid INTEGER
);

-- Email delivery (automatic CURRENT/Morning/Evening digest delivery +
-- manual newsletter sends). One row per (digest, recipient, kind) delivery
-- attempt-group -- never a copy of the digest/newsletter content itself,
-- and never a credential. `kind='automatic'` rows are deduplicated by the
-- partial unique index below (the idempotency key the feature's design
-- requires: same digest + same recipient must never be auto-delivered
-- twice); `kind='manual'` rows deliberately allow multiple rows for the
-- same (digest_id, recipient) pair, since a user may intentionally resend
-- the same newsletter -- accidental duplicate submissions are instead
-- prevented at the UI layer (see `ui/app.py`'s newsletter send controls).
-- `digest_id` is nullable: an automatic delivery always has one (it is
-- delivering a specific persisted Digest), but a manual newsletter send is
-- built from a user's arbitrary cross-digest event selection
-- (`newsletter_selections`), not one single Digest row, so it is recorded
-- with `digest_id = NULL` and `digest_type = 'newsletter'`.
CREATE TABLE IF NOT EXISTS email_deliveries (
    id TEXT PRIMARY KEY,
    digest_id TEXT REFERENCES digests(id),
    digest_type TEXT NOT NULL,
    recipient TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_attempt_at TEXT,
    created_at TEXT NOT NULL,
    sent_at TEXT,
    failure_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_email_deliveries_digest_id ON email_deliveries(digest_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_email_deliveries_auto_dedup
    ON email_deliveries(digest_id, recipient) WHERE kind = 'automatic';

-- The user-editable default-recipient / auto-delivery-enabled preference
-- (Section 5): a single-row singleton table, exactly like `scheduler_state`
-- and `current_run_lock`, so the UI-configured recipient survives a
-- process restart rather than living only in transient Streamlit session
-- state. Absent (no row yet), callers fall back to
-- `config.settings.email.default_recipient`/`.enabled` from settings.yaml.
CREATE TABLE IF NOT EXISTS email_settings (
    id TEXT PRIMARY KEY,
    default_recipient TEXT NOT NULL,
    auto_delivery_enabled INTEGER NOT NULL DEFAULT 1
);

-- CURRENT candidate backlog (Section 2/3 of the "maximise relevant
-- intelligence without losing candidates" fix): a candidate that passes
-- every deterministic gate and has not been conclusively rejected, but
-- whose own run ran out of triage/deep-analysis budget before reaching it,
-- is never discarded -- it is upserted here and durably retried by a LATER
-- run instead. One row per Article (never a duplicate of the Article/
-- Evidence content itself -- `article_id` is a stable reference into the
-- already-persisted `articles`/`evidence` tables, which already retain
-- every retrieved article+evidence regardless of whether a ResearchItem
-- resulted). `status` is this candidate's current disposition and is NEVER
-- a guess: every value written here corresponds to an explicit, observed
-- pipeline outcome (see `pipeline.nodes`'s backlog-update construction).
-- Claim safety is inherited for free from the existing CURRENT
-- `current_run_lock` singleton (only one CURRENT run is ever in flight),
-- so this table needs no separate claim/lease protocol of its own.
CREATE TABLE IF NOT EXISTS candidate_backlog (
    article_id TEXT PRIMARY KEY REFERENCES articles(id),
    agent TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    triage_relevant INTEGER,
    triage_confidence REAL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    -- Reliability patch (pre-live-scan audit finding): a pure budget-skip
    -- deferral (never an actual deep-analysis attempt) never touched
    -- `attempt_count`, so the aging-fairness boost could not improve a
    -- repeatedly-budget-deferred candidate's priority across runs. Counts
    -- ONLY genuine lost processing opportunities due to a budget limit
    -- (triage-pool bound or deep-analysis selection/budget) -- never
    -- incremented merely because a candidate was discovered/loaded, never
    -- for an explicit irrelevant/date-gate rejection or a real attempt
    -- (those already have `attempt_count`). See `pipeline.nodes._compute_
    -- backlog_updates` and `_aging_boosted_rank_key`.
    defer_count INTEGER NOT NULL DEFAULT 0,
    first_seen_run_id TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_attempt_run_id TEXT,
    last_attempt_at TEXT,
    research_item_id TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidate_backlog_status ON candidate_backlog(status);
CREATE INDEX IF NOT EXISTS idx_candidate_backlog_first_seen_at ON candidate_backlog(first_seen_at);

-- On-demand Tier B newsletter reports (schema version 13): a structured
-- report generated for ONE Tier B article (see `candidate_backlog`), ONLY
-- when a user selects it for the newsletter and explicitly generates one
-- -- never during a CURRENT scan. Keyed by `article_id`, never a fake
-- Event -- Tier B articles have none by definition, and `reports.event_id`
-- carries a real, live `REFERENCES events(id)` foreign key (unlike
-- `newsletter_selections`'s now-removed one), so reusing that table for a
-- non-Event id would be unsafe; this is a small, separate, purely
-- additive table instead. `content_hash` is the source article's
-- `articles.content_hash` at the time of generation (the exact text
-- version the report was grounded in) -- a later re-extraction producing
-- different/better text changes the hash, and the resolver (`reporting.
-- tier_b_reports`) treats that as a cache miss, never a stale hit.
-- `schema_version` is this feature's own prompt/output-schema version
-- (`reporting.tier_b_reports.TIER_B_REPORT_SCHEMA_VERSION`), so a future
-- prompt/schema change invalidates old cached rows without a migration.
-- `status='failure'` rows are NEVER treated as a valid cache hit by the
-- resolver -- they exist purely so a failed generation attempt is
-- visible/auditable, exactly like `reports.status='failure'` already is
-- for Tier A. One row per article (PRIMARY KEY `article_id`) -- a later
-- regeneration is an upsert, never a duplicate row/duplicate report.
CREATE TABLE IF NOT EXISTS tier_b_reports (
    article_id TEXT PRIMARY KEY REFERENCES articles(id),
    status TEXT NOT NULL,
    content_hash TEXT,
    schema_version TEXT NOT NULL,
    title TEXT,
    opening_context TEXT,
    key_insights_json TEXT,
    strategic_implications TEXT,
    nomura_opportunity TEXT,
    content_partial INTEGER NOT NULL DEFAULT 0,
    source_json TEXT,
    failure_reason TEXT,
    failure_detail TEXT,
    generated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tier_b_reports_status ON tier_b_reports(status);
"""


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl_type: str) -> None:
    """Additive-only column migration: a no-op if `column` already exists
    (including on a brand-new database, where `_SCHEMA_SQL` already created
    it) — never drops, renames, or alters existing data."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


def _drop_newsletter_selections_event_fk(conn: sqlite3.Connection) -> None:
    """One-time, idempotent rebuild: removes the `REFERENCES events(id)`
    foreign key from an EXISTING `newsletter_selections` table (present
    through schema version 11) so it can hold a Tier B article selection
    (a `tb:`-prefixed id, never a real event id) -- see the two-tier
    CURRENT digest feature. `PRAGMA foreign_key_list` returning no rows
    means either a brand-new database (the FK-free DDL above already
    applies) or a database this migration already ran against -- a safe,
    cheap no-op check before ever touching the table. The rebuild
    preserves every existing selection row exactly (same ids, same
    `selected_at` values) -- it only changes the table's own constraint,
    never any data in it."""
    fk_rows = conn.execute("PRAGMA foreign_key_list(newsletter_selections)").fetchall()
    if not fk_rows:
        return
    conn.execute(
        "CREATE TABLE newsletter_selections_new (event_id TEXT PRIMARY KEY, selected_at TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO newsletter_selections_new SELECT event_id, selected_at FROM newsletter_selections")
    conn.execute("DROP TABLE newsletter_selections")
    conn.execute("ALTER TABLE newsletter_selections_new RENAME TO newsletter_selections")


class Database:
    """Thin wrapper around a SQLite file. Repositories/PersistenceService take
    a `sqlite3.Connection` explicitly (via `connect()`) rather than this
    class doing any persistence itself — this is connection/schema
    management only."""

    def __init__(self, database_path: Path) -> None:
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        conn = self.connect()
        try:
            conn.executescript(_SCHEMA_SQL)
            _add_column_if_missing(conn, "research_items", "is_dated_development", "INTEGER NOT NULL DEFAULT 1")
            _add_column_if_missing(conn, "research_items", "development_date_rationale", "TEXT")
            # Phase 14: explicit execution/digest-persistence metadata on
            # every pipeline run -- additive only, existing rows get the
            # documented safe defaults (manual/not-attempted) rather than
            # a guess, since their true execution_mode predates this column.
            _add_column_if_missing(conn, "pipeline_runs", "execution_mode", "TEXT NOT NULL DEFAULT 'manual'")
            _add_column_if_missing(conn, "pipeline_runs", "period_start", "TEXT")
            _add_column_if_missing(conn, "pipeline_runs", "period_end", "TEXT")
            _add_column_if_missing(conn, "pipeline_runs", "digest_attempted", "INTEGER NOT NULL DEFAULT 0")
            _add_column_if_missing(conn, "pipeline_runs", "digest_persisted", "INTEGER NOT NULL DEFAULT 0")
            _add_column_if_missing(conn, "pipeline_runs", "digest_persistence_status", "TEXT NOT NULL DEFAULT 'not_attempted'")
            _add_column_if_missing(conn, "pipeline_runs", "digest_id", "TEXT")
            # Phase 15: content-grounded vs link-grounded intelligence
            # (research_assistant.models.enums.GroundingType) -- additive
            # only. Every existing row predates this distinction and was
            # built entirely from retrieved article content, so 'content'
            # is not a guess for historical data, it is simply true.
            _add_column_if_missing(conn, "research_items", "grounding_type", "TEXT NOT NULL DEFAULT 'content'")
            _add_column_if_missing(conn, "events", "grounding_type", "TEXT NOT NULL DEFAULT 'content'")
            # Phase 17: explicit run ownership (Section 4) -- NULL for every
            # pre-existing row (predates this column; its true originating
            # run is genuinely unknown, never guessed) and for any row built
            # outside a real PipelineRunner-driven run (e.g. tests). This,
            # never `Article.retrieved_at` (shared/mutable across every run
            # that re-fetches the same URL), is the authoritative answer to
            # "which run produced this?".
            _add_column_if_missing(conn, "research_items", "run_id", "TEXT")
            _add_column_if_missing(conn, "events", "run_id", "TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_research_items_run_id ON research_items(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)")
            # Phase 17 (stale-lock audit): which OS process holds the
            # current_run_lock singleton row -- lets a later try_acquire
            # distinguish a genuinely active CURRENT run from one whose
            # owning process has died without reaching the `finally`
            # release (crash, kill, machine power-loss). NULL for a lock
            # row created before this column existed; such a row is
            # treated as having no verifiable live owner (see
            # CurrentRunLockRepository.try_acquire).
            _add_column_if_missing(conn, "current_run_lock", "owner_pid", "INTEGER")
            # Deterministic recall safeguard (Qwen-disagreement audit trail):
            # NULL for every pre-existing row and for any item the safeguard
            # never touched -- only set when `agents.base._resolve_relevance_
            # override` actually overrode a deep-analysis IRRELEVANT verdict.
            _add_column_if_missing(conn, "research_items", "relevance_override", "TEXT")
            # Reliability patch: `defer_count` on a `candidate_backlog` table
            # created before this column existed (version 10) -- additive,
            # defaults to 0 for every pre-existing row (their true deferral
            # history before this column existed is genuinely unknown, never
            # guessed upward).
            _add_column_if_missing(conn, "candidate_backlog", "defer_count", "INTEGER NOT NULL DEFAULT 0")
            # Two-tier CURRENT digest (schema version 12): see the function's
            # own docstring -- removes a now-outdated FK so a Tier B article
            # selection can share the existing selections table.
            _drop_newsletter_selections_event_fk(conn)
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value "
                "WHERE CAST(excluded.value AS INTEGER) > CAST(schema_meta.value AS INTEGER)",
                (str(CURRENT_SCHEMA_VERSION),),
            )
            conn.commit()
        finally:
            conn.close()

    def schema_version(self) -> Optional[int]:
        conn = self.connect()
        try:
            row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
            return int(row["value"]) if row else None
        finally:
            conn.close()
