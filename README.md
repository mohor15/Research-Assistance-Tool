# Nomura AI & Fintech Research Assistant

A local, agentic research-intelligence system that monitors peer banks, AI
providers, AI/fintech startups, and strategy consultancies for meaningful AI
developments — and turns them into manager-ready morning (08:00 IST) and
evening (20:00 IST) digests.

This is **not** a chatbot, a generic web scraper, or an LLM summarizer bolted
onto search results. It is a pipeline:

```text
Sources → Retrieval → Evidence → Agents → Event clustering → Scoring →
Ranking → Report synthesis → Digest → UI
```

Every stage has a single responsibility and consumes only the structured
output of the stage before it:

| Stage      | Responsibility                                      |
|------------|------------------------------------------------------|
| Sources    | Define WHERE to look (`config/sources.yaml`)          |
| Topics     | Define WHAT to look for (`config/topics.yaml`)        |
| Retrieval  | HOW content is fetched (RSS → HTTP → extraction → browser fallback) |
| Evidence   | Source-grounded, provenance-tracked article data      |
| Agents     | Reason over evidence, produce structured research items |
| Events     | Cluster corroborating articles into real-world events |
| Scoring    | Deterministic, config-driven relevance + confidence   |
| Reports    | Synthesize scored events into evidence-backed digests |
| UI         | Present results — no business logic                   |

The system is built in strict, sequential phases (see `NEXT PHASE` at the
bottom of this README). Each phase preserves and validates everything built
before it.

## Local LLM

All reasoning runs through a **local Ollama** instance — no paid or
cloud LLM API is used anywhere in this project.

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

`qwen3:8b` remains installed locally and is still fully supported by the
same config mechanism (change `system.llm.model` in `config/settings.yaml`
or set `OLLAMA_MODEL`) — the project switched its *default* to `qwen2.5:3b`
because `qwen3:8b`'s hybrid "thinking" behavior made single-article
classification calls too slow for real runs (observed 120s+ timeouts in
production validation). See "Local LLM model choice" below.

Configuration is never hardcoded into agent or pipeline logic — everything
reads from `research_assistant.config.get_config()`, which itself reads
`config/settings.yaml` plus environment variable overrides.

## Repository structure

```text
Research Assistant/
├── config/
│   ├── settings.yaml       # system, research, retrieval, ranking, agents, storage config
│   ├── sources.yaml        # source registry: peers, ai_providers, analysts, secondary_sources, startups
│   └── topics.yaml         # AI + financial-services topic/event/theme registry
│
├── src/research_assistant/
│   ├── config/             # YAML loading + Pydantic validation
│   ├── models/             # shared enums, plus the canonical Article/Evidence models (Phase 2)
│   ├── retrieval/          # RSS/Atom, HTTP client, extraction, browser fallback
│   ├── evidence/           # ExtractedArticle -> Article -> Evidence normalization (this phase's core deliverable)
│   ├── agents/             # Phase 3/4/6
│   ├── analysis/           # Phase 5 (event clustering, scoring)
│   ├── pipeline/           # Phase 6 (LangGraph orchestration)
│   ├── storage/            # Phase 7 (SQLite persistence)
│   ├── reporting/          # Phase 6 (report/digest synthesis)
│   ├── ui/                 # Phase 7 (Streamlit dashboard)
│   └── utils/              # cross-cutting helpers (project paths)
│
├── tests/                  # config-loading tests, tests/retrieval/, tests/evidence/
├── data/                   # generated at runtime — gitignored
│   └── cache/              # on-disk page cache (retrieval.cache_pages)
├── logs/                   # generated at runtime — gitignored
├── .env / .env.example
├── pyproject.toml          # src-layout packaging (editable install)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `import research_assistant` work everywhere
cp .env.example .env      # already populated with local defaults in this repo
```

Requires a running local Ollama with `qwen2.5:3b` pulled:

```bash
ollama pull qwen2.5:3b
ollama serve
```

(Not required to pass the test suite — nothing in Phases 0–1 connects to
Ollama; the config loader only validates that `provider`/`model`/`base_url`
are structurally well-formed.)

### Optional retrieval dependencies

`feedparser`, `trafilatura`, and `playwright` are listed in
`requirements.txt` but the retrieval engine treats all three as optional at
import time and degrades gracefully when they're missing:

| Package       | Used for                          | If missing                                              |
|----------------|-----------------------------------|----------------------------------------------------------|
| `feedparser`   | Lenient RSS/Atom parsing          | Falls back to a pure-stdlib `xml.etree` RSS/Atom parser   |
| `trafilatura`  | Higher-quality article extraction | Falls back to a BeautifulSoup + metadata/JSON-LD extractor |
| `playwright`   | Rendering JS-heavy pages          | Browser fallback is skipped; the plain HTTP result is kept |

If Playwright is installed, its browser binaries still need a separate
one-time install:

```bash
playwright install chromium
```

`research_assistant.retrieval.feeds.is_feedparser_available()`,
`.extractor.is_trafilatura_available()`, and `.browser.is_browser_available()`
report which backend is active.

## Validating configuration

```bash
python -c "from research_assistant.config import get_config; c = get_config(); print(c.settings.system.llm); print(len(c.sources.peers), 'peers loaded'); print(len(c.topics.ai_topics.core), 'core AI topics')"
```

Run the test suite:

```bash
pytest
```

## Retrieval architecture (Phase 1)

`RetrievalService` (`research_assistant.retrieval`) is the single entry
point later agents/pipeline code should use — it is the only place that
knows about RSS parsing, HTTP retries, robots.txt, caching, or Playwright:

```text
config.sources.yaml → SourceDefinition
                          ↓
             discover_candidates()  →  RSS/Atom (feeds.py) or HTML link
                          │             scraping (discovery.py), same-domain
                          │             only, navigation links filtered out
                          ↓
                    CandidateURL
                          ↓
                    retrieve()      →  HTTPClient (client.py): timeout,
                          │             retries, robots.txt (robots.py),
                          │             per-domain pacing (rate_limiter.py),
                          │             on-disk cache (cache.py); falls back
                          │             to Playwright (browser.py) only if
                          │             HTTP failed AND browser_fallback: true
                          ↓
                    RetrievedPage
                          ↓
                    extract()       →  trafilatura, else a BeautifulSoup +
                          │             OpenGraph/JSON-LD fallback (extractor.py)
                          ↓
                    ExtractedArticle
```

`retrieve_source(source)` and `retrieve_all()` compose the above for one or
all configured sources. Everything is sequential and rate-limited by
design — this is a research assistant, not a crawler.

Supported retrieval methods:
- **RSS/Atom** — preferred when a source publishes a feed; malformed feeds
  log a warning and yield zero candidates rather than aborting the run.
- **HTML link discovery** — for sources without a usable feed; restricted to
  the source's own domain and filtered for obvious navigation links
  (privacy/careers/login/etc.), never a general web crawl.
- **Direct HTTP** — configurable timeout/retries/retry-delay, sanitized
  headers (no cookies/auth tokens ever logged or cached), honors
  `respect_robots_txt` and `respect_rate_limits`.
- **Browser fallback (optional)** — only when plain HTTP retrieval fails and
  `browser_fallback: true`; never used to bypass logins, paywalls, or
  anti-bot controls, and degrades to the plain HTTP result if Playwright
  isn't installed.

The `startups` registry section has no fixed URLs to crawl yet (it's
discovery-provider-based per the original spec) so
`RetrievalService.list_sources()` does not yet return startup sources.

### Local testing

`tests/retrieval/` never touches the network: HTTP is exercised through a
scripted fake `requests.Session`, robots.txt fetches are monkeypatched, and
the Playwright fallback is exercised by monkeypatching `sync_playwright`
with fakes. Tests specific to `feedparser`/`trafilatura` (`pytest.mark.skipif`)
auto-skip in this environment and start running the moment those packages
are installed — the rest of the suite already exercises the stdlib/BS4
fallback paths for real.

## Evidence & Article normalization (Phase 2)

Retrieval produces heterogeneous, retrieval-layer-only `ExtractedArticle`
objects. Phase 2 turns them into two canonical, deterministic,
LLM-free models that later agents (Phase 3+) actually reason over:

```text
ExtractedArticle + RetrievedPage + SourceDefinition
                      ↓
           normalize_article()   →  URL canonicalization (urls.py), whitespace
                      │              cleanup (text.py), content hashing
                      │              (hashing.py), a deterministic id derived
                      │              from the canonical URL (ids.py) — never
                      │              guesses a missing publication date
                      ↓
                   Article
                      ↓
          deduplicate_articles()  →  collapses copies sharing a canonical URL
                      │              or a content hash; keeps the most
                      │              complete copy of each group
                      ↓
              deduplicated Articles
                      ↓
            article_to_evidence()  →  only for Articles with usable content
                      ↓
                   Evidence
```

`NormalizationService` (`research_assistant.evidence`) is the entry point:
`normalize()`/`normalize_many()`, `deduplicate()`, `to_evidence()`/
`to_evidence_many()`, and `process()` (all four steps composed).

**`Article`** (`research_assistant.models.article`) is one retrieved
publication/page: `id`, `url`, `canonical_url`, `domain`, `source_name`,
`source_type`/`source_priority` (reused from `models.enums` — no separate
"source quality" concept was introduced), `publisher`, `title`, `author`,
`publication_date` (+ a best-effort `publication_date_precision`:
`exact`/`date_only`/`unknown` — see the model's docstring for why `date_only`
is heuristic, not certain), `retrieved_at`, `raw_text`/`clean_text`,
`content_hash`, and the Phase 1 `extraction_method`/`extraction_status` it
came from. It holds no relevance score, no strategic interpretation, no
event/cluster membership.

**`Evidence`** (`research_assistant.models.evidence`) is the source-grounded
unit later agents cite: it answers *where this came from, when it was
published, when it was retrieved, which configured source produced it, was
that source primary/secondary, and how the content was extracted* —
every field traces back to the `Article` and, through it, to retrieval.
Evidence is never generated for an `Article` with no usable text, and its
`content` is exactly the cleaned article text — never an AI summary
(`evidence_span` is a reserved field for a future phase's claim-level
quotes; it stays `None` in Phase 2).

**Provenance, not just a database id:** every `Article`/`Evidence` object
answers the six provenance questions in its own fields (see above) rather
than requiring a lookup elsewhere.

**URL canonicalization** (`evidence/urls.py`) is conservative: it drops
fragments, strips a trailing slash (except the root path), lowercases only
scheme/host (paths/queries can be case-sensitive on real servers), removes
known tracking parameters (`utm_*`, `gclid`, `fbclid`, ...) and exact
duplicate query parameters, and prefers a page-supplied canonical URL when
one is available in `ExtractedArticle.metadata["canonical_url"]` — nothing
in Phase 1 populates that key yet, so this path is forward-compatible
rather than currently exercised.

**Content hashing** (`evidence/hashing.py`) is SHA-256 over the *cleaned*
text, and is an identity check, not a similarity score — it returns `None`
rather than hashing empty content, so two failed extractions are never
mistaken for duplicates of each other.

**Article-level deduplication is not event clustering.** Two `Article`s are
only ever collapsed here if they share a canonical URL or a content hash —
i.e. they are the same underlying page (a mirror, a tracking-parameter
variant, a re-crawl). **Multiple distinct articles describing the same
real-world event — a Reuters story and a Bloomberg story about the same
announcement — remain separate `Article`/`Evidence` objects through all of
Phase 2.** Recognizing that they describe the same event is Phase 5's job
(semantic/entity-based event clustering), not this layer's.

## Phase 0–1 scope

Implemented: the `src/research_assistant` package skeleton and config
loading (Phase 0), and the shared retrieval engine — RSS/Atom, HTTP client,
robots.txt, rate limiting, caching, article extraction, optional Playwright
fallback (Phase 1). See git history / prior phase reports for full detail.

Phases 3–6 (Peer/AI Product/Startup/Analyst/Report agents, event
clustering + deterministic scoring, LangGraph orchestration) are also
implemented — see git history / prior phase reports for detail; this
README was not updated section-by-section for each of those phases.

## Persistence & dashboard (Phase 7)

SQLite (stdlib `sqlite3`, no ORM) is the historical store; `research_assistant.storage`
has `Database` (connection + idempotent schema init — safe to re-run
against an existing file), one repository per model under
`storage.repositories` (persistence/query infrastructure only — no LLM,
retrieval, or scoring logic), and `PersistenceService`, the coordinating
entry point the LangGraph pipeline's final `persistence` node calls.
Persistence is optional and additive: a graph built without a
`persistence_service` behaves exactly as it did in Phase 6.

The database path comes from `config/settings.yaml`'s `storage.database_path`
(already present since Phase 0) — never hardcoded in the dashboard.

`research_assistant.ui.dashboard_data` is a Streamlit-free, fully
unit-tested read-only query layer; `research_assistant.ui.app` is the
actual Streamlit page (`streamlit run src/research_assistant/ui/app.py`)
and only ever calls into `dashboard_data` — no retrieval, no LLM calls, no
clustering/scoring inside the UI.

**Streamlit could not be installed in this sandboxed development
environment** (no outbound network access — the same limitation noted
above for feedparser/trafilatura/playwright). `ui/app.py` was written and
syntax-checked (`py_compile`) but never actually run or import-tested here;
install `streamlit` from `requirements.txt` and run the command above to
use it for real.

## Scheduling & operational hardening (Phase 8)

`research_assistant.pipeline.runner.PipelineRunner` is the orchestration
entry point scheduled jobs and manual runs use: `PipelineRunner.build()`
wires up the real Ollama-backed agents, `RetrievalService`,
`PersistenceService`, etc. from config, then `.run(digest_type,
period_start, period_end)` invokes the unchanged Phase 6 LangGraph graph
and returns a `PipelineRunResult` (a deterministic `run_id` — derived from
`digest_type` + period bounds, never a random UUID or timestamp — plus
`status` (`success`/`partial_success`/`failed`), structured `diagnostics`,
and funnel `counts`). Every pipeline node now also reports simple
structured counters into `PipelineState.metrics` (sources discovered,
candidates retrieved, articles normalized, evidence created, relevant/
ambiguous/failed counts per agent stage, events formed, reports succeeded/
failed, digest event/theme counts) — this is what `PipelineRunResult.counts`
surfaces, and it's plain `logging` output, not a metrics framework.

`research_assistant.scheduling`:
- `jobs.py` — `run_morning_digest`/`run_evening_digest`, independently
  callable with no APScheduler dependency at all (each computes a 12-hour
  period ending at the runner's own injected `Clock.now()`).
- `scheduler.py` — `ResearchScheduler`, a thin APScheduler
  (`BackgroundScheduler` + `CronTrigger`) wrapper registering exactly those
  two jobs at `research.morning_digest_hour`/`.evening_digest_hour` in
  `system.timezone` (`Asia/Kolkata` — both settings already existed since
  Phase 0, so no new configuration was needed for the schedule itself).
  Guards against starting twice in one process; `shutdown()` is safe to
  call even if never started.
- `cli.py` — manual run support (`python -m research_assistant.scheduling.cli
  --digest-type morning|evening`) for development/demonstration, entirely
  outside Streamlit.

A lightweight `pipeline_runs` SQLite table (schema version bumped 1 → 2,
purely additively — re-running `Database.initialize()` against an existing
v1 database never drops anything) records each run's outcome, upserted by
the same deterministic `run_id` so a scheduler retry/restart never
duplicates a run record.

**APScheduler could not be installed in this sandboxed development
environment** (no outbound network access — the same limitation as
feedparser/trafilatura/playwright/Streamlit). `scheduler.py` was written
and syntax-checked but its tests (`tests/scheduling/test_scheduler.py`)
auto-skip via `pytest.importorskip` and will start running the moment it's
installed; `jobs.py` and `cli.py` have zero dependency on it and are fully
tested regardless.

CURRENT has no scheduled variant and no automatic email delivery at all
— it is fired manually only (the Dashboard's "Run Live Current Scan"
button, or `python -m research_assistant.scheduling.cli --digest-type
current`), always `execution_mode=MANUAL`. The scheduler
(`ResearchScheduler`) registers only the Morning/Evening jobs; it never
registers, owns, or invokes a CURRENT job. Completion is surfaced to the
user directly in the Dashboard UI (see "CURRENT scan completion
notification" below), never via email.

### Login

The Streamlit UI is gated behind a single shared username/password (not
a multi-user/per-account system — see `config.schema.AuthSettings`).
Configure it:

- `APP_LOGIN_PASSWORD` — environment variable only (`.env`, matching
  `SMTP_PASSWORD`'s existing convention), never checked into
  `settings.yaml`. Leave it unset and the login screen shows a clear
  "not configured" message instead of accepting any password.
- `auth.username` in `config/settings.yaml` (defaults to `admin`).
- `auth.enabled: false` turns the gate off entirely (defaults to `true`).

On success, `st.session_state["authenticated"]` is set for that browser
session and the app renders normally, landing by default on "CURRENT /
Live Scan (Admin)" — the one page with the "Run Live Current Scan now"
button, its completion notification, and the CURRENT digest right below
it. A "Log out" button in the sidebar clears the session.

### CURRENT scan completion notification

Because the live CURRENT scan runs as a detached background subprocess
(20-40+ minutes, surviving Streamlit restarts), the "Run Live Current
Scan now" page watches the scan's own status (the existing
`current_run_lock` state — no new table, no polling thread): while that
lock is held, the page emits a browser-side `<meta http-equiv="refresh">`
that reloads itself every ~8 seconds, and the moment the lock is released
it shows an in-page toast ("CURRENT scan finished"). The reload already
re-reads the latest run/digest from the database each time, so the page
is simply already showing the finished result by the time the toast
appears — a manual refresh works exactly as well. Tier A/Tier B
rendering, "Add to newsletter", "Generate Report", and the newsletter's
.md/.pdf download buttons are completely unchanged.

### Newsletter PDF download

The newsletter panel's "Generate Newsletter" step now offers a "Download
newsletter (.pdf)" button alongside the existing "Download newsletter
(.md)" one. It is a pure, in-memory re-render of the exact same
already-generated Markdown (`reporting.newsletter_pdf.
render_newsletter_pdf`) — it never re-reads the database, never calls the
local model, and a failure here never affects the Markdown download, the
current selection, or any persisted report.

**This was the final planned phase for the current architecture.** No
further phases are planned.
