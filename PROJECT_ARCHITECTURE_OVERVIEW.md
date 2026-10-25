# NOMURA AI & FINTECH INTELLIGENCE
## End-to-End Architecture & Design Rationale

---

# 1. Executive Summary

**What this is.** NOMURA AI & FINTECH INTELLIGENCE is a self-contained, locally-run research and market-intelligence system that continuously monitors a curated set of companies, analysts, and news sources for developments relevant to AI and financial services, and turns the raw noise of the public web into a small number of fully-vetted, human-readable intelligence briefs.

**Who it's for.** It replaces (or supplements) a manual process where an analyst would have to visit dozens of company newsrooms, analyst report pages, and financial news outlets every day, read everything, decide what matters, and write it up. Instead, the system does that scanning and first-pass filtering automatically, and presents a curated shortlist — with full source attribution and a confidence/relevance score for every claim — for a human to review, select from, and turn into a newsletter.

**The problem it solves.** Three things, specifically:
1. **Coverage** — it watches ~30+ sources (banks, AI labs, consultancies, analysts, financial press) that no single analyst has time to check daily.
2. **Signal extraction** — of the hundreds to low-thousands of articles discovered per run, it mechanically narrows this down to the handful that are genuinely new, dated, AI/fintech-relevant developments — not evergreen marketing pages, not recycled press-release boilerplate, not habitual quarterly filings.
3. **Evidence discipline** — every claim the system makes is traceable back to a specific retrieved source; it never invents a date, a deal value, a partner, or a person. When it isn't sure, it says "not verified" or "unknown" rather than guessing.

**How it runs.** Entirely on a local machine, using a local open-weight language model (Qwen, via Ollama) — no data is ever sent to a third-party AI API (OpenAI, Anthropic, Google, Azure). This is a deliberate choice (see §3) with real performance consequences (see §7), but it means no business-sensitive source content ever leaves the machine it's analyzed on.

**Three operating modes:**
- **Morning / Evening** — scheduled, automatic digests (08:00 / 20:00 IST), auto-emailed via SMTP.
- **CURRENT** — an on-demand, manually-triggered live scan a user fires themselves from the web UI, producing an immediate, two-tier (fully-analyzed + awaiting-analysis) snapshot, with an in-app completion notification. (This mode's automation history is itself a deliberate design story — see §8.)

The system is gated behind a login screen (see §9) and exposes a Streamlit web dashboard for viewing digests, running scans, curating a newsletter from selected developments, and exporting/emailing that newsletter (see §10).

---

# 2. Glossary of Terms

**Article** — One discovered web page/URL, as retrieved and extracted by the retrieval layer. Has a title, publisher, URL, extracted clean text, and a detected publication date (with a precision level — see *Date Precision* below). An Article with no usable extracted text is still kept (for observability) but can never become Evidence.

**Evidence** — The subset of an Article's data that is actually usable as a grounding source for an LLM judgment: it requires both non-empty clean text *and* a computed content hash. An Article that failed extraction (paywalled, blocked, JS-only page with no fallback) never becomes Evidence — there is nothing substantive to reason over.

**Candidate / Candidate URL** — A discovered link, before it has been fetched and extracted. Discovery (RSS, sitemap, HTML link-scraping) produces Candidates; retrieval turns (some of) them into Articles.

**Source** — A configured entity in the source registry (`config/sources.yaml`): a company, analyst firm, or publication the system watches, with one or more URLs, a source type, and a priority.

**Source Type** — `primary` (the company's own official channel — e.g. a bank's newsroom), `secondary` (independent journalism/analysis about that company), `analyst` (a research-firm's own publication), or `press_distribution` (a wire service like PR Newswire — syndicated, never promoted to "primary" even though it's first-party-sourced content, because it's a distribution channel, not an editorial voice).

**Triage** — A cheap, first-pass LLM screening step for CURRENT runs only: given a batch of candidates, a (comparatively fast, lightly-schemaed) LLM call decides which are plausibly relevant before committing to the much more expensive full "deep analysis" pass. Triage is the gate between "discovered" and "worth fully analyzing."

**Deep Analysis** — The expensive, full-schema LLM call (`AgentAssessment`) that actually extracts relevance, category, event type, summary, entities, partnerships, deal value, strategic significance, etc. from a piece of Evidence. This is what a specialist Agent (see below) runs, and it is the step that is *budget-limited* (see §7) because it is the slow one.

**Research Item** — The structured output of one successful deep-analysis call: one agent's considered judgment about one article, including its relevance/confidence scores, extracted facts, and a link back to the Evidence it was grounded in.

**Agent** (a.k.a. "Specialist Agent") — One of five domain-focused analysts (`peer`, `ai_product`, `startup`, `analyst`, `financial_news`), each with its own system prompt and routing scope, that turns an (Article, Evidence) pair into a Research Item (or rejects it as irrelevant). A sixth, `report`, is not a content analyst — it writes the final human-readable report text from already-extracted facts.

**Event** — One or more Research Items clustered together because they describe the same real-world development (e.g. the same partnership announcement covered by both the company's own press release and a news outlet). An Event has one *primary source* (deterministically chosen by source-type/priority/date, never by the LLM) and zero or more secondary sources.

**Grounding Type: CONTENT vs LINK** — Most Events are **CONTENT**-grounded: the article body was actually retrieved and the summary/facts are based on real extracted text. A **LINK**-grounded Event is one where the article body could not be retrieved at all (blocked, paywalled) but a dated, relevant development was still identified purely from trusted *metadata* (title, source, publication date) — with an explicit guarantee that nothing is claimed beyond what that metadata actually states.

**Report** — The final, human-readable write-up for one Event, generated by the `report` agent from the Event's already-extracted facts (never a new round of fact discovery).

**Digest** — A named, dated snapshot of qualifying Events, built and persisted at the end of a pipeline run. There are three Digest Types: **Morning**, **Evening**, and **CURRENT**.

**Tier A** — Within a CURRENT digest, the set of Events that received full deep analysis *and* have a successfully generated Report. These are what render as the normal, fully-detailed cards on the CURRENT page.

**Tier B** — Within a CURRENT run, the set of candidates that *passed triage* (judged plausibly relevant) but never got a deep-analysis budget slot, or whose deep-analysis attempt timed out/failed recoverably. These render as lightweight "awaiting specialist analysis" cards: a verbatim excerpt of already-retrieved text, with no generated summary, no relevance score, no strategic implications — because no analysis was ever actually performed on them. (Full definition and rationale: §6.)

**Backlog** (`candidate_backlog`) — The persistence mechanism that carries Tier-B-eligible (and failed/retrying) candidates forward across runs, so a candidate the budget didn't reach this run isn't lost — it's reconsidered next run, with an explicit, inspectable status (see *Backlog Status* below) rather than silently dropped.

**Backlog Status** — One of: `pending_triage_budget`, `pending_deep_analysis_budget`, `pending_retry` (non-terminal — must be reconsidered later), or `completed`, `rejected_triage_irrelevant`, `rejected_deep_analysis_irrelevant`, `rejected_date_gate`, `rejected_retry_exhausted`, `quarantined` (terminal — every one carries an explicit, auditable reason; nothing is ever a silent drop).

**Deep-Analysis Budget** — The hard cap (`max_deep_analysis`, currently 4) on how many expensive deep-analysis LLM calls one CURRENT run will make, regardless of how many candidates passed triage. See §7 for the full hardware rationale.

**Relevance Override (deterministic recall safeguard)** — A narrow, auditable rule that can promote a candidate the LLM judged "irrelevant" back to "relevant," but *only* when deterministic evidence (hits in ≥2 independently-defined intelligence-scope categories, plus confirmed dated-announcement language) disagrees with the model. Never the reverse (relevant → irrelevant is never overridden).

**Date Precision** — How confidently an Article's publication date is known: `EXACT` (a real timestamp), `DATE_ONLY` (a real calendar date, no time), `MONTH` (only a month/year could be determined — e.g. from a URL path), or `UNKNOWN`. Only `EXACT`/`DATE_ONLY` dates are trusted enough to qualify a development for a date-bounded digest window.

**Execution Mode** — `SCHEDULED` (set only by the APScheduler cron jobs) or `MANUAL` (everything else — the CLI, the UI's scan button, tests). Never inferred after the fact.

**Digest Persistence Status** — Exactly one of `PERSISTED` (a digest with ≥1 qualifying event was built and saved), `NO_QUALIFYING_EVENTS` (a digest was built but had zero events), `PERSISTENCE_FAILED` (building succeeded but the database write failed), or `NOT_ATTEMPTED` (the pipeline never reached that stage). Exists specifically so "no digest" is never an ambiguous, unexplained outcome.

**Run Status** — `SUCCESS`, `PARTIAL_SUCCESS` (isolated, non-fatal errors occurred — e.g. a few sources failed — but the run completed and persisted something usable), or `FAILED`.

**`current_run_lock`** — A single-row database table acting as a mutex: while a CURRENT scan subprocess is running, this table holds its process id and start time, both to prevent two CURRENT scans running concurrently and to let the UI know a scan is in flight.

**Lookback Window / CURRENT Window** — The rolling time window (24h for Morning/Evening's underlying retention, 48h specifically for CURRENT's own qualifying window) that bounds which articles are eligible to be considered "current" at all.

**Link-Grounded** — See *Grounding Type* above; also used loosely to describe the specific small code path (`agents/link_grounded.py`) that builds these Events deterministically, without an LLM call.

---

# 3. High-Level Architecture

## 3.1 The pipeline, end to end

```
   Source Registry (config/sources.yaml)
            |
            v
   DISCOVERY        -- find candidate URLs (RSS, sitemap, HTML links)
            |
            v
   RETRIEVAL         -- fetch each candidate (HTTP -> browser fallback)
            |
            v
   EXTRACTION        -- turn raw HTML into clean article text
            |
            v
   EVIDENCE          -- Article -> Evidence (deterministic; drops anything with no usable text)
            |
            v
   ROUTING           -- which specialist Agent(s) is this article in scope for?
            |
            v
   [CURRENT only: TRIAGE]  -- cheap LLM pre-screen, bounded candidate pool
            |
            v
   DEEP ANALYSIS      -- the expensive LLM call -> Research Item (or rejected)
            |
            v
   EVENT FORMATION    -- cluster Research Items describing the same development
            |
            v
   SCORING            -- relevance + confidence, from configured weights
            |
            v
   REPORT GENERATION  -- LLM writes the human-readable report (Tier A only)
            |
            v
   DIGEST PERSISTENCE -- save the qualifying Events + metadata as a Digest
            |
            v
   UI / DELIVERY      -- Streamlit dashboard, SMTP email (Morning/Evening + manual newsletter)
```

This is a literal LangGraph pipeline (`pipeline/graph.py`) — each stage above is a graph node (`pipeline/nodes.py`), threaded through a single `PipelineState` object, orchestrated by `PipelineRunner` (`pipeline/runner.py`). The same graph runs for all three digest types; CURRENT simply activates extra, bounded-budget logic (triage + deep-analysis budget + backlog) that Morning/Evening never touch — Morning/Evening retain the original, simpler "every routed candidate gets analyzed" per-agent loop.

## 3.2 Why 100% local, no paid AI APIs

This is a foundational, non-negotiable constraint of the project, not an incidental implementation detail. Two reasons:

1. **Data sensitivity.** The sources being analyzed include competitor banks' strategic announcements and financial-services content. Routing that content through a third-party AI API means that content leaves the organization's control. Running entirely against a local Ollama instance means no source text, no extracted facts, and no generated analysis ever leaves the machine the system runs on.
2. **Cost.** A paid API charged per token would scale directly with how many articles are discovered and analyzed — a cost that grows with the system's own usefulness. A local model has a one-time hardware cost and zero marginal cost per call.

The trade-off this constraint creates — a consumer-grade CPU running a 3-billion-parameter model is meaningfully slower than a cloud GPU-backed frontier model — is the direct cause of the budget/timeout tuning discussed in §7. That trade-off is accepted deliberately, not accidentally.

## 3.3 Why evidence-first

Every extracted fact (a date, a deal value, a partner company, a person's name) must be traceable to something actually present in retrieved source text or metadata — never inferred, never filled in from the model's own background knowledge, never guessed when ambiguous. Concretely, this shows up as:
- Evidence conversion refusing to create Evidence at all for an Article with no usable text (§5.2).
- Deterministic guardrails in `agents/base.py` that can only *narrow* what the LLM claimed (e.g. confirming a claimed partnership is actually backed by real partnership language in the text, not just two company names appearing near each other) — never widen it.
- Never bypassing robots.txt, paywalls, or anti-bot protections to get content a publisher didn't intend to make freely available — if a source blocks the system, the system respects that block and reports it as a diagnosable retrieval failure, not something to route around.
- The Relevance Override (§2) being allowed to add a relevant item back in, but never to add a *fact* — it only changes a true/false relevance verdict, under strict, auditable conditions.

This matters for a financial-services intelligence product specifically: a wrong but confident-sounding claim (a fabricated deal size, a fictitious partner) is far more dangerous than a missed article. The system is deliberately biased toward "say nothing / say unknown" over "guess."

---

# 4. End-to-End Data Flow — Following One Article

To make the abstract pipeline concrete, here is literally what happens to a single article, from the moment it's discovered to its final fate (digest, backlog, or rejection):

1. **Discovery.** A source's RSS feed, sitemap, or HTML page is scanned for links (`retrieval/discovery.py`, `retrieval/feeds.py`, `retrieval/sitemap.py`). Each found link becomes a `CandidateURL`, optionally carrying a feed-supplied title/date/summary. Candidates are ranked by topical keyword relevance and truncated to `max_candidates_per_source` (100) before being fetched at all — this exists so that a source with hundreds of nav links doesn't waste fetch budget on pages that are obviously off-topic by title alone.

2. **Retrieval.** Each surviving candidate URL is fetched via a tiered strategy: a plain HTTP GET first (`retrieval/client.py`), honoring `robots.txt` (`retrieval/robots.py`) and a per-domain rate limiter (`retrieval/rate_limiter.py`). If that fails outright, or succeeds but yields zero usable links/content (common on JavaScript-rendered pages like Yahoo Finance), the system falls back to a real headless-browser render (`retrieval/browser.py`, Playwright) — but *never* for a URL already confirmed `robots_disallowed`, and never for a confirmed 404/410 (rendering can't recover a page that was never there).

3. **Classification of failure (if any).** If retrieval ultimately fails, the failure is classified into one of a fixed vocabulary — `ROBOTS_BLOCKED`, `ANTI_BOT_ACCESS_BLOCKED`, `CONNECTION_ERROR`, `CONNECTION_TIMEOUT`, `HTTP_403`/`HTTP_404`, etc. (`retrieval/classification.py`) — so every failure is diagnosable, not a generic "it didn't work." A 401/403 confirmed independently by *both* the plain HTTP client and a real rendered browser is specifically reclassified as `ANTI_BOT_ACCESS_BLOCKED` rather than a generic connection error, because that double-confirmation is strong evidence of deliberate blocking rather than a transient issue.

4. **Extraction.** A successfully-fetched page's HTML is turned into clean article text (`retrieval/extractor.py`), with the publication date recovered from the most reliable available signal (`<time>`/JSON-LD metadata preferred over guesswork) and tagged with a Date Precision level.

5. **Evidence conversion.** The resulting `Article` is checked (`evidence/conversion.py`): if it has no `clean_text` or no computable `content_hash`, it produces **no Evidence at all** — there's nothing substantive to ground an LLM judgment in, so the article simply cannot proceed past this point via the normal content path. (It may still become a *Link-grounded* Event later, if its metadata alone is sufficient — see §6.)

6. **Routing.** Articles with usable Evidence are routed to whichever specialist Agent(s) their source maps to (a bank → `peer`; an AI lab → `ai_product`; a wire/financial press source → `financial_news`; etc. — see §5.3). A cheap keyword pre-filter (`_passes_keyword_prefilter`) skips an LLM call entirely if nothing AI-related is even mentioned.

7. **(CURRENT only) Triage.** Before committing to full deep analysis, a cheap, smaller-schema LLM call batches candidates and asks a lighter relevance question. Only "winners" proceed. This step exists purely because CURRENT's candidate pool can be 150+ articles per run — deep-analyzing all of them is computationally infeasible (§7), so triage is the affordable first filter.

8. **(CURRENT only) Domain-fair deep-analysis selection.** Triage winners are cut down to the deep-analysis budget (4) *fairly across specialist domains* — not simply "the 4 globally strongest" — so a single very newsy domain (e.g. financial news) can't starve every other domain of analysis slots in a given run.

9. **Deep analysis.** The winning candidates each get one full `AgentAssessment` LLM call. The specialist Agent's deterministic guardrails (§5.3) then validate/narrow what the model claimed (dated-development confirmation, partnership validation). The result is either a `ResearchItem` (relevant) or nothing (rejected as irrelevant, possibly after the Relevance Override check).

10. **Event formation.** Research Items describing the same real development are clustered into one `Event`, with a deterministically-chosen primary source and canonical title.

11. **Scoring.** The Event receives relevance and confidence scores, computed from the configured weights (§12 Appendix) — never hidden inside a prompt.

12. **Report generation (Tier A only).** For Events with enough deep-analysis coverage, the `report` agent writes the final human-readable report from the already-extracted facts — this is a *writing* step, never a new fact-discovery step.

13. **Digest persistence.** Qualifying Events (and their Reports) are saved as part of a `Digest` row, tagged with a `DigestPersistenceStatus`.

14. **Resolution for everything else.** A candidate that didn't make it this far — never reached triage, triaged out, lost the deep-analysis budget cut, or timed out — gets an explicit `BacklogStatus` and is carried forward to the next CURRENT run rather than silently disappearing. This is what feeds Tier B.

---

# 5. Component Deep-Dives

## 5.1 Retrieval (`src/research_assistant/retrieval/`)

**What it is.** The sole gateway between the system and the public internet. Per the architecture mandate, *no agent or pipeline code ever fetches a URL itself* — everything goes through `RetrievalService.retrieve_source()` / `.discover_candidates()` / `.retrieve()` / `.extract()`.

**Why it's centralized.** If every agent implemented its own scraping, a fix to (say) robots.txt handling or rate-limiting would need to be replicated N times and would inevitably drift. Centralizing it means one robots-compliance implementation, one rate limiter, one cache, one browser-fallback policy, used identically regardless of which specialist agent eventually consumes the result.

**Key internal pieces:**
- `discovery.py` / `feeds.py` / `sitemap.py` — three separate discovery strategies (RSS/Atom feeds, sitemaps, raw HTML link-scraping), tried per source according to what that source actually offers.
- `client.py` (`HTTPClient`) — retries, timeouts, and routes every request through the shared robots checker, rate limiter, and page cache.
- `browser.py` — Playwright-based headless rendering, used *only* as a fallback, never as the default path (real rendering is slow; most sources don't need it).
- `robots.py` (`RobotsChecker`) — enforced unconditionally; the system will not fetch a URL robots.txt disallows, and will not attempt to recover it via the browser fallback either.
- `rate_limiter.py` (`DomainRateLimiter`) — a minimum delay between requests to the same domain, so the system behaves like a polite, bounded crawler rather than a burst scraper.
- `cache.py` (`PageCache`) — avoids re-fetching the same URL repeatedly within a short window.
- `classification.py` — the fixed failure-reason vocabulary described in §4 step 3.
- A handful of **per-domain special cases** exist deliberately and are explicitly scoped to one domain each, never generalized speculatively: Yahoo Finance (unconditional browser rendering, because its plain HTML is nav-only), CNBC (sitemap scavenging for higher discovery volume), NatWest (a dedicated content-hub parser), and FT (a dedicated search-based discovery path). Each was added in response to a *confirmed, live-verified* coverage gap for that specific source, not a speculative generalization.

**Design decision — browser fallback is narrow, not default.** Rendering a page in a real browser is slow (seconds, not milliseconds) and most of the ~30 configured sources don't need it. It is only invoked when: the plain HTTP fetch failed outright, or succeeded but returned zero candidate links (a strong, specific signal of a JS-only shell page), or for the handful of domains confirmed to require it unconditionally (Yahoo Finance). This keeps typical retrieval fast while still recovering sources that would otherwise be invisible to the system.

## 5.2 Evidence (`src/research_assistant/evidence/`)

**What it is.** The deterministic, LLM-free boundary between "a page was fetched" and "this is something an agent is allowed to reason over." `conversion.py`'s `article_to_evidence()` is intentionally trivial: no clean text or no content hash → `None`, full stop. There is a companion `evidence_rejection_reason()` purely so this (otherwise silent) rejection is observable/testable, without changing the actual gate.

**Why this boundary exists as a separate step** (rather than just letting agents check `article.clean_text` themselves): it makes "can this even be reasoned about" a single, auditable, independently-testable decision, decoupled from any one agent's logic, and it's the natural place to compute `content_hash` (used for deduplication — `evidence/hashing.py`, `evidence/dedup.py`) and a stable `evidence_id` (`evidence/ids.py`).

**What gets rejected, and why it's tracked by reason, not just dropped:**
- `missing_clean_text` — extraction produced nothing usable (most commonly: paywalled, blocked, or a JS-only page the browser fallback also couldn't recover).
- `missing_content_hash` — a rarer internal-consistency case.
- Both missing — the article was never meaningfully retrieved at all.

Each of these is surfaced in pipeline diagnostics (e.g. the `evidence_rejected_missing_both` counter seen in CURRENT run funnel metrics) specifically so a human reviewing a run's health can distinguish "we found articles but couldn't read most of them" from "we simply didn't find anything this run" — two very different operational problems with different fixes.

## 5.3 Agents (`src/research_assistant/agents/`)

**What they are.** Five domain-specialist analysts plus one writer:

| Agent | Routing scope | Job |
|---|---|---|
| `peer` | Configured peer banks | Competitive/strategic developments at peer institutions |
| `ai_product` | Configured AI providers/labs | AI model/platform/product developments |
| `startup` | Discovery-driven (no fixed registry) | AI/fintech startup funding, launches |
| `analyst` | Configured analyst firms | Analyst research findings/reports |
| `financial_news` | Financial press (Reuters, FT, Bloomberg, CNBC, WSJ, etc.) | Broad financial-services + AI news |
| `report` | N/A — not a content analyst | Writes the final human-readable Report from an Event's already-extracted facts |

**Why a shared base class, not five independent implementations.** `agents/base.py`'s `ResearchAgent` owns everything that isn't domain-specific: the keyword pre-filter, prompt construction (with the evidence-content truncation cap — see below), schema-constrained LLM invocation, and — critically — a set of **deterministic post-LLM guardrails** that apply identically to every agent's output, regardless of domain:

1. **Dated-development confirmation** (`_resolve_dated_development`). Qwen's own `is_dated_development` judgment is only ever *narrowed*, never promoted: it's trusted as `True` only if the evidence text itself contains explicit announcement/release language (a large, generated set of verb-tense patterns for `announce/launch/unveil/introduce`, covering "today announced," "has launched," bare present-tense "introduces," etc., while explicitly excluding habitual/recurring phrasing like "announces its quarterly results every year" and future-intention phrasing like "plans to announce"), *or* the article's own metadata date is precisely known (EXACT/DATE_ONLY). This exists because losing a genuine dated development's `event_date` silently excludes it from every date-bounded digest even though it was correctly judged relevant — a real, repeatedly-found bug class in this project's history, now closed generically (new announcement verbs get full tense coverage automatically) rather than patched one phrase at a time forever.

2. **Partnership validation** (`_validate_partnerships_against_evidence`). A company name the LLM listed under `partnerships` only survives if real relationship language ("partnered with," "in collaboration with," "joint venture with," etc.) appears near that name in the actual evidence text — specifically *excluding* "available through/on," "built on," "powered by," "compatible with" style mentions, which describe distribution/integration, not a confirmed partnership. This closes a confirmed false-positive class (e.g. AWS/Microsoft "availability" language being mistaken for a strategic partnership).

3. **Relevance Override** (`_resolve_relevance_override`) — described in §2/§4 step 9. Deliberately narrow: requires hits in ≥2 independently-defined scope categories *and* confirmed dated-announcement language, and every time it fires, the disagreement is permanently recorded on the resulting Research Item (`relevance_override` field) — never hidden.

**Why evidence content is truncated to 4,000 characters before being sent to the model** (`MAX_EVIDENCE_CONTENT_CHARS`). This is a direct consequence of the hardware constraint in §7: an uncapped ~11,000-character article measurably pushed a single call past the timeout. The cap keeps the *beginning* of the article (news/press-release writing is front-loaded — the who/what/when is almost always in the opening paragraphs) and appends an explicit truncation marker so neither the model nor a human auditor mistakes a cut body for a complete one. Crucially, this cap is **prompt-only** — the full untruncated text is still stored in `Evidence.content`; only what's sent to the LLM is capped.

**Why `StartupAgent` has no fixed company registry.** Unlike `peer`/`ai_product`/`analyst` (which monitor a known, configured list of companies), there is no practical fixed list of "every AI/fintech startup" — so `supports()` accepts any article structurally, and the actual narrowing happens entirely through the keyword pre-filter and the LLM's own relevance judgment.

## 5.4 Pipeline Orchestration (`src/research_assistant/pipeline/`)

**What it is.** `pipeline/graph.py` wires the stages from §3.1/§4 into an explicit LangGraph state machine; `pipeline/nodes.py` implements each node's actual logic; `pipeline/state.py` defines the single `PipelineState`/`PipelineDiagnostic` objects threaded through every node; `pipeline/runner.py`'s `PipelineRunner` is the one orchestration entry point both the scheduler and the manual CLI/UI call — it is also the *only* place that constructs real, Ollama-backed agents and a real SQLite-backed persistence service (every test, and every lower-level direct caller, injects fakes instead).

**Why CURRENT has a fundamentally different `agent_analysis` code path than Morning/Evening.** Morning/Evening retain the original, simple design: every specialist agent analyzes every article routed to it, up to a per-agent `top_k`. This is fine at Morning/Evening's cadence and candidate volume. CURRENT, by design, needs to process a much larger, continuously-changing pool of candidates (currently 150+ per run) within a single interactive session a human is waiting on — an unbounded per-agent loop here would make a CURRENT run take hours. So CURRENT activates an entirely separate, budget-bounded funnel: triage → domain-fair deep-analysis selection → the budget-capped deep-analysis loop → backlog carry-forward for everything the budget didn't reach. This is the single most consequential architectural fork in the whole pipeline, and it is the origin of Tier A/Tier B (§6).

**Why deep analysis is strictly sequential (`concurrency=1`), never parallel.** This machine runs the model on CPU only. Firing multiple inference calls concurrently against one CPU-bound model doesn't parallelize the *work* — it forces the calls to compete for the same compute, typically making each one slower with no net throughput gain (and losing the benefit of Ollama's warm-prompt cache, discussed in §7). Sequential execution is therefore not a missed optimization; it's the correct choice for this specific hardware profile.

**Why the candidate backlog exists instead of just discarding unprocessed candidates.** A candidate that passed every deterministic gate and was judged (by a cheap signal, or simply by rank) worth triaging, but didn't get a slot this run purely because of the budget, is not "not relevant" — it's "not yet evaluated." Dropping it would silently lose real, possibly important developments purely because of a run's compute ceiling, and would make Tier B's "awaiting analysis" framing a lie (there'd be nothing persisted to show). The backlog (`storage/repositories/backlog.py`, `candidate_backlog` table) instead gives every such candidate an explicit, inspectable, non-terminal status, resumes it automatically on the next CURRENT run, and bounds *how* persistent that resumption is (`backlog_max_attempts` — a repeatedly-failing candidate eventually becomes a terminal, human-inspectable rejection rather than retrying forever; `backlog_retention_days` — how long a resolved row is kept before cleanup-eligibility).

## 5.5 Events (`src/research_assistant/events/`)

**What it is.** Turns a flat list of Research Items into clustered Events (`clustering.py`, `builder.py`), then scores each one (`scoring.py`) against the configured relevance/confidence weights.

**Why primary-source selection is fully deterministic, never LLM-decided.** `events/builder.py` ranks cluster members by `source_type` (primary > analyst > secondary > press_distribution), then `source_priority`, then earliest publication date, then a stable id tiebreak — all facts already known from configuration, not something requiring judgment. This guarantees the same cluster always produces the same "primary source" choice regardless of processing order, and means the single highest-stakes editorial decision in the whole pipeline (whose account of an event gets treated as authoritative) is never subject to model inconsistency.

**Why press-distribution sources can never outrank even an ordinary secondary source.** A wire-service release (PR Newswire, etc.) is first-party-sourced but is a distribution channel, not independent journalism or an official company channel — ranking it below "secondary" (independent reporting) keeps the primary-source hierarchy meaning what it claims to mean.

## 5.6 Reporting (`src/research_assistant/reporting/`)

Covers: `digest_service.py` (assembling a `Digest` from qualifying Events), `themes.py` (cross-market theme extraction, used by the "Cross-Market Themes" UI page), `newsletter_pdf.py` (pure, in-memory re-render of already-generated newsletter Markdown into PDF — explicitly never a second data pass or LLM call), `tier_a_report_regeneration.py` (lets a user explicitly re-run the `report` agent for one already-formed Tier A event, from its existing extracted facts — never re-discovery), and `tier_b_newsletter_resolution.py` (the one place a Tier B article can be promoted into a full report at newsletter-generation time — see §10).

## 5.7 Storage (`src/research_assistant/storage/`)

**What it is.** A SQLite database (`data/research.db`) accessed exclusively through `PersistenceService` — a facade over one repository class per concept (articles, evidence, research_items, events, reports, digests, pipeline_runs, scheduler_state, newsletter_selections, current_run_lock, email_deliveries, email_settings, candidate_backlog, tier_b_reports, and a couple more). Schema is currently at version **13**.

**Why SQLite, not a server database.** This is a single-machine, single-operator-triggered system — there is no need for the operational overhead of a client-server database, and SQLite handles the system's actual concurrency need (multiple browser tabs reading while, at most, one scan subprocess writes) perfectly adequately.

**Why migrations are strictly additive.** `_add_column_if_missing()` and `CREATE TABLE IF NOT EXISTS` mean a schema change can only ever *add* a column or table, never rename/remove one in place, and `CURRENT_SCHEMA_VERSION` only ever increases. This is a deliberate safety constraint: an additive-only migration can be applied to the live production database with no plausible path to data loss, and a version bump that turns out to be unwanted (as literally happened during this project's login-feature work — bumped to 14, then reverted back to 13 once the feature's shape changed) is safe to roll back precisely because nothing destructive ever happened to the existing data.

**Why `current_run_lock` is a single-row table, not an in-memory flag.** A CURRENT scan runs as a detached OS subprocess (see §8) that can outlive the Streamlit process that launched it, and multiple browser sessions (or a Streamlit restart) need to agree on "is a scan currently running" without any in-memory state. A single persisted row with the owning process id and start time is the simplest construct that is simultaneously a mutual-exclusion lock (preventing two concurrent CURRENT scans) and an observable fact the UI can poll.

## 5.8 Scheduling (`src/research_assistant/scheduling/`)

Covered in depth in §8.

## 5.9 UI (`src/research_assistant/ui/app.py`, `ui/dashboard_data.py`)

**What it is.** A Streamlit dashboard. `app.py` is explicitly documented as presentation-only: *"this file never retrieves, scrapes, clusters, scores, or calls an LLM. Every piece of data comes from `dashboard_data`, which itself only ever reads through `PersistenceService`'s repositories."* `dashboard_data.py` is the sole query layer between the raw persistence repositories and anything rendered on screen.

**Why this separation matters.** It means every page except the one dedicated "CURRENT / Live Scan (Admin)" page is provably incapable of triggering a scan, a retrieval, or an LLM call just by being viewed or refreshed — the heavy, slow, resource-consuming operations are deliberately concentrated in one auditable place rather than scattered implicitly behind "just loading a page."

Covered further in §9 (auth) and §10 (newsletter workflow).

## 5.10 Authentication (`config/schema.py::AuthSettings`, `ui/app.py`)

Covered in depth in §9.

---

# 6. The Two-Tier CURRENT Model (Tier A / Tier B)

This is the central user-facing concept of the CURRENT mode, and it exists directly because of the budget constraint in §7 — it would not need to exist on hardware with no practical analysis limit.

**The problem it solves.** A CURRENT run's candidate pool (currently often 70-150+ relevant-enough-to-triage candidates) vastly exceeds what the deep-analysis budget (4) can fully analyze in one run. Two bad options were rejected:
- *Silently drop everything the budget doesn't reach* — this would make the system appear to have much lower coverage than it actually achieved, and genuinely relevant developments (confirmed by triage) would simply vanish with no trace.
- *Wait indefinitely until everything is analyzed* — on CPU-only hardware, fully analyzing 70-150 candidates sequentially (each taking 40-140 seconds) would mean multi-hour-to-overnight runs for what's supposed to be an on-demand, check-it-now tool.

**The resolution: two tiers, with the system being explicit about which is which.**

- **Tier A** — Events that received the full deep-analysis treatment *and* have a persisted Report: relevance score, strategic implications, "Opportunity for Nomura" framing, related companies, the works. These are what the CURRENT page's main card list shows.
- **Tier B** — Candidates that *triage already confirmed as plausibly relevant* but that never got (or didn't survive) a deep-analysis slot. Rendered as "Relevant articles awaiting analysis" — deliberately showing **only** what's already known for free: a verbatim excerpt of the retrieved text (capped at `tier_b_excerpt_chars` = 550 characters, plain truncation, never an LLM summarization step), the source's own key points if available, and — when present — the triage model's own confidence label, explicitly captioned as "a preliminary relevance screen, not factual accuracy." No generated relevance score, no strategic-implications text, no inferred event type — because none of those were ever actually computed for a Tier B item, and presenting them would misrepresent the system's own confidence.

**Why Tier B deliberately excludes never-triaged candidates.** A candidate that was discovered but never even reached the triage LLM call has *no* confirmed relevance signal at all — showing it under "awaiting specialist analysis" would falsely imply the system already believes it's relevant, which it has never actually judged one way or the other.

**Why a Tier B item can be promoted later.** From the newsletter curation panel, a user can explicitly request "Fetch full article" (a live, one-time network re-fetch, never automatic) or have "Generate Newsletter" resolve a Tier B selection into a full report (`tier_b_newsletter_resolution.py`) — because a human has now decided this specific item is worth the one additional LLM call the automated budget didn't have room for. This keeps the expensive work demand-driven and human-gated rather than happening automatically for every Tier B item.

---

# 7. LLM Budget & Hardware Constraints

This section exists because it is the single most consequential engineering trade-off in the whole system, and it is a **hardware** limitation, not a limitation of the Qwen model or a flaw in the pipeline's code.

**The measured facts (from direct benchmarking on this machine, documented in `config/settings.yaml`):**
- The machine runs `qwen2.5:3b` via Ollama **entirely on CPU** — zero GPU offload (`size_vram: 0`).
- Structured-output generation (the grammar-constrained JSON-Schema decoding every deep-analysis call uses — see §5.3/LLM client) measures **~7.5-7.8 tokens/second**, versus **~16.3 tokens/second** for free-form generation. Grammar-constrained decoding roughly halves throughput, because the model has to validate every candidate token against the target schema, not just generate freely.
- A real production deep-analysis call measured **39.5-60.6 seconds** even before the evidence-content cap (§5.3) was tightened.
- A **cold** call (the first call using a given specialist's system prompt since the model was last unloaded, before Ollama's own prompt-prefix cache is warm) measured up to **98.98 seconds** — within 1.2% of a 100-second timeout that was, in production, genuinely exceeded (3 of 4 deep-analysis attempts timed out in one real run before the timeout was corrected).
- Once a specialist's system prompt is warm in that cache, the *same* specialist's later calls drop to **40-60 seconds**.

**Why the deep-analysis budget is 4, specifically.** Deep-analysis calls run strictly sequentially (one CPU, no real parallelism benefit — see §5.4). At the generic per-agent default of 8 calls, worst case is 8 × 90s = 720 seconds just for deep analysis, on top of everything else a run does (discovery, retrieval, triage) — too long and too failure-prone for an interactive, on-demand tool. 4 was chosen as the number this machine's CPU inference can sustain *reliably* within the timeout, with domain-fair selection (§4 step 8) so the 4 slots aren't simply handed to whichever domain ranked globally strongest.

**Why the deep-analysis timeout is 140 seconds, not 100 or 180.** 100s was tried and measured to be insufficient — the real cold-call ceiling is ~99s, leaving under 1.2% margin, and production genuinely hit this ceiling. 140s gives the measured ~99s cold worst case roughly 40% headroom — large enough to absorb realistic variance, while still being a *real*, bounded ceiling (not 180s, which would just let a truly hung call run needlessly long).

**This is a hardware limitation, not a code or model defect.** The causal chain is: no GPU on this machine → token generation happens on CPU instruction throughput rather than GPU parallel throughput → each structured-output call costs 40-99+ seconds → only a small number of such calls fit inside a tolerable total run time → the code's budget/timeout constants were deliberately tuned to match that reality, with the reasoning for each number documented inline in `config/settings.yaml` rather than left as an unexplained magic number.

**What would change on better hardware.**
- A machine with real GPU acceleration for Ollama — most practically an **Apple Silicon Mac** (M-series chip; Ollama uses Apple's Metal framework for real GPU-accelerated inference there) or a machine with a discrete NVIDIA GPU — would substantially cut the per-call time, because matrix-multiplication work moves from serial CPU execution to a GPU's parallel cores.
- Faster per-call time directly permits raising `max_deep_analysis` well above 4 while staying inside a safe timeout, and potentially enabling real concurrent calls (meaningful only once there's a GPU or multiple independent compute units — concurrency on a single CPU does not help, as explained in §5.4).
- It would **not** automatically make Tier B disappear: Tier B exists because the candidate *pool size* can exceed whatever budget is set, not purely because the budget is small. Faster hardware shrinks the backlog dramatically in practice, but Tier B as a safety-valve concept would likely remain in the system regardless, because a run's candidate pool can always, in principle, exceed any fixed budget.
- Note also: more RAM alone, without better compute (GPU), would **not** meaningfully help — RAM is not the current bottleneck (the model already fits comfortably and stays warm); the bottleneck is raw compute throughput for token generation.

---

# 8. Scheduling & Automation Model

**Current state.** Two genuinely scheduled, automatic jobs — **Morning** (08:00 Asia/Kolkata) and **Evening** (20:00 Asia/Kolkata) — managed by APScheduler (`scheduling/scheduler.py`), each auto-emailed via SMTP on completion. **CURRENT has no scheduled variant at all.** It is triggered exclusively by a user clicking "Run Live Current Scan now" in the UI, which launches the existing manual CLI entry point (`python -m research_assistant.scheduling.cli --digest-type current`) as a detached background subprocess, and it **never** sends an automatic email — its completion is only ever surfaced inside the app itself (a browser-side auto-refresh while the scan's lock is held, and a one-shot toast notification the moment that lock is released).

**Why this is worth documenting as a design decision, not just a fact.** This is the end state of a real, deliberate iteration within this project, not the first design tried:
1. An earlier iteration explored making CURRENT *also* run on a fixed schedule, with the resulting digest automatically emailed to a recipient the moment the scan finished — functionally mirroring Morning/Evening.
2. Further discussion clarified that automatic delivery would have sent the **raw digest** (Tier A content only — Tier B, by definition, was never going to be meaningfully includable in an automatic email, since it's explicitly unanalyzed content), with newsletter curation remaining a fully separate, manual, human-curated step regardless.
3. This was ultimately reversed in full: scheduled CURRENT and its automatic-email path were removed entirely, and the system was returned to "CURRENT is manual-only, no automatic email," specifically so that a human always triggers CURRENT scans deliberately and reviews results interactively in the app, rather than receiving an unreviewed automatic artifact.

The reason this history matters for a manager-facing document: it demonstrates the system was built with a feedback loop with its actual users, not designed once and left alone — the final shape of CURRENT (manual trigger, in-app notification, no auto-email) reflects a real, considered preference, discovered through building and then un-building the alternative, not a limitation the team settled for.

**Heartbeat.** A third, lightweight job (`scheduler_heartbeat`, every 60 seconds by default) writes operational health state (`scheduler_state` table) so the UI can distinguish "the scheduler process is alive but nothing is currently due" from "the scheduler process itself has died" — these are different operational problems requiring different responses, and conflating them (as "no recent run" alone would) would make a dead scheduler invisible until the next missed Morning/Evening run was noticed.

**Misfire handling.** Both Morning and Evening jobs use a generous 4-hour misfire grace window, specifically because a real incident occurred where a scheduled run was silently skipped entirely (APScheduler's default grace window is just 1 second) due to the process being briefly unavailable at the exact scheduled moment. The fix is a real, confirmed-necessary one, not speculative hardening.

---

# 9. Authentication & Access Control

**Current state.** The entire Streamlit application is gated behind a single, shared username/password (`AuthSettings` in `config/schema.py`): a configurable username (default `admin`) and a password read *only* from the `APP_LOGIN_PASSWORD` environment variable (never from `settings.yaml`, never logged, never cached anywhere beyond the one comparison at login time) — deliberately mirroring how the SMTP delivery credential is already handled in this codebase, for consistency. A successful login sets a boolean flag in that browser session; the password itself is never stored anywhere beyond the single comparison. Once authenticated, the user lands by default on the "CURRENT / Live Scan (Admin)" page — the page combining the scan-trigger button, its completion notification, and the CURRENT digest results directly below it, matching the primary intended workflow (log in → fire a scan → get notified → review results).

**What this protects today.** Without the correct credential, no page of the application renders at all — not the digest views, not the scan control, nothing. This is a real, functioning access boundary against anyone without the shared credential.

**A known, deliberately-scoped current limitation.** The system currently has exactly one role: anyone who has the shared credential can do everything an "admin" can do, including triggering a live scan and using every control on the newsletter curation panel (adding/removing newsletter selections, regenerating a report, fetching a Tier B article's full text live, and sending the generated newsletter by email to an arbitrary address via the configured SMTP account). There is currently no separate "viewer" role that could see digest/newsletter content without also being able to trigger these actions. This is a known next step, not an oversight: the single-credential model was the right scope for the immediate requirement ("a person should be able to log in and run the tool"), and role separation is a small, well-understood addition (a second credential plus conditionally hiding the relevant controls) to make when multiple distinct user types are actually needed — e.g. giving two colleagues read-only visibility into results without giving them the ability to trigger scans or send emails on the team's behalf.

---

# 10. Newsletter & Reporting Workflow

1. **Selection.** From either the plain "CURRENT" page or the admin "CURRENT / Live Scan" page, a user clicks "+ Add to Newsletter" on any Tier A event card or Tier B article card. This is a pure database write (`newsletter_selections` table) — it never triggers analysis.
2. **(Optional) Tier B promotion.** For a selected Tier B item whose already-retrieved text is judged insufficient for a real report, a user can explicitly click "Fetch full article" — one live, one-time network re-fetch of that specific URL, never automatic.
3. **(Optional) Report regeneration.** For a selected Tier A event, a user can click "Regenerate report" — re-runs the existing `report` agent against that event's already-extracted facts (no re-discovery, no re-fetch, no specialist re-analysis), overwriting its persisted Report. Explicit and per-event only; never triggered automatically by selection or by "Generate Newsletter."
4. **Generate Newsletter.** Compiles the current selection into one Markdown document. If the selection includes any Tier B items, this is the one point at which those specific items may each cost one additional local-model call (`tier_b_newsletter_resolution.resolve_tier_b_reports`), bounded by a configured per-request cap — demand-driven cost, never an automatic background job.
5. **Export.** The generated Markdown can be downloaded directly, or rendered to PDF (`newsletter_pdf.py`) — explicitly documented as a pure, in-memory re-render of the *same* already-generated Markdown, never a second data-gathering or LLM pass; a PDF-rendering failure never affects the Markdown download or the underlying selection.
6. **Email delivery.** Two explicit, separate controls: "Send to myself" (the configured default recipient) or "Send to this address" (any address typed in, validated as a well-formed email address) — both via the existing SMTP configuration, both requiring SMTP to actually be configured, and both entirely separate from the *automatic* digest-delivery setting (which governs whether successfully generated Morning/Evening/CURRENT digests — note: CURRENT's automatic path was removed per §8 — are emailed to a default recipient without any manual newsletter-curation step at all).

---

# 11. Current Limitations & Future Directions

These are presented as open decisions, not commitments — none of the following has been implemented; each was discussed and deliberately deferred.

**1. Hardware-bound analysis coverage (§7).** The deep-analysis budget of 4 per CURRENT run is a direct function of this machine's CPU-only inference throughput. Meaningfully increasing coverage requires better compute (a GPU-accelerated machine — Apple Silicon or a discrete-GPU machine being the most practical near-term options), not a code change.

**2. No viewer/admin role separation yet (§9).** The single shared credential is sufficient for the system's current single-operator usage; extending it to safely support additional read-only users is a scoped, well-understood piece of future work (a second credential plus conditionally hiding write-capable controls), not yet built.

**3. Machine-dependency for hosting.** The application currently only runs while the developer's own machine is on and the Streamlit process is running. Three options were discussed for removing this dependency, none yet chosen or implemented:
   - **(a) Local automation only** — run it as a Windows background service/scheduled task on the same machine. Removes the "must keep a terminal window open" friction, but the tool is still only available while that specific machine is powered on.
   - **(b) Small always-on cloud VM** — migrate the entire stack (Ollama + Qwen + Streamlit + the SQLite database) to a modest, continuously-running cloud server (roughly $10-20/month, no GPU needed for a model this size). This is the only option that makes the tool available independent of the developer's own machine, while preserving the "self-hosted, local-model-only" principle — the model still never leaves infrastructure the team controls, it simply isn't the developer's personal laptop anymore.
   - **(c) End-user-local installation** — package the project so that anyone who wants to use it installs and runs Ollama + Qwen + Streamlit entirely on their *own* machine. Zero ongoing hosting cost, but each user bears a one-time setup burden (sufficient RAM, a few GB of disk, ~10-15 minutes of setup), and the data never leaves whichever machine runs it — plausibly a genuine plus given the sensitivity of the source material being analyzed.

**4. Generated newsletter content is currently session-scoped, not durably persisted.** The *selection* of which events/articles belong in a newsletter is saved durably (`newsletter_selections` table), but the actual rendered newsletter Markdown/PDF produced by clicking "Generate Newsletter" currently lives only in that browser session's memory. If multiple people were ever given independent access to view "the newsletter as it currently is" (relevant to limitation #2 above), this would need a small persistence addition so a generated newsletter outlives the session that generated it.

---

# 12. Appendix: Key Configuration Values

All values below are read live from `config/settings.yaml` / `.env` — nothing here is hardcoded in application code.

| Setting | Value | Why |
|---|---|---|
| LLM provider | Ollama (local) | §3.2 — no paid third-party AI API |
| LLM model | `qwen2.5:3b` | The only model configured; chosen as a lightweight model practical to run on CPU-only consumer hardware |
| LLM host | `http://127.0.0.1:11434` | `127.0.0.1` specifically, not `localhost` — measured ~2.1s DNS/connection overhead difference per call on this machine |
| Base LLM call timeout | 150s | Headroom above a measured ~90s worst case for a large (~11k-char) article, before the evidence-content cap existed |
| CURRENT triage batch size | 3 | Deliberately small — triage is the cheap, schema-light step, tuned for throughput over richness |
| CURRENT max triage batches | 24 (72-candidate pool bound) | Tripled from an original 24-candidate bound (2026-10-15) to keep pace with a discovery-volume increase that had silently dropped triage coverage to ~16% of the real candidate pool |
| CURRENT deep-analysis budget (`max_deep_analysis`) | **4** | The central hardware-bound constraint — see §7 in full |
| CURRENT triage timeout | 90s | |
| CURRENT deep-analysis timeout | 140s | Re-benchmarked after 100s was measured to produce real timeouts (cold calls ~99s) |
| Backlog claim limit per run | 8 | Bounded resumption — never a full-backlog drain in one run |
| Backlog max attempts | 5 | After which a failing/timing-out candidate becomes a terminal, human-inspectable rejection |
| Backlog retention | 14 days | How long a resolved backlog row is kept before cleanup-eligibility |
| Tier B excerpt length | 550 characters | Plain truncation of already-retrieved text — never an LLM summarization step |
| CURRENT scan window | 48 hours | Widened from an original 24h per a management requirement for broader coverage |
| Base lookback/retention | 24 hours | Independent of the CURRENT window above |
| Morning digest time | 08:00 Asia/Kolkata | |
| Evening digest time | 20:00 Asia/Kolkata | |
| Scheduler misfire grace window | 4 hours | Fixes a confirmed real incident where a scheduled run was silently skipped |
| Max articles per source | 60 | Raised in step with the CURRENT window widening (24h→48h) |
| Max candidates per source | 100 | Same reasoning |
| Relevance weights | AI relevance 30% / Financial-services relevance 25% / Strategic significance 20% / Recency 15% / Novelty 10% | Configured, not hidden in a prompt — auditable and independently tunable |
| Confidence weights | Primary-source confirmation 35% / Multi-source corroboration 25% / Publication-date certainty 15% / Entity-extraction confidence 10% / Event-classification confidence 10% / Content completeness 5% | Same reasoning |
| Deduplication similarity threshold | 0.85 | |
| Database | SQLite, `data/research.db` | Single-machine, bounded-concurrency system — no server database needed |
| Schema version | 13 | Additive-only migrations (§5.7) |
| Auth | Single shared username/password, `APP_LOGIN_PASSWORD` via `.env` only | §9 |
| Email delivery | Disabled by default (`email.enabled: false`) until SMTP credentials/host are configured | Safe-by-default — the rest of the application is fully functional with email delivery off |

---

*This document reflects the system's state as actually found in the current codebase at the time of writing, verified by direct inspection of configuration files and source code — not from historical notes, which may describe an earlier, since-changed state of the project.*
