# NOMURA AI & FINTECH INTELLIGENCE — End-to-End Architecture Deep Dive

**Purpose of this document:** a complete, detailed, "every nitty-gritty"
walkthrough of how this system works, written so it can be handed to another
AI (or a person) to generate a presentation, onboarding doc, or technical
review — without needing to read the source code. It explains not just
*what* each part does, but *why* it exists, *how* it is calculated, and what
real production problems shaped its current design.

---

## 1. What this project is

A local, automated market-intelligence pipeline for Nomura, focused on
AI/GenAI developments relevant to financial services, fintech, capital
markets, and enterprise technology. It:

1. **Discovers** new content from ~30+ configured sources (banks/peers, AI
   providers, analyst firms, and secondary news outlets).
2. **Retrieves and extracts** full article text and trustworthy publication
   dates from each candidate URL.
3. **Screens and analyzes** candidates using a local LLM (Ollama running
   `qwen2.5:3b`, entirely on-machine — no external AI API calls, no data
   leaves the machine).
4. **Clusters** related coverage of the same real-world event across
   multiple sources into a single "Event."
5. **Scores** each Event deterministically (not by the LLM) on relevance and
   confidence.
6. **Writes** a structured analytical report per Event (opening context, key
   insights, strategic implications, a diplomatically-framed "opportunity
   for Nomura" angle).
7. **Packages** everything into a Digest (Morning, Evening, or a rolling
   "CURRENT" live-scan window) and **persists** it to a local SQLite
   database.
8. **Presents** it through a Streamlit dashboard, and optionally **emails**
   a curated newsletter.

Everything is designed around one hard constraint: **the only LLM available
is a small (3B parameter), CPU-only, locally-hosted model.** Nearly every
architectural decision in this system — batching, budgets, bounded funnels,
deterministic guardrails, backlog/resume logic — exists to make a slow,
small, unreliable local model produce trustworthy, high-recall output
without the system grinding to a multi-hour crawl or losing content.

---

## 2. The three "digest types" — and why CURRENT is different

The system produces three kinds of output, all built from the *same*
pipeline, but with different time windows and (for CURRENT) a materially
different processing strategy:

| Digest type | Period | Triggered by | Candidate volume | LLM strategy |
|---|---|---|---|---|
| **Morning** | Previous evening-cutoff → today's morning-cutoff (anchored to configured hours, e.g. 20:00 → 08:00 IST) | Scheduler (cron) or manual | Moderate | Each specialist agent independently analyzes its own `top_k` best candidates |
| **Evening** | Morning-cutoff → evening-cutoff | Scheduler (cron) or manual | Moderate | Same as Morning |
| **CURRENT** | A rolling window ending *right now* (default 48 hours, configurable) | Manual only ("Run Live Scan") | Large, unbounded at discovery time | A **bounded, budgeted LLM funnel**: cheap batched triage first, then a small, strictly budgeted number of expensive "deep analysis" calls |

Morning/Evening are **anchored to wall-clock business-day boundaries**, not
"the last N hours" — a Morning job that happens to run late in the day still
reports the same fixed 20:00→08:00 window, because that window already fully
elapsed. CURRENT is the opposite: a **pure rolling window**, `[now − window_hours,
now)`, with no anchoring at all — because "live scan" means "what's
happened recently," not "since the last scheduled checkpoint."

**Why CURRENT needs its own LLM strategy:** Morning/Evening already bound
their own candidate volume by each source's configured `top_k`/`current_top_k`
limits, so a full per-agent "LLM-analyze everything" loop is tractable. CURRENT
is explicitly an *intelligence-universe scan* — it wants to see everything
relevant across a 48-hour window across ~30 sources, which can easily surface
150–1000+ raw candidates. Running a full "deep analysis" LLM call (the
expensive, detailed-extraction kind) on every single one would take hours on
a CPU-only 3B model. CURRENT's bounded funnel (Section 6 below) exists purely
to solve this.

---

## 3. Configuration — the three registries

Every piece of knowledge the system has about *what to look at* and *what
matters* lives in three YAML files, each independently schema-validated
(Pydantic) at startup. Nothing here is hardcoded in Python.

### 3.1 `settings.yaml` — operational knobs

- **`system`**: timezone (IANA, e.g. `Asia/Kolkata`), LLM provider/model/base
  URL/timeout.
- **`research`**: `lookback_hours` (how much article history is retained,
  separate from any one digest's own period), `top_k_per_agent`,
  `morning_digest_hour`/`evening_digest_hour`, `current_window_hours`
  (default 48).
- **`retrieval`**: HTTP timeout/retries, `max_articles_per_source`,
  `max_candidates_per_source`, browser-fallback toggle, robots.txt
  compliance toggle, page-caching toggle.
- **`ranking`**: the deterministic scoring weights (Section 7) — two sets,
  `relevance` and `confidence`, each validated to sum to exactly 1.0.
- **`agents`**: per-agent `enabled`/`top_k`/`current_top_k` toggles for Peer,
  AI Product, Startup, Analyst, Financial News, Report.
- **`current_llm`**: the entire CURRENT-mode bounded-funnel budget surface —
  `triage_batch_size` (default 3), `max_triage_batches` (default 8),
  `max_deep_analysis` (default 8), `triage_timeout_seconds` (90s),
  `deep_analysis_timeout_seconds` (140s in this deployment, raised from a
  90s default), plus the candidate-backlog knobs (`backlog_claim_limit`,
  `backlog_max_attempts`, `backlog_retention_days`) and the Tier B
  excerpt/report knobs (`tier_b_excerpt_chars`, `tier_b_report_min_sufficient_chars`,
  `max_tier_b_reports_per_newsletter_request`, manual-fetch timeout/size
  limits).
- **`report`**: which optional fields a generated report includes
  (relevance score, confidence score, source evidence, Nomura opportunity).
- **`storage`**: SQLite path.
- **`email`**: SMTP delivery config (credentials never live here — only
  `SMTP_PASSWORD` env var).

### 3.2 `sources.yaml` — the entire research surface

Defines every source the system is allowed to look at, in four buckets:
`peers` (banks/financial institutions), `ai_providers` (OpenAI, Anthropic,
Cohere, NVIDIA, etc.), `analysts` (BCG, consulting/research firms),
`secondary_sources` (Reuters, Bloomberg, CNBC, FT, WSJ, Barron's,
TechCrunch, The Information, PR Newswire). Each entry has:
- `name`, `domain`, `urls` (the actual pages/feeds to discover from),
  `source_type` (primary/secondary/analyst/press_distribution),
  `priority` (high/medium/low).
- Optionally `search` (currently only Financial Times uses an interactive
  site-search fallback) and `research_domain` (routes a secondary source to
  a *specific* specialist agent instead of the generic financial-news
  catch-all — e.g. PR Newswire → `startup_funding`, FT → `ai_technology`).

Also defines `startups` (discovery categories/funding-event vocabulary — not
a fixed company list, since startups are discovery-driven, not registry-driven)
and `retrieval_policy` (source-priority ordering, blocked-source behavior).

**The system never crawls off this registry.** Discovery only ever follows
links on the *same domain* as a source's own configured URLs — it cannot
become a general-purpose web crawler.

### 3.3 `topics.yaml` — the relevance/ranking vocabulary

A curated vocabulary of AI topics (`core`, `models`, `infrastructure`,
`enterprise`, `governance`) and financial-services topics (`core`, `markets`,
`wealth`, `risk`, `operations`), plus `event_types` (the taxonomy of what a
"development" looks like: product_launch, partnership, acquisition,
funding, etc.) and `strategic_themes`.

**Important nuance:** this vocabulary is used for *ranking* (which candidates
rise to the top when a budget forces a cutoff) and for *excluding
non-companies from "related companies" lists* — it is deliberately **never**
used as a hard relevance gate. A candidate is never rejected just because it
doesn't contain one of these exact phrases; that judgment is reserved for the
LLM (triage/deep analysis), which can reason about implicit relevance a
keyword list cannot. (This distinction caused a real regression — see
Section 12.3.)

---

## 4. The pipeline graph — 10 stages, always in this order

Implemented with LangGraph as a simple linear state machine (no branching,
no loops at the graph level — all the "smart" conditional logic is *inside*
individual nodes, not in the graph structure):

```
START → load_config → discover_sources → retrieve → normalize → evidence
      → agent_analysis → event_scoring → report_generation
      → digest_generation → persistence → END
```

Each node is a thin wrapper around an already-independently-tested service
(`RetrievalService`, `NormalizationService`, the research agents,
`EventService`, `ReportAgent`, `DigestService`, `PersistenceService`). The
graph itself contains **zero** business logic — it only sequences calls and
passes a shared `PipelineState` object forward. This separation is
deliberate: every stage can be unit-tested in isolation with fake services,
with no LangGraph/LLM/network/database dependency at all.

---

## 5. Stages 1–4: Discovery → Retrieval → Normalization → Evidence

### 5.1 Discovery (`retrieval/discovery.py`)

For each configured source, turns its hub/RSS page into a list of candidate
URLs. Three discovery mechanisms, tried in order of preference:

1. **RSS/Atom feeds** — preferred when the fetched page is recognizably a
   feed; entries are trusted as-is (a feed is already a curated,
   publisher-declared list — not run through the HTML link filter below).
2. **HTML link scanning** — for an ordinary page, every `<a href>` on the
   same domain is a candidate, but only after passing a cheap, deterministic
   relevance pre-filter (`_is_relevant_candidate`): reject navigation/
   legal/careers/pricing/directory pages outright; keep anything with a
   news/press-release/blog/research keyword in its URL path; demote (never
   hard-reject) a bare short product-name-shaped path unless its own link
   text signals news content; and reject an out-of-scope link (a different
   subdomain/section than the source's own declared research area) that has
   no keep-keyword at all.
3. **Targeted fallbacks for specific sources**: Financial Times gets an
   interactive site-search call (its hub page alone doesn't surface enough);
   NatWest gets a direct content-hub API call (its HTML page yields no real
   article links at all); CNBC and every analyst-category source get a
   **sitemap scavenge** (`sitemap_news.xml` or similar) as an *additional*
   source of candidates, because some real articles are never linked from
   the hub page at the moment it's crawled.
4. **PR Newswire pagination**: PR Newswire's AI topic page is paginated
   (`rel="next"` links); the system follows up to 3 pages, because each page
   holds a fully distinct ~25-release set.

All discovered candidates (regardless of mechanism) are then **ranked by
topical-keyword presence** (using the `topics.yaml` vocabulary) before being
truncated to `max_candidates_per_source` — so a genuinely on-topic link
appearing later on a hub page is never silently dropped in favor of an
earlier off-topic one.

**Several of these mechanisms exist because of specific, confirmed
production misses** — e.g. PR Newswire's `/news-releases/...` URL shape
makes its release pages indistinguishable from its own category/listing
pages by keyword alone, so a numeric-release-ID regex is used instead; FT's
topic/section pages were being discovered and assigned a fabricated "today"
date before a `/content/<uuid>`-shape filter was added; a corporate site's
148-locale homepage selector menu was flooding the candidate pool before a
`/xx/yy/home.html`-shape filter was added.

### 5.2 Retrieval & Extraction (`retrieval/service.py`, `retrieval/extractor.py`)

Each candidate URL is fetched (respecting robots.txt, rate limits, and a
configurable retry policy; falling back to a headless-browser render if
plain HTTP fails and `browser_fallback` is enabled) and its article content
extracted.

**Extraction** uses `trafilatura` as the primary backend (a specialized
article-extraction library — much better than naive HTML stripping at
finding the actual article body vs. nav/ads/comments), with a
BeautifulSoup/`lxml`-based fallback for pages trafilatura can't parse.

**Date extraction is the single most forensically-hardened part of this
codebase.** A wrong or missing publication date cascades through the entire
pipeline (it determines whether CURRENT's date gate keeps the article,
whether an Event can ever get an `event_date`, and what badge the UI shows).
The extraction logic tries, in order:
1. Standard `<meta>` tags (`article:published_time`, etc.) and JSON-LD
   `datePublished`/`dateCreated`.
2. Extended/less-common meta tag variants.
3. Framework hydration-state JSON blobs some sites embed.
4. GA4 analytics `PublishDate` events embedded in page scripts.
5. Visible, labelled or unlabelled date text in the rendered page.
6. URL-path dates (e.g. `/2026/10/15/...`).
7. A handful of narrow, source-specific fallbacks.

**The corroboration guardrail**: trafilatura's own "best guess" date (which
can be mined from body prose, e.g. "a February 2023 investigation") is
**never trusted alone** — it must be independently corroborated by some
tag-level value (a `<meta>` tag, a `<time>` tag, or JSON-LD) agreeing on the
same calendar day, or the date is discarded entirely (`published_at=None`)
rather than risk a fabricated date.

This guardrail itself had a real, confirmed blind spot (fixed 2026-10-16):
a page's `<time>` tags were trusted even when they belonged to *unrelated*
widget content (e.g. OpenAI's "latest from OpenAI" carousel, with its own,
different, much more recent date) rather than the article's own byline —
this caused a 3-month-old GPT-5.6 announcement to be stamped with a
fabricated "today" date and appear fresh in the digest. The fix: a page's
`<time>` tags are now trusted only when **every** `<time>` tag on the page
agrees on the same calendar day — a page mixing several distinct dates among
its `<time>` tags is recognized as a listing/carousel of other items, not
this article's own byline, and none of them are trusted.

Every Article also carries a **`DatePrecision`** (`EXACT`, `DATE_ONLY`,
`MONTH`, `UNKNOWN`) — recording not just *what* date was found, but *how
much to trust it*. Only `EXACT`/`DATE_ONLY` are ever treated as trustworthy
enough to check against CURRENT's narrow rolling window; `MONTH` (stored as
day-1-of-month, a storage placeholder) and `UNKNOWN` are never falsely
treated as "inside" or "outside" a 48-hour window.

### 5.3 Normalization & Deduplication (`evidence/normalizer.py`, `evidence/dedup.py`)

Normalization turns a raw `ExtractedArticle` into the canonical `Article`
model: canonicalizing the URL, resolving `source_name`/`source_type`/
`source_priority` from the registry, and attaching the extracted
title/author/date/clean text.

Deduplication is **identity detection, not event clustering** — two Articles
are the same only if they share a canonical URL or a content hash (exact or
near-exact same text). This is deliberately narrower than Event clustering
(Section 7): two *different* articles from two *different* publishers
covering the *same real-world announcement* are NOT duplicates here — both
survive into event clustering, which is the stage actually responsible for
recognizing "these are coverage of the same thing." Dedup uses the same
union-find (transitive grouping) approach as clustering: if A~B and B~C are
both duplicates, all three merge into one group, and the *most complete*
copy (longest text, has an author, has a date) is kept.

### 5.4 Evidence (`evidence/conversion.py`)

Converts a normalized `Article` into an `Evidence` record — the
LLM-facing/analysis-facing unit, carrying the actual content, publisher,
dates, and source-quality classification that every later LLM call and every
deterministic score will read from. An article with no retrievable content
(robots.txt blocked, anti-bot wall, browser-extraction failure) produces no
Evidence — this is the trigger for the separate **LINK-grounded** path
(Section 6.7).

---

## 6. Stage 5: Agent Analysis — the heart of the system

This is where routing, relevance screening, and detailed extraction happen.
Morning/Evening and CURRENT diverge sharply here.

### 6.1 Deterministic routing (`agents/router.py`) — always runs first, zero LLM calls

Decides *which* specialist agent(s) an article is even eligible for, purely
from configuration and string matching — no LLM involved at all:

1. **Primary-source membership**: an article from a configured
   peer/ai_provider/analyst source always routes to that domain's agent.
2. **Secondary-source strong-signal detection**: an article *not* from a
   primary registry can still route to Peer/AIProduct/Analyst if its text
   contains a strong, specific signal — the full configured company name, or
   a short alias (its first word, if that word isn't a generic noun like
   "bank"/"group"/"technology"). A bare generic category word is never
   itself a signal.
3. **Secondary-source membership**: Reuters/Bloomberg/CNBC/FT/WSJ/
   TechCrunch/PR Newswire/etc. route by exact source-name membership to
   whichever specialist their configured `research_domain` names (default:
   the generic Financial News agent).
4. **Startup catch-all**: since there's no fixed startup-source registry,
   a press-distribution/secondary article routes to the Startup agent only
   if it *also* contains a funding/acquisition/startup-category keyword.

An article can route to more than one agent when independent strong signals
for each are genuinely present. **`FINANCIAL_NEWS` is deliberately checked
before `STARTUP`** in the fixed agent order — a real regression showed
generic startup-vocabulary words ("growth", "AI infrastructure") routinely
appearing in ordinary earnings articles with zero startup angle; when a
single article qualifies for both, the system prefers the explicit,
source-configured route over the generic catch-all so the article isn't
sent *only* to the agent that will correctly find it irrelevant.

### 6.2 Morning/Evening: the "simple" path

Each enabled specialist agent independently takes its own routed articles,
applies a cheap keyword pre-filter (`_passes_keyword_prefilter` —
AI-topic-vocabulary presence in title+body; a necessary but not sufficient
check), ranks by `_candidate_rank_key` (substantive body > contains
dated-announcement language > not a generic topic-overview title > topical
keyword hit count > recency), takes its configured `top_k`/`current_top_k`
best, and runs a **full deep-analysis LLM call** on each one directly — no
triage stage, no shared budget.

### 6.3 CURRENT: the bounded LLM funnel

CURRENT replaces that per-agent unbounded loop with a **shared, globally
budgeted, two-stage funnel**: *cheap batched triage first, then a small,
strictly bounded number of expensive deep-analysis calls.* This is the
single biggest architectural difference in the system, and exists purely
because of the CPU-only local model constraint.

**Step 1 — CURRENT date gate** (`_apply_current_date_gate`): before routing
even happens, every article without a trustworthy (EXACT/DATE_ONLY)
publication date *inside* the rolling window is dropped outright. This moved
a pre-existing drop rule (event formation already required a known date)
**much earlier** in the pipeline — a real production run once let 279 of
300 "in-period" articles flow all the way through evidence creation,
routing, and competed for the scarce triage budget, before being dropped
only at the very last step. Moving the gate earlier means undated noise
never costs a single LLM call.

**Step 2 — deterministic structural gate** (`_passes_current_deterministic_gate`):
removes candidates no triage/deep-analysis could usefully reason about — an
empty title, a hub/index page whose extracted body is too thin (<400 chars)
to be a real article, or a page that extracted to nothing but
subscription-paywall marketing copy. **Deliberately never a keyword/topic
check** — the explicit, enforced principle here is that only triage/deep
analysis (the LLM) is allowed to decide topical relevance; a cheap
structural gate is only allowed to remove content that was never really
retrieved in the first place. (This same principle is why a later attempt
to add a *post-triage* keyword-override was reverted — see Section 12.3.)

**Step 3 — candidate pool construction & ranking**: one (article, evidence)
pair per distinct article (an article routed to multiple agents costs only
one triage slot — first-match-wins by a fixed agent order), ranked by
`_current_candidate_rank_key` with exact-dated candidates prioritized first
(so the scarce budget is spent on candidates that could even qualify for
the digest), then the same substantive/action-language/keyword/recency
composite as Morning/Evening.

**Step 4 — the durable candidate backlog is merged in** (see 6.6 below)
before the triage budget is spent — so every run, a resumed backlog
candidate competes on equal footing with fresh discoveries.

**Step 5 — domain-fair triage** (`_run_current_triage_by_domain`): the
bounded triage pool (`triage_batch_size × max_triage_batches`, default
3×8=24 slots) is split **fairly across specialist domains** (round-robin,
one slot per domain per round) before being split further into
**per-source-diverse, per-domain-pure batches** — so neither one prolific
source nor one specialist domain can consume the whole triage budget while
others go completely unrepresented, and no single LLM call ever mixes
candidates from more than one specialist's domain (each batch gets a
domain-specific hint, e.g. "financial institutions — AI adoption, strategy,
partnerships...").

**Step 6 — triage itself** (Section 6.4).

**Step 7 — domain-fair deep-analysis selection** (`_select_deep_analysis_winners`):
of however many candidates triage approved, only `max_deep_analysis` (a
small, deliberately conservative number — e.g. 4–8) get the expensive full
analysis call, again split fairly across domains, and *within* each
domain's allocation, ranked by the triage model's own confidence score
(descending) — not by pre-triage discovery order, which a real regression
once let win out incorrectly.

**Step 8 — deep analysis itself** (Section 6.5), strictly stopping the
instant the global budget is spent; every remaining triage winner is
explicitly reported as budget-skipped (never silently dropped).

### 6.4 Triage (`agents/triage.py`) — what it is, and what "confidence" means

Triage is a **fast, cheap, batched first-pass screen** — a single Ollama
call is given several candidates at once (a title + a small content
snippet, capped at 600 characters per candidate, deliberately much smaller
than deep analysis's 4000-character cap, because a batch carries several
candidates at once and must stay cheap in aggregate). For each one, it
answers exactly two questions: *is this relevant at all?* and *roughly what
kind of development is it?* It explicitly does **not** extract entities,
people, products, deal values, or write any analysis — that's deep
analysis's job, reserved for the small number of candidates triage approves.

**Triage is deliberately, explicitly recall-biased**, stated outright in its
system prompt: *"a false positive here costs one extra specialist call; a
false negative here is unrecoverable."* The prompt instructs: when in doubt
whether a dated, substantive development is specific enough, mark it
RELEVANT and let deep analysis make the stronger call. A candidate should
only be marked NOT relevant when it is *clearly* generic commentary, a
reference/evergreen page, routine news with zero AI/fintech angle, or a
navigation/listing page. This is why triage's relevant rate is typically
high (roughly 40–50% of what reaches it) — it is designed to over-admit,
not to make the final call.

**Triage confidence** is a single float in `[0.0, 1.0]`, produced directly
by the LLM alongside its relevant/not-relevant verdict and (optional)
event-type/category guess. It represents **the triage model's own
self-reported certainty in its screening verdict** — nothing more. It is
explicitly labelled in the UI as *"a preliminary relevance screen, not a
measure of the article's factual accuracy."* It is used for exactly one
downstream purpose: **ranking** *within* a domain's allocated deep-analysis
slots (the strongest-confidence triage winners from each domain get the
scarce deep-analysis budget first) — it is never itself a relevance
decision on its own, and it never feeds into the Event-level relevance/
confidence scores described in Section 7 (those are computed from entirely
separate, deep-analysis-stage signals).

A real production bug (fixed) is worth noting here: `qwen2.5:3b` sometimes
returns `confidence` on a 0–100 scale (e.g. `90`) instead of the documented
0.0–1.0 one. Before the fix, this single malformed field failed Pydantic
validation and **discarded the entire triage batch** — 13 of 24 real
candidates lost in one observed run, not just the one bad value. The fix:
any numeric value strictly greater than 1 and at most 100 is interpreted as
a percentage and rescaled (`90 → 0.90`); anything still out of range is
still correctly rejected.

### 6.5 Deep analysis — the specialist agents (`agents/base.py`, `agents/peer.py`, etc.)

Five specialist agents share one base class (`ResearchAgent`): **Peer**
(banks/financial institutions), **AIProduct** (AI/technology developments),
**Startup** (funding/venture activity), **Analyst** (consulting/research
firm publications — the only one with an extra structured output,
`AnalystFindings`: thesis, key findings, reported quantitative evidence
points, industry pattern, financial implication, strategic themes), and
**FinancialNews** (general catch-all for substantive financial/tech
reporting from an independent outlet).

Each agent's `analyze(article, evidence)` call:
1. Re-checks its own keyword pre-filter (defense in depth even after
   routing already restricted candidates).
2. Builds a prompt: source/publisher/date + the evidence body, **capped at
   4000 characters** (an isolated benchmark showed an ~11k-character article
   pushing a single CPU call to ~90 seconds — unbounded input size is the
   single biggest lever on local-LLM latency; the cap keeps the
   *beginning* of the article, since news/press-release writing is
   front-loaded with the who/what/when).
3. Calls the LLM with a JSON-Schema-constrained `AgentAssessment` response:
   relevance (label + reason), category, optional event_type, optional
   summary, `is_dated_development` (+ rationale), entities (typed:
   company/product/platform/model/technology/organization/person/location),
   people, products, deal_value, partnerships, technologies, geography, and
   two qualitative assessments (`financial_services_relevance`,
   `strategic_significance`, each low/medium/high + rationale).
4. Applies several **deterministic post-LLM guardrails** that can only ever
   *narrow* what the model claimed, never add to it (Section 6.6).
5. Returns a `ResearchItem`, or `None` if ultimately judged irrelevant.

**`nomura_relevance` is deliberately never asked of the model** — a
cost-reduction decision, confirmed to be read nowhere downstream except
being persisted; it's derived deterministically instead, mirroring
`strategic_significance`, so a required database column stays populated
without spending a third free-text LLM generation on it.

### 6.6 Deterministic post-LLM guardrails — why the model is never fully trusted

`qwen2.5:3b` does not reliably obey pure prompt instructions for two
grounding-critical judgments, so both are **re-derived from the evidence
text itself** after the call returns:

- **`is_dated_development`** (`_resolve_dated_development`): the model's own
  claim of "this is a dated, one-time development" is only trusted when the
  evidence text is independently corroborated by explicit
  announcement/release language (a large, deliberately generated set of
  verb-tense patterns for announce/launch/unveil/introduce — "today
  announced", "has launched", "is unveiling", bare present/past tense
  checked against recurring-cadence/historical-distance/future-intention
  guards so "announces its quarterly results every year" or "unveiled years
  ago" or "plans to announce" are never mistaken for a fresh, one-time
  development) **or** the article's own metadata date is precisely known
  (EXACT/DATE_ONLY). This can only ever demote True→False, never promote
  False→True. This directly controls whether the article's date can ever
  become an Event's `event_date` (Section 7) — an evergreen/reference page
  must never contribute a date just because *today's* retrieval happened to
  find it.
- **Partnership validation** (`_validate_partnerships_against_evidence`):
  a claimed partnership survives only if the evidence text contains real
  relational language ("partnered with", "in collaboration with", etc.)
  near that company's name — not just two company names appearing near each
  other.
- **Relevance override** (`_resolve_relevance_override`) — the opposite
  direction, a narrow *safety net*: if the model says IRRELEVANT but the
  evidence independently shows strong, dated, in-scope signal (multiple
  broad-scope category hits **and** dated-announcement language), the
  verdict is overridden to RELEVANT — but **always recorded, never hidden**
  (`ResearchItem.relevance_override` names exactly why).

### 6.7 LINK-grounded items — when there's no content at all

When retrieval could not get the article body at all (robots.txt,
anti-bot wall, browser-extraction failure), the system can still produce a
`ResearchItem` from **trusted discovery metadata alone** (title, source,
publication date) — never inventing a fact that wasn't in that metadata.
`is_dated_development` for a LINK item is set purely from
`publication_date_precision` (EXACT/DATE_ONLY only). The UI always discloses
this explicitly: *"Article content was unavailable... this development was
identified from its title and publication metadata only."* LINK-grounded
items never compete for triage/deep-analysis budget — zero LLM calls.

### 6.8 The candidate backlog — never losing a candidate to a one-run budget

Any candidate that passes every deterministic gate but whose run ran out of
triage or deep-analysis budget before reaching it is **never discarded** —
it is durably upserted into `candidate_backlog` and revalidated/retried by a
**later** run.

**`BacklogStatus`** values (every one an explicit, observed outcome — never
a guess):
- `pending_triage_budget` — never offered a triage slot at all.
- `pending_deep_analysis_budget` — triage already said relevant, but
  deep-analysis budget/domain-fair selection didn't reach it this run.
- `pending_retry` — a deep-analysis attempt was made and failed/timed out;
  recoverable, up to `backlog_max_attempts` (default 5).
- `completed` — deep analysis succeeded (names the resulting ResearchItem).
- `rejected_triage_irrelevant` — explicit triage verdict, irrelevant.
- `rejected_deep_analysis_irrelevant` — the full specialist call completed
  normally and genuinely judged it not relevant (never confused with a
  failed/timed-out attempt).
- `rejected_date_gate` — no longer falls inside the period being
  reconsidered (revalidated on every resume).
- `rejected_retry_exhausted` — too many recoverable failures/timeouts.
- `quarantined` — a malformed persisted row that couldn't be safely
  reconstructed; isolated so one bad row can never abort a whole run.

On every CURRENT run: a bounded share (`backlog_claim_limit`, default 8) of
pending rows is read, revalidated against *this* run's own period/
deterministic gate, and split into never-triaged (merged into the fresh
candidate pool, goes through triage normally) vs. already-triaged (a valid
persisted `triage_relevant=True` — bypasses triage entirely, straight into
the deep-analysis pool). A **bounded aging boost** (`_aging_boosted_rank_key`,
capped at 5 tiers) raises a repeatedly-deferred candidate's effective rank
each time it's deferred *or* fails, so an old backlog candidate eventually
rises to the top of its tier rather than being perpetually out-ranked by a
continuous stream of fresher candidates — but it never wins outright forever
once capped; it just competes on ordinary strength ranking after that.

---

## 7. Stage 6: Event Scoring — clustering, then deterministic scoring

### 7.1 Clustering (`events/clustering.py`, `events/similarity.py`)

Groups multiple `ResearchItem`s that describe the **same real-world event**
(e.g. three outlets all covering one product launch) into one `Event`. This
is a separate, later concern from Phase-2 article dedup (Section 5.3) — here
*different* articles with *different* text/wording are intentionally
grouped.

Uses **union-find** (transitive grouping: if A~B and B~C cluster, so does
A~C) over a **weighted multi-signal pairwise score**:

| Signal | Weight | What it measures |
|---|---|---|
| Entity overlap | 0.35 | Jaccard similarity over {company, extracted entities, partnerships, products} |
| Event type match | 0.15 | 1.0 same type, 0.5 either unknown, 0.0 different |
| Date proximity | 0.15 | 1.0 same day, linear decay to 0 over 5 days |
| Title similarity | 0.20 | token-set Jaccard over normalized titles |
| Content similarity | 0.15 | `difflib` ratio over agent-authored summaries |

Pairs cluster together if their combined score reaches **0.55**, *unless* a
**hard veto** fires first (checked before the score, never overridden by
it): a known, *different* event type; disjoint named products under the
same release-like event type (distinguishes "OpenAI releases Model A" from
"OpenAI releases Model B"); publication dates more than 14 days apart; or
zero shared entities at all (prevents clustering purely on generic topical
similarity — "do not simply cluster everything by topic" is an explicit
design rule). One narrow **override** exists in the opposite direction: a
near-verbatim (≥90% title-token overlap) same-day title match is treated as
overwhelming evidence of the same release even if the weighted score alone
would have missed it — added after a real case where the *same* PR Newswire
release, syndicated under two different numeric URLs, scored just under
threshold (0.51) because both items' summaries were empty and two
independent LLM calls extracted noticeably different entity lists from
near-identical text.

### 7.2 Event construction (`events/builder.py`)

From a cluster, deterministically picks:
- **Primary source**: ranked by source_type (primary > analyst > secondary
  > press_distribution), then source_priority, then earliest date — never
  an LLM decision.
- **Canonical title**: the primary item's title.
- **Primary company**: normally the primary item's own source name — but
  for a press-distribution wire service (PR Newswire, etc.), that name is
  the *distributor*, never the real subject company. A real defect once
  persisted `primary_company="PR Newswire"` for releases that were clearly
  about CaixaBank/KONST/Senao. Fixed: for press-distribution sources only,
  a company/organization-typed entity that the release's own headline
  actually names is promoted instead — never fabricated, only ever
  something already present in the evidence.
- **`event_date`**: the earliest `publication_date` among only the members
  whose `is_dated_development=True` — an evergreen/reference item's date is
  never allowed to stand in as the event's date. If no member qualifies, the
  event's date is honestly `None`, which (Section 7.3) permanently excludes
  it from any date-bounded digest rather than inventing a fallback date.
- **Related companies**: every company/organization-typed entity name
  across the cluster except the primary company, minus anything matching
  the configured non-company vocabulary (technologies/topics the LLM might
  have mislabeled, e.g. "Agentic AI" is a technology, never a company).

### 7.3 Deterministic scoring (`events/scoring.py`) — exactly how the numbers are calculated

**Every score here is a pure function of already-existing deterministic
data — never produced or guessed by the LLM, and never a network/LLM call.**
Weights come from `settings.yaml`'s `ranking.relevance`/`ranking.confidence`
(both independently validated to sum to 1.0).

**Relevance score** = weighted sum of:
- `ai_relevance` — the strongest (max across cluster members) combination
  of each item's topic-category multiplier (1.0 for any real AI bucket,
  0.4 for the catch-all `OTHER`) × its relevance-label score (RELEVANT=1.0,
  AMBIGUOUS=0.6, IRRELEVANT=0.0 — though IRRELEVANT items never reach
  clustering at all, since agents already drop them).
- `financial_services_relevance` — max across members of the qualitative
  level mapped to a number (LOW=0.25, MEDIUM=0.50, HIGH=0.75).
- `strategic_significance` — same mapping, same max-across-members logic.
- `recency` — a bucketed decay from the event's own date to *now*: 1.0
  same-day, 0.9 ≤3 days, 0.7 ≤7 days, 0.5 ≤14 days, 0.3 ≤30 days, 0.1 ≤90
  days, 0.0 beyond. A missing `event_date` scores 0.0 (unknown recency is
  never assumed recent); a future-dated event (clock skew) is clamped to
  age=0, never penalized.
- `novelty` — 1.0 unless an *earlier*-dated event in the **same scoring
  batch** already shares this event's primary company and event type (a
  recurring theme already covered earlier *in this run*), in which case
  0.4. (Documented limitation: this is relative to the current batch only
  — a rehash of something from a *previous* run isn't detected.)

**Confidence score** = weighted sum of:
- `primary_source_confirmation` — 1.0 primary source, 0.8 analyst, 0.5
  secondary, 0.3 press-distribution.
- `multiple_source_corroboration` — 1.0 for 3+ total sources, 0.6 for 2, 0.3
  for just 1.
- `publication_date_certainty` — binary: 1.0 if the event has a resolved
  `event_date`, else 0.0.
- `entity_extraction_confidence` — `min(1.0, distinct_names / 3)` across
  entities/people/products/technologies — 3+ distinct extracted names is
  "full" confidence.
- `event_classification_confidence` — 1.0 if an event_type was resolved,
  else 0.5.
- `content_completeness` — a 5-item checklist (has a summary / has an
  event_type / has entities / has technologies-or-partnerships-or-products /
  has geography) scored as the fraction present.

A hard **ceiling of 0.6** is applied to the confidence score when *every*
member of the event is LINK-grounded (metadata-only, no retrieved body) —
no matter how the individual components happen to land, a purely
metadata-based event can never claim full confidence.

Both final scores are clamped to `[0.0, 1.0]` and rounded to 4 decimal
places; both the per-component breakdown and the weighted total are
persisted (`scoring_breakdown_json`) for full auditability.

---

## 8. Stage 7: Report Generation (`agents/report.py`)

One more LLM call per Event (not per article) — `ReportAgent` reads the
Event and all its member `ResearchItem`s and produces a `ReportNarrative`:
`opening_context` (what happened), `details_narrative`, a bounded
`key_insights` list (2–4 short takeaways, each ≤280 characters, added as a
Tier A/Tier B quality-parity fix), `strategic_implications`, and
`nomura_opportunity`.

Two **defense-in-depth guardrails** apply regardless of what the prompt
says:
1. **Forbidden prescriptive phrasing** — `nomura_opportunity` fails schema
   validation outright if it contains "Nomura should/must/needs to/..." —
   the system is only ever allowed to frame things diplomatically ("may be
   relevant to Nomura's consideration of...").
2. **Partnership-assertion grounding** — a report sentence claiming a
   partnership survives only if every company name it mentions is already
   in the deterministically-validated `partnerships` fact list from Section
   6.6 — never because the narrative free-text asserts one on its own.

Every factual list (companies/people/products/sources) is deterministically
aggregated in Python from the Event's ResearchItems, not asked of the model
— the model is only ever asked for interpretation/narrative, never to
restate facts it could get wrong.

**Tier B** (CURRENT-only, Section 9.2) gets a parallel, lighter-weight
`TierBReportNarrative` (bounded character limits throughout — opening
context/strategic implications capped at 1500 chars, nomura_opportunity at
800) generated **on demand**, never during the scan itself — this is a
separate, newsletter-only feature (`reporting/tier_b_newsletter_resolution.py`),
cached by content hash so re-generation only happens if the underlying text
actually changed.

---

## 9. Stage 8: Digest Generation — and CURRENT's two-tier design

### 9.1 Morning/Evening digests

A straightforward `Digest`: the qualifying Events for the period, plus
deterministic theme groupings (`reporting/themes.py` — template-based, no
LLM).

### 9.2 CURRENT's Tier A / Tier B split

CURRENT shows **two tiers** of content, because the bounded funnel
(Section 6.3) deliberately produces more *triage-approved* candidates than
it can afford to *deep-analyze* in one run:

- **Tier A** ("digest events") — fully analyzed Events that qualify for the
  current period's digest. This is the standard `Digest`/`Event`/`Report`
  pipeline output, period- and date-filtered exactly like Morning/Evening.
- **Tier B** ("Relevant articles awaiting analysis") — candidates from the
  **latest run specifically** (not necessarily the same run Tier A is
  showing, which may fall back to an older still-usable snapshot) that
  triage already approved (`triage_relevant=1`) but deep analysis never
  reached (`status IN (pending_deep_analysis_budget, pending_retry)`).
  **Deliberately excludes** `pending_triage_budget` rows — Tier B is
  specifically "triage said yes, deep analysis hasn't gotten there yet,"
  not "haven't even screened it."

**Tier B cards deliberately carry no LLM-generated content at all** — no
relevance/confidence score, no generated summary, no strategic implications.
Everything shown is either copied verbatim from already-retrieved
Article/Evidence, or the backlog's own recorded status:
- **Excerpt**: a deterministic, no-LLM prose extract of the article's own
  text — strips a recognized wire-service dateline boilerplate
  ("DENVER, Oct. 15, 2026 /PRNewswire/ --"), then truncates to a configured
  character budget (550, raised from 400) preferring to end on a **complete
  sentence** rather than an arbitrary word boundary (fixed 2026-10-16 — a
  mid-sentence cutoff like "...gained fresh momentum on Oct…" reads as an
  uninformative dangling fragment; now it prefers the last `.`/`!`/`?`
  within the budget, falling back to the old word-boundary+ellipsis
  behavior only when no sentence end is near).
- **Key points**: the source's *own* authored bullet lines (`- `/`* `/`• `
  prefixed), reproduced verbatim, never generated — requires at least 2
  qualifying bullets to count as a real list.
- **Triage confidence**: shown, explicitly labelled as "a preliminary
  relevance screen, not a measure of the article's factual accuracy."

**Tier A's "Key Entities" box was removed** (2026-10-16, user feedback) —
it listed companies/partnerships/technologies/products pulled from the
Report's existing deterministic facts, but the user wanted the card's
visual weight on `opening_context` ("what happened"), not a bare entity
dump.

### 9.3 Why a zero-event digest doesn't always mean "nothing happened"

A documented, real forensic distinction: a zero-event CURRENT digest can
mean either "a clean scan genuinely found nothing qualifying" **or** "triage
found real candidates but deep analysis couldn't finish assessing them
(Ollama timeouts/failures) before the budget ran out." Both persist the same
`DigestPersistenceStatus.NO_QUALIFYING_EVENTS`, but the UI banner
distinguishes them (`no_qualifying_events_incomplete_analysis` vs.
`no_qualifying_events`) by checking whether `deep_analysis_timeouts +
deep_analysis_failures > 0` *and* `triage_relevant > 0` — because "nothing
of substance happened" and "analysis simply didn't finish" are very
different operational signals.

---

## 10. Stage 9: Persistence — the SQLite schema

Everything lands in one local SQLite file. Key tables (there are more, but
these carry the core intelligence):

- **`articles`** — every retrieved/normalized page: URL, canonical URL,
  domain, source metadata, title, author, publication date +
  `publication_date_precision`, raw/clean text, content hash, extraction
  method/status.
- **`evidence`** — the analysis-facing unit derived from an article
  (content, publisher, dates, source quality) — one-to-one with articles in
  practice today.
- **`research_items`** — one row per successful agent analysis: every
  `AgentAssessment` field, plus `is_dated_development`, `grounding_type`
  (content/link), `relevance_override` (when the safety-net fired),
  `run_id`.
- **`events`** — the clustered, scored output: canonical title, event type,
  primary/related companies, event_date, primary/secondary source refs
  (JSON), `relevance_score`/`confidence_score` + their full
  `scoring_breakdown_json`, `cluster_metadata_json` (which pairwise signals
  matched), `grounding_type`.
- **`event_research_items`** / **`event_articles`** — many-to-many join
  tables tying an Event back to its member ResearchItems/Articles.
- **`reports`** — one per Event: status, the narrative fields, sources,
  scores, or a failure reason/detail if generation failed.
- **`digests`** / **`digest_events`** — a digest's period/type/themes, and
  which events it contains, in order.
- **`pipeline_runs`** — per-run observability: `run_id` (deterministic,
  derived from digest_type+period, so a retry upserts rather than
  duplicates), status, a full `diagnostics_json` list (every per-candidate
  forensic trace mentioned throughout this document), and `counts_json`
  (every aggregate metric — `triage_relevant`, `deep_analysis_timeouts`,
  `candidates_pending_carried_forward`, etc.).
- **`candidate_backlog`** — the durable, cross-run candidate carry-forward
  described in Section 6.8: one row per article, its current
  `BacklogStatus`, persisted `triage_relevant`/`triage_confidence` (so a
  resumed candidate is never re-triaged), `attempt_count`/`defer_count`
  (feeds the aging boost), first/last-seen run ids.
  **Claim safety** is inherited for free from the pre-existing
  `current_run_lock` singleton (only one CURRENT run is ever in flight at
  once) — no separate claim/lease protocol needed.
- **`tier_b_reports`** — cached, on-demand LLM-generated Tier B newsletter
  reports, keyed by article_id, invalidated by content hash + schema
  version.
- **`current_run_lock`** — an explicit application-level mutex: CURRENT has
  no scheduled variant, so it's the one digest type an operator could
  accidentally trigger twice (e.g. double-clicking "Run Live Scan"),
  overwhelming the single local Ollama instance.
- **`scheduler_state`**, **`newsletter_selections`**, **`email_deliveries`**,
  **`email_settings`** — operational/UI-state singleton and tracking
  tables.

Persistence happens **once, atomically, at the very end of a run**
(`make_persistence_node`) — nothing is written mid-run, so a crash partway
through analysis never leaves a half-written run in the database.

---

## 11. The UI (Streamlit, `ui/app.py` + `ui/dashboard_data.py`)

A presentation-only layer — **never** retrieves, scrapes, clusters, scores,
or calls an LLM itself; every piece of data comes from `dashboard_data`,
which itself only ever reads through `PersistenceService`. Pages:

- **CURRENT** — the live-scan view: Tier A event cards + Tier B
  awaiting-analysis cards (Section 9.2), newsletter-selection toggles, a
  banner distinguishing "clean zero" from "incomplete analysis" zero
  (Section 9.3).
- **Morning Digest** / **Evening Digest** — the scheduled digest views.
- **Dashboard** — cross-digest overview.
- **Peers / AI Providers / Startups / Analysts** — entity-centric views.
  Peers/AI Providers/Analysts are driven by the configured source registry
  (every configured entity always shows, even with zero events this period);
  Startups is event-centric/discovery-driven (no fixed registry to
  enumerate against).
- **Cross-Market Themes** — the deterministic theme groupings.
- **CURRENT / Live Scan (Admin)** — operational controls (trigger a scan,
  monitor progress, see the current_run_lock state).

Every date badge in the UI is explicitly converted to the configured
display timezone (`Asia/Kolkata`) before rendering — a real fix (2026-10-16)
closed a bug where three specific card-rendering call sites called `.date()`
directly on the raw UTC-stored timestamp, so a UTC late-evening instant
(e.g. 22:00–23:59 UTC) rendered as the *next* calendar day compared to every
other timestamp in the same UI (which were already correctly converted).

---

## 12. Newsletter & Email Delivery

Users can curate a cross-digest selection of Events (Tier A) and/or Tier B
articles (`newsletter_selections` — a simple id+timestamp pointer table,
never a copy of the underlying content) and generate a newsletter: Tier A
items render from their already-generated Report; Tier B items get an
on-demand `TierBReportNarrative` generated and cached (Section 8). Delivery
(`delivery/renderer.py`, `delivery/mailer.py`, `delivery/service.py`) builds
an HTML+plaintext email and sends it over SMTP (credentials only ever from
the `SMTP_PASSWORD` env var). `email_deliveries` distinguishes `automatic`
sends (deduplicated per digest+recipient — never double-sent) from `manual`
newsletter sends (intentionally allowed to repeat, since a user might
legitimately resend).

---

## 13. Scheduling & Operations

`scheduling/period.py` holds the pure period-math (deliberately dependency-
light, so even the UI can compute "what period would a Morning digest cover
right now" without pulling in the full retrieval/agents/LLM stack).
`scheduling/jobs.py` wraps that math into the three real entry points
(`run_morning_digest`, `run_evening_digest`, `run_current_digest`) plus a
manual "Run Now" job, every one delegating entirely to `PipelineRunner` —
no retrieval/scoring/database logic lives in the scheduling layer itself.
`scheduling/scheduler.py` is the actual APScheduler-backed cron process;
`scheduling/daemon.py`/`cli.py` are the operational entry points.

CURRENT has **no scheduled variant at all** — it is always manually
triggered, which is exactly why it needed its own explicit mutex
(`current_run_lock`) rather than inheriting APScheduler's own
single-instance guarantee.

---

## 14. LLM infrastructure & its real-world performance characteristics

- **Provider**: Ollama, running entirely locally — no external API calls,
  no data ever leaves the machine.
- **Model**: `qwen2.5:3b` (a 3-billion-parameter model) — chosen for
  feasibility on consumer CPU hardware, not for maximum capability.
- **No GPU**: on this deployment, Ollama runs purely on CPU (a 13th-gen
  Intel i7, 16 logical cores). Every LLM call pays the full cost of
  CPU-bound token generation.
- **Structured output**: every call sends the real Pydantic-derived JSON
  Schema as Ollama's `format` field (not the bare `"json"` string) — this
  constrains generation to the exact response shape, but **schema-
  constrained decoding is measurably slower per output token than free-form
  generation**, which is why deep analysis (a much richer schema than
  triage's) is consistently the closer call against its own timeout.
- **Two dedicated timeout tiers**: `triage_timeout_seconds` (90s) and
  `deep_analysis_timeout_seconds` (140s, raised from a 90s default after
  real timeouts) — separate from the agents' own global 150s timeout used
  by Morning/Evening/manual runs, so CURRENT's bounded funnel can fail fast
  and move on without blocking the whole run on one slow call.
- **A confirmed, recurring real-world failure mode**: under system memory
  pressure (other applications — browsers, IDEs, chat apps — competing for
  the same limited RAM as Ollama's CPU inference), deep-analysis calls that
  are already close to their timeout ceiling tip over it. A repeat
  occurrence (documented from two separate real runs, 10 days apart) showed
  3 of 4 deep-analysis calls timing out while only 1 succeeded — this is a
  resource-contention problem, not a pipeline logic bug, and the two
  practical mitigations are: free up system memory before a scan, or
  further raise the timeout (which doesn't fix the underlying slowness,
  only gives it more room).

---

## 15. Design principles that recur throughout the codebase

These show up, explicitly stated, dozens of times across the source — they
are the actual "constitution" of this project's engineering decisions:

1. **Never invent a fact.** Every claim in a report/event/research item must
   trace back to something actually present in retrieved evidence or
   trusted metadata — never guessed, never filled in from general model
   knowledge.
2. **Recall over precision at every screening stage, correctness over
   recall at every final stage.** Triage is explicitly told a false
   positive is cheap and a false negative is unrecoverable. But the final,
   persisted claims (report narrative, partnership assertions,
   `is_dated_development`) are governed by strict deterministic guardrails
   that can only *narrow*, never *widen*, what the model claims.
3. **A cheap/structural gate must never make a topical-relevance decision.**
   That judgment belongs only to the LLM stages (triage/deep analysis).
   This exact principle caused a real regression to be *reverted*: an
   attempt to add a post-triage keyword-presence override (meant to catch
   one specific false positive) was found, via live production
   verification, to also silently reject several genuinely relevant
   articles (Anthropic, OpenAI-Atlassian, NVIDIA) simply because they used
   the word "AI" without hitting one of the curated multi-word ranking
   phrases. It was fully reverted rather than patched further, because a
   cheap substring check is structurally unsafe for this kind of judgment
   — the same reasoning that already barred the deterministic gate from
   doing this in the first place.
4. **Never silently drop a candidate.** Every rejection, deferral, timeout,
   and budget-skip is recorded with an explicit, inspectable reason
   (`diagnostics_json`, `candidate_backlog.reason`) — "unresolved" is a
   defensive fallback, never the intended outcome.
5. **Disagreements between the model and deterministic logic are always
   recorded, never hidden** — `relevance_override`, the
   `is_dated_development` rationale, and every diagnostic message exist so
   a human can audit exactly where and why the system overrode or
   distrusted the model.
6. **Fix root causes with live production data, not assumptions** — nearly
   every guardrail described in this document cites the exact real run ID
   and the exact real article that exposed the gap it closes.
7. **Prefer reverting over patching when a fix turns out unsafe.** Demonstrated
   directly in this project's own history (see principle 3) — when a fix is
   shown to cause collateral damage, the response is to undo it and report
   honestly, not to keep adding complexity trying to patch around the flaw.

---

## 16. Glossary

- **Article** — a retrieved, normalized page: URL, source metadata,
  extracted title/date/text.
- **Evidence** — the analysis-facing record derived from an Article; what
  the LLM and scoring layers actually read.
- **ResearchItem** — one agent's structured analysis of one Article/Evidence
  pair.
- **Event** — a cluster of one or more ResearchItems judged to describe the
  same real-world development, with deterministic relevance/confidence
  scores.
- **Digest** — a period-bounded collection of qualifying Events (+ themes),
  of type Morning/Evening/CURRENT.
- **Tier A** — CURRENT's fully-analyzed, digest-qualifying Events.
- **Tier B** — CURRENT's triage-approved-but-not-yet-deep-analyzed
  candidates, shown with deterministic (no-LLM) excerpts only.
- **Triage** — the cheap, batched, recall-biased first-pass relevant/
  not-relevant + rough category screen.
- **Deep analysis** — the full, expensive specialist LLM call that produces
  a ResearchItem.
- **Candidate backlog** — the durable, cross-run carry-forward of
  budget-deferred candidates.
- **Grounding type** — CONTENT (real retrieved body) vs. LINK (metadata
  only, body unavailable).
- **DatePrecision** — EXACT / DATE_ONLY / MONTH / UNKNOWN; how much to trust
  an extracted date.
- **`is_dated_development`** — whether evidence establishes a specific,
  dated action (vs. evergreen/reference content) — governs whether a date
  can ever become an Event's `event_date`.
- **Relevance override** — the narrow safety net that can promote a model's
  IRRELEVANT verdict to RELEVANT when deterministic signals strongly
  disagree; always recorded.
- **BacklogStatus** — the full lifecycle state of one candidate in the
  candidate backlog (see Section 6.8 for the complete list).
