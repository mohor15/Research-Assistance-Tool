"""Isolated real-Qwen smoke test -- NOT part of the standard test suite.

Run manually, only when you want to verify the real local Ollama
`qwen2.5:3b` end-to-end:

    python scripts/qwen_smoke_test.py

This lives outside `tests/` on purpose: `pyproject.toml`'s
`testpaths = ["tests"]` means `pytest`/the full suite never collects or
runs this file, and it is never imported by anything the suite imports.
It requires a real, reachable local Ollama server with `qwen2.5:3b`
pulled -- if that is not true, it prints a clear failure and exits
nonzero rather than hanging or silently skipping.

Checks, in order:
  1. Ollama is reachable (`/api/tags`) and `qwen2.5:3b` is installed.
  2. A real triage call (the actual `agents.triage.run_triage_batch`
     path) against the JumpCloud candidate -- the same fixture used by
     `tests/pipeline/test_current_end_to_end_regression.py`, but this
     time against the real model instead of a fake.
  3. A real deep-analysis call (the actual `StartupAgent.analyze` path)
     against the same candidate.
  4. Reports latency and whether the candidate would survive end-to-end
     (triage relevant -> deep analysis produces a non-None ResearchItem).

No cloud API is used or introduced. No OpenAI/Anthropic/Gemini/Azure
imports exist anywhere in this file.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

JUMPCLOUD_URL = (
    "https://www.prnewswire.com/news-releases/"
    "new-jumpcloud-research-finds-ai-access-is-outpacing-enterprise-governance-302901426.html"
)
JUMPCLOUD_TITLE = "New JumpCloud Research Finds AI Access Is Outpacing Enterprise Governance"
JUMPCLOUD_TEXT = (
    "LOUISVILLE, Colo., Oct. 7, 2026 /PRNewswire/ -- JumpCloud Inc. today released a new study, "
    "Enterprise AI Access Is Outpacing Control. The report shows that organizations are giving AI "
    "agents access to systems faster than they can secure them. More than half of organizations now "
    "use AI agents that can change systems, permissions, or files. However, among organizations "
    "using agents, 59% do not follow the same governance rules for human users."
)


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def main() -> None:
    import requests

    from research_assistant.agents.startup import StartupAgent
    from research_assistant.agents.triage import TriageCandidate, run_triage_batch
    from research_assistant.config import get_config
    from research_assistant.evidence.conversion import article_to_evidence
    from research_assistant.evidence.normalizer import normalize_article
    from research_assistant.llm.client import create_ollama_client
    from research_assistant.models.enums import AgentName, SourcePriority, SourceType
    from research_assistant.retrieval.models import ExtractedArticle, SourceCategory, SourceDefinition

    config = get_config()
    base_url = config.settings.system.llm.base_url
    model = config.settings.system.llm.model
    print(f"Ollama base_url={base_url} model={model}")

    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        _fail(f"Ollama not reachable at {base_url}: {exc}")
        return

    installed = [m["name"] for m in resp.json().get("models", [])]
    if model not in installed:
        _fail(f"Model {model!r} is not installed (found: {installed})")
        return
    print(f"OK: Ollama reachable, {model!r} installed.")

    source = SourceDefinition(
        name="PR Newswire", domain="prnewswire.com", urls=[JUMPCLOUD_URL],
        source_type=SourceType.PRESS_DISTRIBUTION, priority=SourcePriority.MEDIUM,
        category=SourceCategory.SECONDARY,
    )
    extracted = ExtractedArticle(url=JUMPCLOUD_URL, title=JUMPCLOUD_TITLE, text=JUMPCLOUD_TEXT, published_at=None)
    article = normalize_article(extracted, source=source)
    evidence = article_to_evidence(article)

    current_llm = config.settings.current_llm
    triage_client = create_ollama_client(config, timeout_seconds=current_llm.triage_timeout_seconds)
    deep_client = create_ollama_client(config, timeout_seconds=current_llm.deep_analysis_timeout_seconds)

    candidate = TriageCandidate(
        candidate_id=article.id, article=article, evidence=evidence, primary_agent=AgentName.STARTUP,
    )

    print("\n--- Real triage call ---")
    t0 = time.time()
    try:
        results = run_triage_batch(triage_client, [candidate])
    except Exception as exc:
        _fail(f"triage call raised: {exc}")
        return
    triage_latency = time.time() - t0
    result = results.get(article.id)
    if result is None:
        _fail("triage returned no result for the candidate id")
        return
    print(f"triage latency: {triage_latency:.2f}s, relevant={result.relevant}, confidence={result.confidence}")

    if not result.relevant:
        print("Triage marked the candidate irrelevant -- stopping here (this is real model output, not forced).")
        return

    print("\n--- Real deep-analysis call ---")
    agent = StartupAgent(deep_client, config)
    t0 = time.time()
    try:
        item = agent.analyze(article, evidence, llm_client=deep_client)
    except Exception as exc:
        _fail(f"deep-analysis call raised: {exc}")
        return
    deep_latency = time.time() - t0

    if item is None:
        print(f"deep-analysis latency: {deep_latency:.2f}s -- model marked it irrelevant (real output).")
        return

    print(f"deep-analysis latency: {deep_latency:.2f}s")
    print(f"ResearchItem produced: relevance={item.relevance.label.value}, is_dated_development={item.is_dated_development}")
    print(f"summary: {item.summary}")
    print("\nPASS: candidate survived triage -> deep analysis -> ResearchItem using the real local qwen2.5:3b.")


if __name__ == "__main__":
    main()
