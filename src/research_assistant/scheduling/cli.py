"""Manual run support (Phase 8 Part 15) — run a digest immediately, without
waiting for the scheduler. Development/demonstration use only. This is a
thin CLI, not Streamlit — no dashboard code calls into this.

Usage:
    python -m research_assistant.scheduling.cli --digest-type morning
    python -m research_assistant.scheduling.cli --digest-type evening
    python -m research_assistant.scheduling.cli --digest-type manual

`--digest-type manual` (Phase 1) covers today's business-day start (the
configured `morning_digest_hour`) through the current instant — e.g. a
21:30 IST run covers 08:00 IST today .. 21:30 IST today. See
`scheduling.jobs.compute_manual_run_period` for the exact semantics. This
is a manual trigger only; no scheduling or email delivery is implemented.
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from research_assistant.pipeline.runner import PipelineRunner, PipelineRunResult
from research_assistant.scheduling.jobs import (
    run_current_digest,
    run_evening_digest,
    run_manual_digest,
    run_morning_digest,
)

_JOBS = {
    "morning": run_morning_digest,
    "evening": run_evening_digest,
    "manual": run_manual_digest,
    "current": run_current_digest,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually run a morning or evening research digest.")
    parser.add_argument("--digest-type", choices=sorted(_JOBS), required=True)
    return parser.parse_args(argv)


def run(digest_type: str, *, runner: Optional[PipelineRunner] = None) -> PipelineRunResult:
    """Runs the named job against a real `PipelineRunner.build()` unless one
    is injected — the injection point is what makes this testable without a
    live Ollama/network."""
    runner = runner or PipelineRunner.build()
    job = _JOBS[digest_type]
    return job(runner)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    result = run(args.digest_type)
    print(f"run_id={result.run_id} status={result.status.value} counts={result.counts}")
    if result.diagnostics:
        print(f"diagnostics ({len(result.diagnostics)}):")
        for diagnostic in result.diagnostics:
            print(f"  [{diagnostic.severity}] {diagnostic.stage}: {diagnostic.message}")

    return 0 if result.status.value != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
