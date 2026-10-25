"""Phase 14: the actual persistent process that keeps `ResearchScheduler`
alive.

Run with:

    python -m research_assistant.scheduling.daemon
    python -m research_assistant.scheduling.daemon --db-path path/to.db --heartbeat-interval-seconds 5

This is the ONLY intended way to get real, automatic 08:00/20:00 IST
execution in production -- closing the forensic-diagnosis finding that
`ResearchScheduler` existed, fully built and tested, with zero real call
sites. It must run independently of Streamlit: the UI (`ui/app.py` via
`ui/dashboard_data.py::get_scheduler_status`) only ever READS the
persisted `scheduler_state` row; it never starts, owns, or is required for
keeping a scheduler alive. `--db-path` exists only so isolated/test runs
never touch the real database by accident -- production always resolves
the path from `config/settings.yaml`'s `storage.database_path`, exactly
like every other entry point in this project.

The manual one-shot CLI (`scheduling/cli.py`) is unchanged and remains the
correct tool for an on-demand run; this module is for continuous,
automatic operation only.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
from pathlib import Path
from typing import List, Optional

from research_assistant.config import get_config
from research_assistant.pipeline.runner import PipelineRunner
from research_assistant.scheduling.scheduler import DEFAULT_HEARTBEAT_INTERVAL_SECONDS, ResearchScheduler
from research_assistant.storage.database import Database
from research_assistant.utils.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the persistent research-assistant scheduler process.")
    parser.add_argument(
        "--db-path", default=None,
        help="Override the database path -- for isolated/test runs only. Production always uses "
             "config/settings.yaml's storage.database_path.",
    )
    parser.add_argument("--heartbeat-interval-seconds", type=int, default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS)
    return parser.parse_args(argv)


def build_scheduler(*, db_path: Optional[str] = None, heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS) -> ResearchScheduler:
    config = get_config()
    database_path = Path(db_path) if db_path else (PROJECT_ROOT / config.settings.storage.database_path)
    database = Database(database_path)
    database.initialize()
    runner = PipelineRunner.build(config=config, database=database)
    return ResearchScheduler(config, runner, heartbeat_interval_seconds=heartbeat_interval_seconds)


def _run_email_backfill_once(scheduler: ResearchScheduler) -> None:
    """Bounded, one-time discovery of digests that became persisted/
    eligible for automatic email delivery before this feature existed or
    before a prior SMTP failure was resolved (see
    `delivery.service.discover_and_deliver_pending` for the exact window/
    count/attempt bounds). Run exactly once at daemon startup -- never on
    every heartbeat tick -- and deliberately only from this one
    long-running process, so Streamlit and the daemon never run competing
    discovery workers concurrently. A failure here must never prevent the
    scheduler itself from starting or staying up."""
    if scheduler.persistence_service is None:
        return
    try:
        from research_assistant.delivery.service import discover_and_deliver_pending

        discover_and_deliver_pending(scheduler.persistence_service, scheduler.config)
    except Exception:
        logger.exception("daemon: email backfill discovery raised unexpectedly (scheduler remains alive)")


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    scheduler = build_scheduler(db_path=args.db_path, heartbeat_interval_seconds=args.heartbeat_interval_seconds)
    scheduler.start()
    _run_email_backfill_once(scheduler)
    print(f"scheduler_started pid={__import__('os').getpid()}", flush=True)

    stop_event = threading.Event()

    def _handle_signal(signum, _frame) -> None:
        logger.info("daemon: received signal %s, shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    if hasattr(signal, "SIGBREAK"):  # Windows Ctrl+Break -- the portable way to signal a console subprocess
        signal.signal(signal.SIGBREAK, _handle_signal)

    logger.info("daemon: scheduler running; waiting for a shutdown signal")
    stop_event.wait()
    scheduler.shutdown()
    logger.info("daemon: exiting cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
