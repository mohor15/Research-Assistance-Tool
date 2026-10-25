"""Scheduling (Phase 8): 08:00/20:00 Asia/Kolkata morning/evening digest runs.

`scheduling.jobs` (`run_morning_digest`/`run_evening_digest`) has no
APScheduler dependency at all and is independently callable/testable.
`scheduling.scheduler` (`ResearchScheduler`) is the actual APScheduler
wrapper — timing/orchestration only, never business logic; it invokes the
existing `PipelineRunner`/LangGraph pipeline rather than duplicating it.
`scheduling.cli` provides manual run support outside the scheduler.

This package stays a leaf import (nothing re-exported at the top level),
consistent with `models`/`agents`/`pipeline`: `scheduler.py` requires
`apscheduler` to import, while `jobs.py` does not, so importing
`research_assistant.scheduling.jobs` alone never requires it installed.
"""
