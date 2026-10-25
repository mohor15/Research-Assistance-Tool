"""SQLite persistence for articles, evidence, research items, events,
reports, and digests (Phase 7).

`Database` (connection + idempotent schema init) and `PersistenceService`
(the coordinating entry point) are the two things later phases should
import; individual repositories under `storage.repositories` are
persistence/query infrastructure only — no LLM logic, no scraping logic,
no business reasoning.
"""

from research_assistant.storage.database import CURRENT_SCHEMA_VERSION, Database
from research_assistant.storage.persistence_service import PersistenceService

__all__ = ["Database", "PersistenceService", "CURRENT_SCHEMA_VERSION"]
