"""Re-exposes the storage-package fixtures for ui-package tests, so
dashboard_data tests get a real (temp-directory) database/PersistenceService
without duplicating fixture definitions."""

from tests.storage.conftest import database, persistence_service  # noqa: F401
