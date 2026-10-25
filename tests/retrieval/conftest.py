"""Pytest fixtures for retrieval tests. No test in this package touches the network."""

from __future__ import annotations

import pytest

from tests.retrieval.fakes import FakeSession
from tests.storage.conftest import database, persistence_service  # noqa: F401


@pytest.fixture
def fake_session_factory():
    return lambda responses: FakeSession(responses)
