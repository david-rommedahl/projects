"""Root conftest — shared fixtures for the entire test suite."""

import os

# Set required env vars BEFORE any chat_service imports (CONFIG is instantiated lazily,
# but the db engine is built at import time from DatabaseConfig).
os.environ.setdefault("POSTGRES_USER", "testuser")
os.environ.setdefault("POSTGRES_PASSWORD", "testpass")
os.environ.setdefault("POSTGRES_DB", "testdb")
os.environ.setdefault("OPENAI_API_KEY", "fake-key")

from collections.abc import AsyncGenerator, Generator  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from chat_service.asgi import app  # noqa: E402
from chat_service.auth.dependencies import get_current_user  # noqa: E402
from chat_service.auth.models import User  # noqa: E402
from chat_service.db.session import get_async_session  # noqa: E402

TEST_USER = User(id="test-user")


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    """Provide a ``TestClient`` with DB and auth dependencies overridden.

    No domain endpoints exist yet, but the override seams are wired so future
    endpoint tests inherit a Postgres-free client. The session override yields
    ``None`` for now — replace it with a stub session once endpoints land.
    """

    async def _override_session() -> AsyncGenerator[None, None]:
        yield None

    app.dependency_overrides[get_async_session] = _override_session
    app.dependency_overrides[get_current_user] = lambda: TEST_USER

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()
