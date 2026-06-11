"""Root conftest — shared fixtures for the entire test suite."""

import os

# Set required env vars BEFORE any chat_service imports (CONFIG is instantiated lazily,
# but the db engine is built at import time from DatabaseConfig).
os.environ.setdefault("POSTGRES_USER", "testuser")
os.environ.setdefault("POSTGRES_PASSWORD", "testpass")
os.environ.setdefault("POSTGRES_DB", "testdb")
os.environ.setdefault("OPENAI_API_KEY", "fake-key")

from collections.abc import AsyncGenerator, Generator  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from chat_service.asgi import app  # noqa: E402
from chat_service.auth.dependencies import get_current_user  # noqa: E402
from chat_service.auth.models import User  # noqa: E402
from chat_service.db.session import get_async_session  # noqa: E402

TEST_USER = User(id="test-user")


# ---------------------------------------------------------------------------
# Stub async DB session
# ---------------------------------------------------------------------------


class StubScalars:
    """Mimics the result of SQLAlchemy ``result.scalars()``."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def all(self) -> list[Any]:
        """Return all items."""
        return list(self._items)


class StubResult:
    """Mimics an SQLAlchemy ``execute()`` result over a list of items."""

    def __init__(self, items: list[Any] | None = None) -> None:
        self._items: list[Any] = items if items is not None else []

    def scalar_one_or_none(self) -> Any | None:
        """Return the single item, or ``None`` if empty."""
        return self._items[0] if self._items else None

    def scalars(self) -> StubScalars:
        """Return a :class:`StubScalars` wrapping the stored items."""
        return StubScalars(self._items)


class StubAsyncSession:
    """Lightweight fake of SQLAlchemy ``AsyncSession`` for unit tests.

    Configure per-test via ``execute_results`` — a list of :class:`StubResult`
    returned in order on each ``execute()`` call. Records ``add``/``commit``/
    ``rollback`` so tests can assert what the endpoint persisted.
    """

    def __init__(self) -> None:
        self.execute_results: list[StubResult] = []
        self._execute_call_index: int = 0
        self.added: list[Any] = []
        self.committed: bool = False
        self.rolled_back: bool = False

    async def execute(self, *_args: Any, **_kwargs: Any) -> StubResult:
        """Return the next pre-configured result, or an empty one."""
        if self._execute_call_index < len(self.execute_results):
            result = self.execute_results[self._execute_call_index]
            self._execute_call_index += 1
            return result
        return StubResult()

    def add(self, obj: Any) -> None:
        """Record a single object as added."""
        self.added.append(obj)

    async def commit(self) -> None:
        """Mark the session as committed."""
        self.committed = True

    async def rollback(self) -> None:
        """Mark the session as rolled back."""
        self.rolled_back = True


@pytest.fixture()
def stub_session() -> StubAsyncSession:
    """Provide a fresh :class:`StubAsyncSession` for each test."""
    return StubAsyncSession()


@pytest.fixture(autouse=True)
def _stub_postgres_checkpointer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the Postgres-backed checkpointer with an in-memory stand-in.

    The production lifespan opens an ``AsyncPostgresSaver`` over a psycopg
    connection pool (via ``open_checkpointer``) and calls ``.setup()`` — both
    require a live database. This autouse fixture swaps in an async context
    manager yielding an ``InMemorySaver`` so ``TestClient``'s lifespan succeeds
    and ``app.state.checkpointer`` holds an in-memory saver.
    """

    @asynccontextmanager
    async def _fake_open_checkpointer() -> AsyncGenerator[InMemorySaver, None]:
        yield InMemorySaver()

    monkeypatch.setattr("chat_service.asgi.open_checkpointer", _fake_open_checkpointer)


@pytest.fixture()
def client(stub_session: StubAsyncSession) -> Generator[TestClient, None, None]:
    """Provide a ``TestClient`` with the DB session and auth dependencies overridden."""

    async def _override_session() -> AsyncGenerator[StubAsyncSession, None]:
        yield stub_session

    app.dependency_overrides[get_async_session] = _override_session
    app.dependency_overrides[get_current_user] = lambda: TEST_USER

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()
