"""Tests for the ``POST /api/v1/chat`` NDJSON streaming endpoint."""

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from chat_service.db.models import Conversation
from tests.conftest import TEST_USER, StubAsyncSession, StubResult


class _FakeAgent:
    """Stand-in for a compiled agent whose ``astream`` yields canned chunks.

    Records the ``config`` (second positional arg) it was invoked with, so tests
    can assert the session_id was threaded through as the ``thread_id``.
    """

    last_config: dict[str, Any] | None = None

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    async def astream(
        self, _input: Any, config: dict[str, Any] | None = None, **_kwargs: Any
    ) -> AsyncIterator[tuple[Any, dict[str, Any]]]:
        type(self).last_config = config
        for token in self._tokens:
            yield AIMessageChunk(content=token), {}


@pytest.fixture()
def _stub_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``build_agent`` so the endpoint streams canned tokens, not a real LLM.

    Accepts (and ignores) the ``checkpointer`` kwarg the endpoint passes.
    """
    _FakeAgent.last_config = None
    monkeypatch.setattr(
        "chat_service.api.chat.build_agent",
        lambda **_kwargs: _FakeAgent(["Hello", ", ", "world"]),
    )


def _events(response: Any) -> list[dict[str, Any]]:
    """Parse an NDJSON response body into a list of event dicts."""
    return [json.loads(line) for line in response.text.splitlines() if line]


def _text(events: list[dict[str, Any]]) -> str:
    """Concatenate the content of token events."""
    return "".join(e["content"] for e in events if e["type"] == "token")


def _owned_conversation(session_id: str) -> Conversation:
    """A Conversation row owned by the test user."""
    return Conversation(session_id=session_id, owner_id=TEST_USER.id)


# ---------------------------------------------------------------------------
# New conversation (session_id omitted -> mint + insert)
# ---------------------------------------------------------------------------


def test_chat_starts_new_conversation(
    client: TestClient, stub_session: StubAsyncSession, _stub_agent: None
) -> None:
    """Omitting session_id mints a UUID, streams the answer as NDJSON, persists ownership."""
    response = client.post("/api/v1/chat", json={"question": "hi"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")

    events = _events(response)
    assert _text(events) == "Hello, world"
    assert events[-1] == {"type": "done"}  # always terminated by a done event
    assert not any(e["type"] == "error" for e in events)

    # A valid UUID token is minted and returned for the client to echo back.
    session_id = response.headers["X-Session-Id"]
    UUID(session_id)

    # A Conversation row owned by the caller was added and committed.
    assert len(stub_session.added) == 1
    added = stub_session.added[0]
    assert isinstance(added, Conversation)
    assert added.session_id == session_id
    assert added.owner_id == TEST_USER.id
    assert stub_session.committed is True


def test_chat_threads_minted_session_id_to_checkpointer(client: TestClient, _stub_agent: None) -> None:
    """The minted session_id is passed to the agent as the checkpointer thread_id."""
    response = client.post("/api/v1/chat", json={"question": "hi"})
    session_id = response.headers["X-Session-Id"]
    assert _FakeAgent.last_config == {"configurable": {"thread_id": session_id}}


# ---------------------------------------------------------------------------
# Continuing an owned conversation
# ---------------------------------------------------------------------------


def test_chat_continues_owned_conversation(
    client: TestClient, stub_session: StubAsyncSession, _stub_agent: None
) -> None:
    """A session_id owned by the caller is accepted and continued (no new row)."""
    stub_session.execute_results = [StubResult([_owned_conversation("abc")])]

    response = client.post("/api/v1/chat", json={"question": "hi", "session_id": "abc"})

    assert response.status_code == 200
    assert _text(_events(response)) == "Hello, world"
    assert response.headers["X-Session-Id"] == "abc"
    assert _FakeAgent.last_config == {"configurable": {"thread_id": "abc"}}
    # Continuation reads + authorises only — nothing inserted.
    assert stub_session.added == []


# ---------------------------------------------------------------------------
# Mid-stream error handling
# ---------------------------------------------------------------------------


def test_chat_emits_error_event_on_failure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A mid-stream failure surfaces as an error event followed by a done event."""

    class _FailingAgent:
        async def astream(self, _input: Any, _config: Any = None, **_kwargs: Any) -> AsyncIterator[Any]:
            yield AIMessageChunk(content="partial"), {}
            raise RuntimeError("upstream blew up")

    monkeypatch.setattr("chat_service.api.chat.build_agent", lambda **_kwargs: _FailingAgent())

    response = client.post("/api/v1/chat", json={"question": "hi"})

    # Stream already started (200, headers flushed) — the failure is reported in-band.
    assert response.status_code == 200
    events = _events(response)
    assert _text(events) == "partial"
    assert events[-2]["type"] == "error"
    assert events[-1] == {"type": "done"}


# ---------------------------------------------------------------------------
# Authorisation failures (strict 404)
# ---------------------------------------------------------------------------


def test_chat_unknown_session_id_404(client: TestClient, stub_session: StubAsyncSession, _stub_agent: None) -> None:
    """A supplied session_id that doesn't exist is rejected with 404."""
    stub_session.execute_results = [StubResult([])]  # not found
    response = client.post("/api/v1/chat", json={"question": "hi", "session_id": "ghost"})
    assert response.status_code == 404
    assert stub_session.added == []


def test_chat_other_users_session_id_404(
    client: TestClient, stub_session: StubAsyncSession, _stub_agent: None
) -> None:
    """A session_id owned by a different user is rejected with 404 (no existence leak)."""
    other = Conversation(session_id="abc", owner_id="someone-else")
    stub_session.execute_results = [StubResult([other])]
    response = client.post("/api/v1/chat", json={"question": "hi", "session_id": "abc"})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_chat_requires_question(client: TestClient, _stub_agent: None) -> None:
    """Omitting ``question`` is a 422 validation error."""
    response = client.post("/api/v1/chat", json={})
    assert response.status_code == 422
