"""Tests for the conversations endpoints (list + fetch messages)."""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from chat_service.agent import get_checkpointer
from chat_service.asgi import app
from chat_service.db.models import Conversation
from tests.conftest import TEST_USER, StubAsyncSession, StubResult


def _conversation(session_id: str, created_at: datetime, title: str = "") -> Conversation:
    c = Conversation(session_id=session_id, owner_id=TEST_USER.id, title=title)
    c.created_at = created_at
    return c


def test_list_conversations_returns_owned(client: TestClient, stub_session: StubAsyncSession) -> None:
    """Returns the caller's conversations as summaries, including the title."""
    t1 = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 6, 11, 10, 0, tzinfo=timezone.utc)
    stub_session.execute_results = [
        StubResult([_conversation("s2", t2, title="Newer chat"), _conversation("s1", t1, title="Older chat")])
    ]

    response = client.get("/api/v1/conversations")

    assert response.status_code == 200
    body = response.json()
    assert [c["session_id"] for c in body["conversations"]] == ["s2", "s1"]
    assert [c["title"] for c in body["conversations"]] == ["Newer chat", "Older chat"]
    assert body["conversations"][0]["created_at"] == "2026-06-11T10:00:00Z"


def test_list_conversations_empty(client: TestClient, stub_session: StubAsyncSession) -> None:
    """Returns an empty list when the caller has no conversations."""
    stub_session.execute_results = [StubResult([])]
    response = client.get("/api/v1/conversations")
    assert response.status_code == 200
    assert response.json() == {"conversations": []}


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}/messages
# ---------------------------------------------------------------------------


class _FakeCheckpointer:
    """Returns a canned checkpoint tuple (or None) from ``aget_tuple``."""

    def __init__(self, messages: list[Any] | None) -> None:
        self._messages = messages

    async def aget_tuple(self, _config: dict[str, Any]) -> Any | None:
        if self._messages is None:
            return None
        return SimpleNamespace(checkpoint={"channel_values": {"messages": self._messages}})


@pytest.fixture()
def _override_checkpointer() -> Any:
    """Override the checkpointer dependency with a configurable fake."""

    def _set(messages: list[Any] | None) -> None:
        app.dependency_overrides[get_checkpointer] = lambda: _FakeCheckpointer(messages)

    return _set


def test_get_messages_returns_transcript(
    client: TestClient, stub_session: StubAsyncSession, _override_checkpointer: Any
) -> None:
    """Returns the conversation transcript with mapped roles, in order."""
    stub_session.execute_results = [StubResult([Conversation(session_id="s1", owner_id=TEST_USER.id)])]
    _override_checkpointer([HumanMessage(content="hello"), AIMessage(content="hi there")])

    response = client.get("/api/v1/conversations/s1/messages")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "s1"
    assert body["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_get_messages_empty_when_no_checkpoint(
    client: TestClient, stub_session: StubAsyncSession, _override_checkpointer: Any
) -> None:
    """A conversation that exists but was never messaged returns an empty list."""
    stub_session.execute_results = [StubResult([Conversation(session_id="s1", owner_id=TEST_USER.id)])]
    _override_checkpointer(None)

    response = client.get("/api/v1/conversations/s1/messages")
    assert response.status_code == 200
    assert response.json() == {"session_id": "s1", "messages": []}


def test_get_messages_unknown_conversation_404(client: TestClient, stub_session: StubAsyncSession) -> None:
    """Fetching messages for an unknown conversation is a 404."""
    stub_session.execute_results = [StubResult([])]
    response = client.get("/api/v1/conversations/ghost/messages")
    assert response.status_code == 404


def test_get_messages_other_users_conversation_404(client: TestClient, stub_session: StubAsyncSession) -> None:
    """Fetching another user's conversation is a 404 (no existence leak)."""
    stub_session.execute_results = [StubResult([Conversation(session_id="s1", owner_id="someone-else")])]
    response = client.get("/api/v1/conversations/s1/messages")
    assert response.status_code == 404
