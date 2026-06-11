"""Chat endpoint.

A single ``POST /chat`` that takes a question and streams back the agent's answer
as a sequence of newline-delimited JSON (NDJSON) events. Conversations are owned
per user: a row in the ``conversation`` table maps each session token to its owner,
so a user can only continue their own conversations (the privacy requirement). The
message transcript itself is persisted by the LangGraph checkpointer, keyed by
``thread_id == session_id``.

Stream protocol: each line is a JSON object with a ``type`` discriminator:

- ``{"type": "token", "content": "..."}`` — a chunk of generated text.
- ``{"type": "error", "content": "..."}`` — generation failed mid-stream.
- ``{"type": "done"}`` — terminal event, always emitted last.

A stream is zero or more ``token`` events, optionally one terminal ``error``, and
always a final ``done`` — so a client can tell a completed answer apart from a
dropped connection.
"""

import logging
from collections.abc import AsyncIterator
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from chat_service.agent import CheckpointerDep, build_agent
from chat_service.api.conversations import authorize_owned_conversation
from chat_service.auth.dependencies import CurrentUserDep
from chat_service.auth.models import User
from chat_service.db.models import Conversation
from chat_service.db.session import DBSessionDep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    """Request body for ``POST /chat``.

    Attributes:
        question: The user's question to send to the agent.
        session_id: Conversation token. Omit it to start a new conversation (the
            server mints one and records ownership); supply a token returned by a
            previous call to continue that conversation. A supplied token that
            doesn't exist or isn't owned by the caller is rejected with 404.
    """

    question: str
    session_id: str | None = None


class StreamEvent(BaseModel):
    """Base class for NDJSON stream events."""

    def to_ndjson(self) -> str:
        """Serialize the event as a single JSON line for the NDJSON stream."""
        return self.model_dump_json() + "\n"


class TokenEvent(StreamEvent):
    """A chunk of generated assistant text."""

    type: Literal["token"] = "token"
    content: str


class ErrorEvent(StreamEvent):
    """Signals that generation failed mid-stream."""

    type: Literal["error"] = "error"
    content: str


class DoneEvent(StreamEvent):
    """Terminal event, always emitted last."""

    type: Literal["done"] = "done"


TITLE_MAX_LENGTH = 60


def _derive_title(question: str) -> str:
    """Derive a short conversation title from the first message.

    Uses the leading text of the question (whitespace collapsed, truncated) so the
    sidebar shows something meaningful without an LLM summarisation call.
    """
    collapsed = " ".join(question.split())
    if not collapsed:
        return "New conversation"
    if len(collapsed) <= TITLE_MAX_LENGTH:
        return collapsed
    return collapsed[:TITLE_MAX_LENGTH].rstrip() + "…"


async def _resolve_session(db_session: AsyncSession, user: User, session_id: str | None, question: str) -> str:
    """Resolve the conversation for this request, creating or authorising as needed.

    - ``session_id is None``: start a new conversation — mint a token, insert a
      :class:`Conversation` owned by ``user`` (titled from ``question``), and
      return it.
    - ``session_id`` supplied: it must exist and be owned by ``user``; otherwise
      raise 404. The same status is used for "not found" and "owned by someone
      else" so the endpoint doesn't leak which session tokens exist.

    The insert is committed before streaming begins so ownership is durable
    regardless of whether the stream completes.
    """
    if session_id is None:
        session_id = str(uuid4())
        db_session.add(Conversation(session_id=session_id, owner_id=user.id, title=_derive_title(question)))
        await db_session.commit()
        logger.info("created conversation session_id=%s owner=%s", session_id, user.id)
        return session_id

    await authorize_owned_conversation(db_session, user, session_id)
    return session_id


async def _stream_answer(question: str, session_id: str, checkpointer: BaseCheckpointSaver) -> AsyncIterator[str]:
    """Yield the agent's answer as a stream of NDJSON events.

    The agent is built with the shared checkpointer and invoked against the
    ``session_id`` thread, so the conversation history persists in Postgres and a
    follow-up request with the same session token continues the conversation.

    Yields a :class:`TokenEvent` per non-empty assistant chunk. Any mid-stream
    failure (model timeout, rate limit, upstream error) is caught and surfaced as
    a single :class:`ErrorEvent` rather than abruptly truncating the response —
    the HTTP status is already ``200`` and headers are flushed by the time
    streaming starts, so an error can only be reported in-band. A :class:`DoneEvent`
    is always emitted last so the client can distinguish completion from a dropped
    connection.
    """
    agent = build_agent(checkpointer=checkpointer)
    try:
        async for chunk, _metadata in agent.astream(
            {"messages": [HumanMessage(content=question)]},
            {"configurable": {"thread_id": session_id}},
            stream_mode="messages",
        ):
            # ``messages`` mode yields (message_chunk, metadata) for every node that
            # emits messages. We only forward assistant text; tool/other chunks (none
            # yet, but future-proof) and empty deltas are skipped.
            if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str) and chunk.content:
                yield TokenEvent(content=chunk.content).to_ndjson()
    except Exception:
        logger.exception("error generating answer for session_id=%s", session_id)
        yield ErrorEvent(content="An error occurred while generating the response.").to_ndjson()
    yield DoneEvent().to_ndjson()


@router.post("/chat")
async def chat(
    request: ChatRequest,
    user: CurrentUserDep,
    db_session: DBSessionDep,
    checkpointer: CheckpointerDep,
) -> StreamingResponse:
    """Answer a question, streaming the response as NDJSON events.

    Resolves the conversation against the authenticated user (creating a new one
    when ``session_id`` is omitted, or authorising an existing one), then streams
    the agent's answer as ``token`` / ``error`` / ``done`` events (see the module
    docstring). The resolved session token is returned in the ``X-Session-Id``
    header for the client to echo back on the next turn.
    """
    session_id = await _resolve_session(db_session, user, request.session_id, request.question)
    logger.info("chat request session_id=%s owner=%s", session_id, user.id)
    return StreamingResponse(
        _stream_answer(request.question, session_id, checkpointer),
        media_type="application/x-ndjson",
        headers={
            "X-Session-Id": session_id,
            # Defeat response buffering (e.g. by reverse proxies) so events reach
            # the client as they're produced rather than all at once.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
