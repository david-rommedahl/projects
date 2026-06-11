"""Conversation endpoints.

``GET /conversations`` lists the authenticated user's conversations;
``GET /conversations/{conversation_id}/messages`` returns one conversation's
transcript. Both read the ownership table (``conversation``), so a user only ever
sees their own sessions — never anyone else's (the privacy requirement).
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from chat_service.agent import CheckpointerDep
from chat_service.auth.dependencies import CurrentUserDep
from chat_service.auth.models import User
from chat_service.db.models import Conversation
from chat_service.db.session import DBSessionDep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversations"])


async def authorize_owned_conversation(db_session: AsyncSession, user: User, session_id: str) -> Conversation:
    """Return the conversation if it exists and is owned by ``user``, else raise 404.

    Centralises the privacy rule: a user may only act on their own conversations.
    The same 404 is returned for "not found" and "owned by someone else" so the
    endpoint doesn't leak which session tokens exist.
    """
    result = await db_session.execute(select(Conversation).where(Conversation.session_id == session_id))
    conversation = result.scalar_one_or_none()
    if conversation is None or conversation.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


class ConversationSummary(BaseModel):
    """A single conversation owned by the caller.

    Attributes:
        session_id: The session token; pass it as ``session_id`` to ``POST /chat``
            to continue this conversation.
        title: Human-readable label derived from the conversation's first message.
        created_at: When the conversation was created.
    """

    session_id: str
    title: str
    created_at: datetime


class ConversationsResponse(BaseModel):
    """Response body for ``GET /conversations``.

    Attributes:
        conversations: The caller's conversations, newest first.
    """

    conversations: list[ConversationSummary]


@router.get("/conversations")
async def list_conversations(user: CurrentUserDep, db_session: DBSessionDep) -> ConversationsResponse:
    """List the authenticated user's conversations, newest first."""
    result = await db_session.execute(
        select(Conversation).where(Conversation.owner_id == user.id).order_by(Conversation.created_at.desc())
    )
    conversations = result.scalars().all()
    logger.info("listing %d conversation(s) for owner=%s", len(conversations), user.id)
    return ConversationsResponse(
        conversations=[
            ConversationSummary(session_id=c.session_id, title=c.title, created_at=c.created_at)
            for c in conversations
        ]
    )


# Maps LangChain message types to the chat-facing roles. Anything unmapped
# (e.g. "tool") falls through to its raw type — there are no tools yet.
_ROLE_BY_TYPE = {"human": "user", "ai": "assistant", "system": "system"}


class Message(BaseModel):
    """A single message in a conversation transcript.

    Attributes:
        role: Who produced the message — ``"user"``, ``"assistant"`` or ``"system"``.
        content: The message text.
    """

    role: str
    content: str


class ConversationMessagesResponse(BaseModel):
    """Response body for ``GET /conversations/{conversation_id}/messages``.

    Attributes:
        session_id: The conversation's session token.
        messages: The transcript in chronological order.
    """

    session_id: str
    messages: list[Message]


def _to_message(message: BaseMessage) -> Message:
    """Convert a stored LangChain message into the API representation."""
    content = message.content if isinstance(message.content, str) else str(message.content)
    return Message(role=_ROLE_BY_TYPE.get(message.type, message.type), content=content)


@router.get("/conversations/{session_id}/messages")
async def get_conversation_messages(
    session_id: str,
    user: CurrentUserDep,
    db_session: DBSessionDep,
    checkpointer: CheckpointerDep,
) -> ConversationMessagesResponse:
    """Fetch the message transcript for one of the caller's conversations.

    ``conversation_id`` is the session token; it maps directly to the
    checkpointer ``thread_id``. Authorises ownership first (404 if the
    conversation doesn't exist or belongs to another user), then reads the
    transcript from the checkpointer. A conversation with no checkpoint yet
    (created but never messaged) returns an empty list.
    """
    await authorize_owned_conversation(db_session, user, session_id)

    checkpoint_tuple = await checkpointer.aget_tuple({"configurable": {"thread_id": session_id}})
    raw_messages: list[BaseMessage] = []
    if checkpoint_tuple is not None:
        raw_messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
    logger.info("fetched %d message(s) for session_id=%s", len(raw_messages), session_id)
    return ConversationMessagesResponse(
        session_id=session_id,
        messages=[_to_message(m) for m in raw_messages],
    )
