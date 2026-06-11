from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column


class Base(DeclarativeBase, MappedAsDataclass):
    """Base declarative class for SQLAlchemy.

    Uses a Dataclass mixin to allow us to use additional functionality, like `default_factory`.
    """

    pass


def current_utc() -> datetime:
    """Return the current datetime in UTC.

    Used as a default factory and onupdate callable for timestamp columns.
    """
    return datetime.now(timezone.utc)


class UserAccount(Base):
    """A registered user, authenticated by an API key.

    The raw API key is generated at registration, returned to the user once, and
    never stored — only its SHA-256 hash is kept. Authentication hashes the
    presented key and looks it up here. ``id`` is the stable, non-secret identity
    used as :attr:`Conversation.owner_id`.

    Attributes:
        email: The user's registration email. Unique.
        api_key_hash: SHA-256 hex digest of the user's API key. Unique.
        id: Server-generated stable identifier (UUID string). Not part of initializer.
        created_at: Timestamp when the user registered. Not part of initializer.
    """

    __tablename__ = "user_account"

    email: Mapped[str] = mapped_column(unique=True, index=True)
    api_key_hash: Mapped[str] = mapped_column(unique=True, index=True)
    id: Mapped[str] = mapped_column(primary_key=True, default_factory=lambda: str(uuid4()), init=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default_factory=current_utc, init=False)


class Conversation(Base):
    """Ownership record for a chat session.

    One row per conversation, written when the session is first created. It maps
    a session token to its owning user, so requests can be authorised (a user may
    only continue their own conversations — the case's privacy requirement) and a
    user's conversations can be listed.

    The message transcript itself is **not** stored here: it lives in the LangGraph
    checkpointer tables, keyed by ``thread_id == session_id``. This table is the
    ownership/index layer; the checkpointer is the opaque transcript store.

    Attributes:
        session_id: The session token (a UUID string), also used as the
            checkpointer ``thread_id``. Primary key.
        owner_id: Identifier of the user who owns this conversation.
        title: Human-readable label for the conversation, derived from the first
            message when it's created (so the UI doesn't show the opaque token).
        created_at: Timestamp when the conversation was created. Not part of initializer.
    """

    __tablename__ = "conversation"

    session_id: Mapped[str] = mapped_column(primary_key=True)
    owner_id: Mapped[str] = mapped_column(index=True)
    title: Mapped[str] = mapped_column(default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default_factory=current_utc, init=False)
