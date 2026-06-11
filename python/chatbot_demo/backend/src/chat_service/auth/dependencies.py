from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from chat_service.auth.api_keys import hash_api_key
from chat_service.auth.models import User
from chat_service.db.models import UserAccount
from chat_service.db.session import DBSessionDep

# auto_error=False so a missing/malformed header yields ``None`` and we can raise
# our own 401 with a consistent message, rather than HTTPBearer's default.
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    db_session: DBSessionDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)] = None,
) -> User:
    """Authenticate via ``Authorization: Bearer <api_key>``.

    Hashes the presented key and looks it up in the ``user_account`` table,
    returning the matching user's identity. The raw key is never stored — only
    its hash — so this compares hashes. Endpoints depend on the resulting
    :class:`User` to scope conversations per user.

    Raises:
        HTTPException: 401 if the header is missing or the key is unknown.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    result = await db_session.execute(
        select(UserAccount).where(UserAccount.api_key_hash == hash_api_key(credentials.credentials))
    )
    account = result.scalar_one_or_none()
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return User(id=account.id)


# Reusable annotated FastAPI dependency which resolves the current user.
CurrentUserDep = Annotated[User, Depends(get_current_user)]
