"""User registration endpoint.

``POST /users`` registers a user by email and returns a freshly generated API key.
The key is shown **once** in the response and never stored (only its hash is). In
a production system you'd deliver the key out of band (e.g. email) instead of
returning it.

For the demo this is idempotent: registering an email that already exists doesn't
error — it re-issues a fresh key for that user and returns it. (The original key
can't be returned, since only its hash is kept, so re-registering rotates it.)
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import select

from chat_service.auth.api_keys import generate_api_key, hash_api_key
from chat_service.db.models import UserAccount
from chat_service.db.session import DBSessionDep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["users"])


class RegisterRequest(BaseModel):
    """Request body for ``POST /users``.

    Attributes:
        email: The email to register the user under. Must not already be registered.
    """

    email: str


class RegisterResponse(BaseModel):
    """Response body for ``POST /users``.

    Attributes:
        id: The new user's stable identifier.
        email: The registered email.
        api_key: The user's API key, shown once. Send it as
            ``Authorization: Bearer <api_key>`` on subsequent requests.
    """

    id: str
    email: str
    api_key: str


@router.post("/users")
async def register_user(request: RegisterRequest, db_session: DBSessionDep) -> RegisterResponse:
    """Register a user (or re-issue a key for an existing one) and return the key.

    Idempotent: if the email already exists, a fresh key is issued for that user
    rather than erroring — the original key can't be returned since only its hash
    is stored, so re-registering rotates it.
    """
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)

    result = await db_session.execute(select(UserAccount).where(UserAccount.email == request.email))
    account = result.scalar_one_or_none()
    if account is None:
        account = UserAccount(email=request.email, api_key_hash=key_hash)
        db_session.add(account)
    else:
        account.api_key_hash = key_hash  # rotate: re-issue a key for the existing user

    # Read the id before commit: the default async sessionmaker expires attributes
    # on commit, so touching account.id afterwards would trigger a lazy DB load
    # outside the await context and fail.
    user_id = account.id
    await db_session.commit()
    logger.info("registered/refreshed user id=%s email=%s", user_id, request.email)
    return RegisterResponse(id=user_id, email=request.email, api_key=api_key)
