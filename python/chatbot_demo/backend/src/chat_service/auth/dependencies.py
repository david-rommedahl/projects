from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from chat_service.auth.models import User


async def get_current_user(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> User:
    """Resolve the current user from the ``X-User-Id`` request header.

    A lightweight stub standing in for real authentication: it trusts the
    header value as the user's identity. This is the seam where real auth
    (e.g. a verified JWT) slots in later — endpoints depend on the resulting
    :class:`User` to scope conversations per user, so swapping the
    implementation here leaves the rest of the API unchanged.
    """
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-User-Id header",
        )
    return User(id=x_user_id)


# Reusable annotated FastAPI dependency which resolves the current user.
CurrentUserDep = Annotated[User, Depends(get_current_user)]
