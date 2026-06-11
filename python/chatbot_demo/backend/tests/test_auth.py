"""Tests for the API-key authentication dependency."""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from chat_service.auth.api_keys import hash_api_key
from chat_service.auth.dependencies import get_current_user
from chat_service.db.models import UserAccount
from tests.conftest import StubAsyncSession, StubResult


def _bearer(key: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=key)


async def test_resolves_user_from_valid_key(stub_session: StubAsyncSession) -> None:
    """A key whose hash matches a user resolves to that user's identity."""
    key = "secret-key"
    account = UserAccount(email="a@b.com", api_key_hash=hash_api_key(key))
    stub_session.execute_results = [StubResult([account])]

    user = await get_current_user(stub_session, _bearer(key))

    assert user.id == account.id


async def test_missing_credentials_401(stub_session: StubAsyncSession) -> None:
    """No Authorization header is a 401."""
    with pytest.raises(HTTPException) as exc:
        await get_current_user(stub_session, None)
    assert exc.value.status_code == 401


async def test_unknown_key_401(stub_session: StubAsyncSession) -> None:
    """A key that matches no user is a 401."""
    stub_session.execute_results = [StubResult([])]
    with pytest.raises(HTTPException) as exc:
        await get_current_user(stub_session, _bearer("nope"))
    assert exc.value.status_code == 401
