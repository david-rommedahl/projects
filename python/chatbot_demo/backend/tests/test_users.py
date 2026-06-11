"""Tests for the ``POST /api/v1/users`` registration endpoint."""

from fastapi.testclient import TestClient

from chat_service.auth.api_keys import hash_api_key
from chat_service.db.models import UserAccount
from tests.conftest import StubAsyncSession, StubResult


def test_register_new_user_returns_api_key_and_stores_hash(
    client: TestClient, stub_session: StubAsyncSession
) -> None:
    """Registering a new email returns the key once and persists only its hash."""
    response = client.post("/api/v1/users", json={"email": "alice@example.com"})

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "alice@example.com"
    assert body["id"]
    assert body["api_key"]

    # One UserAccount added and committed, storing the hash — never the raw key.
    assert len(stub_session.added) == 1
    account = stub_session.added[0]
    assert isinstance(account, UserAccount)
    assert account.email == "alice@example.com"
    assert account.api_key_hash == hash_api_key(body["api_key"])
    assert account.api_key_hash != body["api_key"]
    assert stub_session.committed is True


def test_register_existing_email_rotates_key_for_same_user(
    client: TestClient, stub_session: StubAsyncSession
) -> None:
    """Registering an existing email returns that user with a freshly rotated key."""
    existing = UserAccount(email="alice@example.com", api_key_hash="old-hash")
    stub_session.execute_results = [StubResult([existing])]

    response = client.post("/api/v1/users", json={"email": "alice@example.com"})

    assert response.status_code == 200
    body = response.json()
    # Same user (same id), new working key — no new row added.
    assert body["id"] == existing.id
    assert body["email"] == "alice@example.com"
    assert stub_session.added == []
    # The existing user's stored hash was rotated to match the newly issued key.
    assert existing.api_key_hash == hash_api_key(body["api_key"])
    assert existing.api_key_hash != "old-hash"
    assert stub_session.committed is True
