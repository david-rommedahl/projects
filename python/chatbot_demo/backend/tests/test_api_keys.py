"""Tests for API key generation and hashing."""

from chat_service.auth.api_keys import generate_api_key, hash_api_key


def test_generated_keys_are_unique_and_urlsafe() -> None:
    """Each generated key is distinct and URL-safe."""
    keys = {generate_api_key() for _ in range(100)}
    assert len(keys) == 100
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    assert all(set(k) <= allowed for k in keys)


def test_hash_is_deterministic_and_not_the_key() -> None:
    """Hashing is stable for a given key and never returns the raw key."""
    key = "some-api-key"
    assert hash_api_key(key) == hash_api_key(key)
    assert hash_api_key(key) != key
    assert len(hash_api_key(key)) == 64  # sha256 hex digest


def test_distinct_keys_hash_differently() -> None:
    """Different keys produce different hashes."""
    assert hash_api_key("a") != hash_api_key("b")
