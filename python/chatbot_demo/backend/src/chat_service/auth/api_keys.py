"""API key generation and hashing.

Keys are high-entropy random tokens, so a fast hash (SHA-256) is appropriate for
storage — unlike user-chosen passwords, they don't need a slow KDF. Only the hash
is persisted; the raw key is shown to the user once at registration.
"""

import hashlib
import secrets


def generate_api_key() -> str:
    """Generate a new random URL-safe API key."""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Return the SHA-256 hex digest of an API key, for storage and lookup."""
    return hashlib.sha256(api_key.encode()).hexdigest()
