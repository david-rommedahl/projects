"""Tests for ``chat_service.config.Config``."""

import logging

import pytest

from chat_service.config import Config


def _set_required_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    """Set the minimum required env vars for ``Config()`` to instantiate."""
    defaults = {
        "POSTGRES_USER": "testuser",
        "POSTGRES_PASSWORD": "testpass",
        "POSTGRES_DB": "testdb",
        "OPENAI_API_KEY": "fake-key",
    }
    defaults.update(overrides)
    for key, val in defaults.items():
        monkeypatch.setenv(key, val)
    # Prevent reading .env file
    monkeypatch.setattr(Config, "model_config", {"env_file": None, "extra": "ignore"})


# ---------------------------------------------------------------------------
# log_level validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("level_str", "expected_int"),
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_log_level_validator_from_string(level_str: str, expected_int: int) -> None:
    """Validate that string log-level names are converted to their numeric equivalents."""
    assert Config.validate_log_level(level_str) == expected_int


def test_log_level_numeric(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate that a numeric log-level env var is used as-is."""
    _set_required_env(monkeypatch, LOG_LEVEL="10")
    config = Config()
    assert config.log_level == 10


# ---------------------------------------------------------------------------
# database_url computed property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "port", "user", "password", "db", "expected_suffix"),
    [
        ("localhost", "5432", "u", "p", "d", "u:p@localhost:5432/d"),
        ("db.example.com", "9999", "admin", "s3cret", "prod", "admin:s3cret@db.example.com:9999/prod"),
    ],
)
def test_database_url(
    monkeypatch: pytest.MonkeyPatch,
    host: str,
    port: str,
    user: str,
    password: str,
    db: str,
    expected_suffix: str,
) -> None:
    """Validate the computed ``database_url`` for various host/port/credential combos."""
    _set_required_env(
        monkeypatch,
        POSTGRES_HOST=host,
        POSTGRES_PORT=port,
        POSTGRES_USER=user,
        POSTGRES_PASSWORD=password,
        POSTGRES_DB=db,
    )
    config = Config()
    assert config.database_url == f"postgresql+asyncpg://{expected_suffix}"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that omitted optional fields fall back to their documented defaults."""
    _set_required_env(monkeypatch)
    config = Config()
    assert config.postgres_host == "localhost"
    assert config.postgres_port == 5432
    assert config.log_level == logging.INFO
    assert config.chat_model == "gpt-4o-mini"
