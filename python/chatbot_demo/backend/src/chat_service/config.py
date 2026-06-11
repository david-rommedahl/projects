import logging
from functools import cache
from typing import Any
from urllib.parse import quote_plus

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database-only settings.

    Kept as its own class so the Alembic migration container can instantiate
    it without having to supply the OpenAI key the full runtime ``Config``
    requires. Migrations only need to compose a connection URL.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str
    postgres_password: str
    postgres_db: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def base_db_url(self) -> str:
        """Construct PostgreSQL connection URL, without prefix."""
        return (
            f"{quote_plus(self.postgres_user)}:{quote_plus(self.postgres_password)}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Construct the async PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.base_db_url}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def checkpointer_postgres_url(self) -> str:
        """Construct the plain (psycopg-compatible) PostgreSQL connection URL.

        Used by the LangGraph ``AsyncPostgresSaver`` checkpointer, which uses
        psycopg v3 internally and does not accept the ``+asyncpg`` dialect prefix.
        """
        return f"postgresql://{self.base_db_url}"


class Config(DatabaseConfig):
    log_level: int = logging.INFO

    openai_api_key: str
    chat_model: str = "gpt-5.4-1"

    # Optional override for the OpenAI-compatible base URL. Leave unset to use
    # OpenAI directly (api.openai.com). Point it at an Azure AI Foundry resource's
    # OpenAI v1 surface (e.g. "https://<resource>.services.ai.azure.com/openai/v1")
    # to use an internal Azure deployment instead — the protocol is identical, so
    # only base_url + api_key + chat_model (the deployment name) change.
    openai_base_url: str | None = None

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, value: Any) -> int:
        """Allow the user to specify log level as a string.

        Args:
            value: The log level, either as a numeric level or a string name (e.g. "DEBUG").
        """
        if isinstance(value, str):
            # mode="before" means we see the raw env string, which may be a
            # numeric level ("10") or a level name ("DEBUG").
            if value.isdigit():
                return int(value)
            return logging._nameToLevel[value.upper()]
        return value


@cache
def _get_runtime_config() -> Config:
    """Lazily instantiate the runtime Config on first access.

    Wrapped in `@cache` so every `CONFIG` reference returns the same
    singleton instance for the lifetime of the process.
    """
    return Config()


def __getattr__(name: str) -> Any:
    """Module-level PEP 562 hook.

    Lazy-resolve `CONFIG` on first access so that merely importing
    something else from this module (e.g. `DatabaseConfig` in the
    Alembic migration container) does not trigger full Settings
    validation of runtime-only fields like OPENAI_API_KEY.
    """
    if name == "CONFIG":
        return _get_runtime_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
