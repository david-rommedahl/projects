from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from chat_service.config import DatabaseConfig

# Uses DatabaseConfig (not the full runtime Config) so that importing the
# db package from the Alembic migration container doesn't pull the OpenAI
# required fields into validation. Runtime callers see identical behaviour —
# Config inherits DatabaseConfig and reads the same env vars. The throwaway
# instance is discarded after the engine is built.
engine = create_async_engine(DatabaseConfig().database_url)
AsyncSessionLocal = async_sessionmaker(bind=engine)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional async database session via FastAPI dependency injection.

    Yields an AsyncSession that automatically commits on success
    and rolls back on exception. The session is closed on exit.
    """
    async with AsyncSessionLocal() as async_session:
        try:
            yield async_session
            await async_session.commit()
        except Exception:
            await async_session.rollback()
            raise


# Reusable annotated FastAPI dependency which fetches the DB session
DBSessionDep = Annotated[AsyncSession, Depends(get_async_session)]
