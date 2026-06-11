import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from chat_service.api import router as api_router
from chat_service.config import CONFIG
from chat_service.db.session import engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[BaseCheckpointSaver]:
    """Open the LangGraph checkpointer that persists conversation state in Postgres.

    Conversations are keyed by ``thread_id`` (our ``session_id``): the checkpointer
    stores each session's message history, so a follow-up request carrying the same
    session token continues where the previous turn left off.

    Uses a health-checked connection pool rather than a single long-lived
    connection so a reaped idle connection is transparently replaced. ``kwargs``
    replicate what ``AsyncPostgresSaver.from_conn_string`` configures (autocommit +
    dict rows + disabled prepared-statement cache); the checkpointer relies on all
    three. ``.setup()`` creates the checkpoint tables on first run.

    Factored out of :func:`lifespan` so tests can override this single seam with
    an in-memory saver instead of standing up Postgres.
    """
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        conninfo=CONFIG.checkpointer_postgres_url,
        open=False,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        check=AsyncConnectionPool.check_connection,
    )
    async with pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        yield checkpointer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage the application lifespan.

    Configures logging, opens the Postgres-backed checkpointer (see
    :func:`open_checkpointer`) and stores it on ``app.state`` for endpoint
    dependency injection, then disposes the database engine on shutdown.

    Args:
        app: The FastAPI application instance.
    """
    logging.basicConfig(
        level=CONFIG.log_level,
        force=True,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if CONFIG.log_level <= logging.DEBUG:
        logging.getLogger("sqlalchemy.engine").setLevel(CONFIG.log_level)
    async with open_checkpointer() as checkpointer:
        app.state.checkpointer = checkpointer
        yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Log request validation errors and return FastAPI's default 422 response."""
    logger.warning("Validation error: %s", exc.errors())
    return await request_validation_exception_handler(request, exc)


@app.get("/ping")
async def ping() -> dict[str, str]:
    """Health check endpoint."""
    return {"message": "pong"}


if __name__ == "__main__":
    uvicorn.run(app=app)
