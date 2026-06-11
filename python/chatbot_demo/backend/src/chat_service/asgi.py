import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from chat_service.api import router as api_router
from chat_service.config import CONFIG
from chat_service.db.session import engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Manage the application lifespan.

    Configures logging on startup and disposes the database engine on shutdown.

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
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Log request validation errors and return FastAPI's default 422 response."""
    logger.warning("Validation error: %s", exc.errors())
    return await request_validation_exception_handler(request, exc)


@app.middleware("http")
async def add_elapsed_time_header(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Simple middleware to add information about the time elapsed for the response.

    Args:
        request: The incoming HTTP request.
        call_next: Callback that forwards the request to the next middleware or route handler.
    """
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        process_time = time.perf_counter() - start_time
        logger.exception(
            "method=%s path=%s elapsed_ms=%.1f - unhandled exception",
            request.method,
            request.url.path,
            process_time * 1000,
        )
        raise
    process_time = time.perf_counter() - start_time
    response.headers["X-Elapsed-Time"] = str(round(process_time, 4))
    log = logger.warning if response.status_code >= 400 else logger.info
    if request.url.path == "/ping":
        log = logger.debug
    log(
        "method=%s path=%s status=%d elapsed_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        process_time * 1000,
    )
    return response


@app.get("/ping")
async def ping() -> dict[str, str]:
    """Health check endpoint."""
    return {"message": "pong"}


if __name__ == "__main__":
    uvicorn.run(app=app)
