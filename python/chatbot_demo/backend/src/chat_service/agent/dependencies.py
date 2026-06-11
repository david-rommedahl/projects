from typing import Annotated

from fastapi import Depends, Request
from langgraph.checkpoint.base import BaseCheckpointSaver


async def get_checkpointer(request: Request) -> BaseCheckpointSaver:
    """FastAPI dependency returning the application-wide LangGraph checkpointer.

    The concrete checkpointer (``AsyncPostgresSaver`` in production) is opened
    once in the FastAPI lifespan (see ``chat_service.asgi.open_checkpointer``) and
    attached to ``app.state.checkpointer``, so all requests share the same
    connection pool for the lifetime of the process. The return type is the
    abstract base so tests can override this with a different implementation
    (e.g. ``InMemorySaver``) without coupling endpoints to Postgres.
    """
    return request.app.state.checkpointer


CheckpointerDep = Annotated[BaseCheckpointSaver, Depends(get_checkpointer)]
