"""API router aggregator.

The top-level ``router`` is mounted under ``/api/v1`` in
:mod:`chat_service.asgi`. Versioned domain routers (conversations, messages,
...) get included here in a later step; for now it carries no routes.
"""

from fastapi import APIRouter

from chat_service.api.chat import router as chat_router

router = APIRouter()
router.include_router(chat_router)
