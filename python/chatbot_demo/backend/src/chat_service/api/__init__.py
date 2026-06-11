"""API router aggregator.

The top-level ``router`` is mounted under ``/api/v1`` in
:mod:`chat_service.asgi`. Domain routers are included here.
"""

from fastapi import APIRouter

from chat_service.api.chat import router as chat_router
from chat_service.api.conversations import router as conversations_router
from chat_service.api.users import router as users_router

router = APIRouter()
router.include_router(chat_router)
router.include_router(conversations_router)
router.include_router(users_router)
