"""LLM client factory.

A small factory returning a configured chat model. It mirrors the reference
project's ``get_model()`` factory so it's easy to extend to a provider registry
later: today it always returns a :class:`ChatOpenAI`, but the ``model_id`` seam
and the single construction site are where a provider lookup would slot in.

The same :class:`ChatOpenAI` client serves both OpenAI directly and an internal
Azure AI Foundry deployment: Foundry exposes an OpenAI-compatible ``/openai/v1``
surface, so pointing ``OPENAI_BASE_URL`` at it (with the Azure key and the
deployment name as ``CHAT_MODEL``) is all that differs. No ``AzureChatOpenAI``
needed.
"""

import logging

from langchain_openai import ChatOpenAI

from chat_service.config import CONFIG

logger = logging.getLogger(__name__)


def get_chat_model(model_id: str | None = None) -> ChatOpenAI:
    """Return a configured chat model.

    Args:
        model_id: Optional model identifier override. Falls back to the
            server-configured ``CHAT_MODEL`` (``CONFIG.chat_model``) when not
            given, so callers can stay provider-agnostic.
    """
    model = model_id or CONFIG.chat_model
    logger.debug("Building chat model %s (base_url=%s)", model, CONFIG.openai_base_url or "openai")
    return ChatOpenAI(model=model, api_key=CONFIG.openai_api_key, base_url=CONFIG.openai_base_url)
