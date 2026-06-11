"""A minimal LangChain agent.

Builds a basic agent with ``create_agent`` on top of the configured chat model
(:func:`chat_service.llm.get_chat_model`). It has no tools yet — that's the seam
where domain tools slot in later — so for now it's a single model node wrapped
in LangChain's agent graph. Not wired into the API yet.
"""

from langchain.agents import create_agent
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from chat_service.llm import get_chat_model

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def build_agent(
    *,
    model_id: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build a basic agent backed by the configured chat model.

    Args:
        model_id: Optional model identifier override passed to
            :func:`chat_service.llm.get_chat_model`. Falls back to the
            server-configured ``CHAT_MODEL`` when not given.
        system_prompt: System prompt steering the agent's behaviour.
        checkpointer: Optional LangGraph checkpointer. When provided, the agent
            persists conversation state per ``thread_id``, so invoking it with
            ``{"configurable": {"thread_id": session_id}}`` continues an existing
            session. When ``None`` the agent is stateless (each call starts fresh).

    Returns:
        A compiled agent graph. Invoke it with a message list, e.g.
        ``agent.invoke({"messages": [{"role": "user", "content": "Hi"}]})``.
    """
    model = get_chat_model(model_id)
    return create_agent(model, tools=[], system_prompt=system_prompt, checkpointer=checkpointer)
