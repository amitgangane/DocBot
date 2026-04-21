"""RAG service using LangGraph pipeline."""

from typing import Optional
from app.services.graph import get_rag_graph
from app.core.config import settings
from app.core.logger import setup_logger
from app.core.cache import cache_get, cache_set, make_cache_key

logger = setup_logger("rag_service")


def query(
    question: str,
    thread_id: str = "default",
    chat_history: Optional[list] = None,
    use_cache: bool = True
) -> dict:
    """
    Query the RAG system using LangGraph pipeline.

    Args:
        question: User's question
        thread_id: Session ID for maintaining conversation context
        chat_history: Optional existing chat history
        use_cache: Whether to use response cache (disabled for follow-ups)

    Returns:
        dict with answer, sources count, and updated chat history
    """
    graph = get_rag_graph()

    # Build config with thread_id for memory persistence
    config = {"configurable": {"thread_id": thread_id}}

    # Check if this is a follow-up (has existing history)
    # Don't cache follow-ups as they depend on conversation context
    state = graph.get_state(config)
    has_history = bool(state.values.get("chat_history")) if state.values else False

    # Only cache first queries (no history context)
    if use_cache and not has_history and not chat_history:
        cache_key = make_cache_key("response", question)
        cached = cache_get(cache_key)
        if cached is not None:
            logger.info(f"Response cache HIT")
            return cached

    # Build initial state
    initial_state = {"query": question}
    if chat_history:
        initial_state["chat_history"] = chat_history

    # Run the graph
    result = graph.invoke(initial_state, config=config)

    response = {
        "answer": result["answer"],
        "sources": len(result.get("reranked_docs", [])),
        "chat_history": result.get("chat_history", []),
    }

    # Cache the response (only for first queries without history)
    if use_cache and not has_history and not chat_history:
        cache_set(cache_key, response, settings.CACHE_TTL_RESPONSE)
        logger.info(f"Response cached")

    return response


def query_simple(question: str) -> dict:
    """
    Simple query without conversation memory.
    For backwards compatibility with existing API.
    """
    result = query(question, thread_id="simple-query")
    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }
