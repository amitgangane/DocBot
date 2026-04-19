"""RAG service using LangGraph pipeline."""

from typing import Optional
from app.services.graph import get_rag_graph


def query(
    question: str,
    thread_id: str = "default",
    chat_history: Optional[list] = None
) -> dict:
    """
    Query the RAG system using LangGraph pipeline.

    Args:
        question: User's question
        thread_id: Session ID for maintaining conversation context
        chat_history: Optional existing chat history

    Returns:
        dict with answer, sources count, and updated chat history
    """
    graph = get_rag_graph()

    # Build config with thread_id for memory persistence
    config = {"configurable": {"thread_id": thread_id}}

    # Build initial state
    initial_state = {"query": question}
    if chat_history:
        initial_state["chat_history"] = chat_history

    # Run the graph
    result = graph.invoke(initial_state, config=config)

    return {
        "answer": result["answer"],
        "sources": len(result.get("reranked_docs", [])),
        "chat_history": result.get("chat_history", []),
    }


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
