"""LangGraph RAG pipeline."""

import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app.services.state import AgentState
from app.services.nodes import (
    rewrite_query_node,
    retrieve_node,
    rerank_node,
    build_context_node,
    generate_node,
    update_memory_node,
)

# Persistent checkpointer (singleton)
_checkpointer = None
_graph = None


def get_checkpointer() -> SqliteSaver:
    """Get or create SQLite checkpointer for persistent memory."""
    global _checkpointer
    if _checkpointer is None:
        conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
    return _checkpointer


def build_rag_graph() -> StateGraph:
    """Build the RAG pipeline graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("rewrite", rewrite_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("generate", generate_node)
    graph.add_node("memory", update_memory_node)

    # Define edges (linear flow)
    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "build_context")
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate", "memory")
    graph.add_edge("memory", END)

    return graph


def get_rag_graph():
    """Get compiled RAG graph (singleton)."""
    global _graph
    if _graph is None:
        graph = build_rag_graph()
        checkpointer = get_checkpointer()
        _graph = graph.compile(checkpointer=checkpointer)
    return _graph
