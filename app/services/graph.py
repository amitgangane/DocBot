"""LangGraph RAG pipeline."""

import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app.core.config import settings
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


def get_checkpointer():
    """Get or create checkpointer for persistent memory.

    Uses PostgreSQL (Supabase) if configured, otherwise falls back to SQLite.
    """
    global _checkpointer
    if _checkpointer is None:
        if settings.SUPABASE_DB_URL:
            # Use Supabase PostgreSQL
            from psycopg import Connection
            from psycopg.rows import dict_row
            from langgraph.checkpoint.postgres import PostgresSaver

            conn = Connection.connect(
                settings.SUPABASE_DB_URL,
                autocommit=True,
                row_factory=dict_row,
            )
            _checkpointer = PostgresSaver(conn)
            _checkpointer.setup()  # Create tables if they don't exist
            print("Using Supabase PostgreSQL for chat history")
        else:
            # Fallback to SQLite for local development
            conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
            _checkpointer = SqliteSaver(conn)
            print("Using SQLite for chat history (set SUPABASE_DB_URL for PostgreSQL)")
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
