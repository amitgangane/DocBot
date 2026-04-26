"""LangGraph RAG pipeline."""

import asyncio
import threading
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import settings
from app.core.logger import setup_logger
from app.services.state import AgentState

logger = setup_logger("graph")
from app.services.nodes import (
    rewrite_query_node,
    retrieve_node,
    rerank_node,
    build_context_node,
    generate_node,
    update_memory_node,
)

# Persistent checkpointer (singleton)
_checkpointer: Any = None
_graph = None
_pool = None
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None


def _run_background_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def _create_async_checkpointer():
    pool = AsyncConnectionPool(
        conninfo=settings.SUPABASE_DB_URL,
        min_size=settings.CHECKPOINTER_POOL_MIN_SIZE,
        max_size=settings.CHECKPOINTER_POOL_MAX_SIZE,
        kwargs={"autocommit": True, "prepare_threshold": None},
    )
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver, pool


def _reset_async_runtime():
    global _pool, _loop, _loop_thread
    _pool = None
    _loop = None
    _loop_thread = None


def _init_checkpointer_sync():
    """Initialize a checkpointer that works in both FastAPI and langgraph dev."""
    global _checkpointer, _pool, _loop, _loop_thread

    if _checkpointer is not None:
        return

    if settings.SUPABASE_DB_URL:
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=_run_background_loop,
            args=(loop,),
            daemon=True,
            name="langgraph-checkpointer-loop",
        )
        thread.start()

        try:
            future = asyncio.run_coroutine_threadsafe(_create_async_checkpointer(), loop)
            saver, pool = future.result()
        except Exception:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=1)
            _reset_async_runtime()
            raise

        _loop = loop
        _loop_thread = thread
        _pool = pool
        _checkpointer = saver
        logger.info("Async Postgres checkpointer initialized")
    else:
        _checkpointer = MemorySaver()
        logger.info("Using in-memory checkpointer")

async def init_checkpointer():
    """Initialize checkpointer once at app startup."""
    _init_checkpointer_sync()


async def close_checkpointer():
    """Cleanly close pooled checkpointer resources."""
    global _checkpointer, _graph, _pool, _loop, _loop_thread

    if _pool is not None and _loop is not None:
        future = asyncio.run_coroutine_threadsafe(_pool.close(), _loop)
        future.result()
        _loop.call_soon_threadsafe(_loop.stop)
        if _loop_thread is not None:
            _loop_thread.join(timeout=1)
        logger.info("Checkpointer connection pool closed")

    _checkpointer = None
    _graph = None
    _reset_async_runtime()

def get_checkpointer():
    if _checkpointer is None:
        _init_checkpointer_sync()
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
    global _graph

    if _graph is None:
        logger.info("Building RAG graph...")
        graph = build_rag_graph()
        _graph = graph.compile(checkpointer=get_checkpointer())
        logger.info("RAG graph compiled successfully")

    return _graph
