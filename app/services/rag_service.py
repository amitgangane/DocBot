"""RAG service using LangGraph pipeline."""

import json
from typing import Optional, AsyncGenerator
from langchain_core.documents import Document
from app.services.graph import get_rag_graph
from app.core.config import settings
from app.core.logger import setup_logger
from app.core.cache import cache_get, cache_set, make_cache_key
from langsmith import traceable

logger = setup_logger("rag_service")

STREAM_STAGE_LABELS = {
    "rewrite": "Rewriting your question",
    "retrieve": "Searching the document index",
    "rerank": "Ranking the most relevant chunks",
    "build_context": "Preparing the final context",
    "generate": "Generating the answer",
    "memory": "Saving this conversation",
}


def _build_source_items(docs: list[Document]) -> list[dict]:
    """Create frontend-friendly source cards from reranked documents."""
    source_items: list[dict] = []
    seen_keys: set[tuple[str, int | None, str | None]] = set()

    for doc in docs:
        metadata = doc.metadata or {}
        item = {
            "document_id": metadata.get("document_id") or metadata.get("source_path") or "unknown-document",
            "filename": metadata.get("filename") or metadata.get("source") or "Unknown source",
            "source_path": metadata.get("source_path") or metadata.get("source") or "",
            "page_number": metadata.get("page_number") or metadata.get("page"),
            "chunk_id": metadata.get("chunk_id"),
            "excerpt": doc.page_content[:260].strip(),
        }
        dedupe_key = (item["document_id"], item["page_number"], item["chunk_id"])
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        source_items.append(item)

    return source_items


@traceable(name="RAG Query")
async def query(
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
    state = await graph.aget_state(config)
    has_history = bool(state.values.get("chat_history")) if state.values else False

    # Only cache first queries (no history context)
    if use_cache and not has_history and not chat_history:
        cache_key = make_cache_key("response", question)
        cached = cache_get(cache_key)
        if cached is not None:
            logger.info(f"Response cache HIT")
            # Still need to save Q&A to graph state for history
            from langchain_core.messages import HumanMessage, AIMessage
            current_history = state.values.get("chat_history", []) if state.values else []
            new_history = list(current_history)
            new_history.append(HumanMessage(content=question))
            new_history.append(
                AIMessage(
                    content=cached["answer"],
                    additional_kwargs={"source_items": cached.get("source_items", [])},
                )
            )
            await graph.aupdate_state(config, {"chat_history": new_history})
            logger.info(f"Updated chat history with cached response")
            return {
                "answer": cached["answer"],
                "sources": cached["sources"],
                "source_items": cached.get("source_items", []),
                "chat_history": new_history,
            }

    # Build initial state
    initial_state = {"query": question}
    if chat_history:
        initial_state["chat_history"] = chat_history

    # Run the graph
    result = await graph.ainvoke(initial_state, config=config)

    response = {
        "answer": result["answer"],
        "sources": len(result.get("reranked_docs", [])),
        "source_items": _build_source_items(result.get("reranked_docs", [])),
        "chat_history": result.get("chat_history", []),
    }

    # Cache the response (only for first queries without history)
    if use_cache and not has_history and not chat_history:
        cache_set(
            cache_key,
            {
                "answer": response["answer"],
                "sources": response["sources"],
                "source_items": response["source_items"],
            },
            settings.CACHE_TTL_RESPONSE,
        )
        logger.info(f"Response cached")

    return response


async def query_simple(question: str) -> dict:
    """
    Simple query without conversation memory.
    For backwards compatibility with existing API.
    """
    result = await query(question, thread_id="simple-query")
    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }


async def query_stream(
    question: str,
    thread_id: str = "default"
) -> AsyncGenerator[str, None]:
    """
    Stream RAG response using LangGraph's astream_events.

    This properly traces the full graph in LangSmith while streaming LLM tokens.

    Yields SSE events:
    - event: metadata (sources count)
    - event: token (streamed answer tokens)
    - event: done (completion signal)
    """
    graph = get_rag_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Build initial state
    initial_state = {"query": question}

    full_answer = ""
    sources_count = 0
    source_items: list[dict] = []
    metadata_sent = False
    emitted_statuses: set[str] = set()

    try:
        # Use astream_events to get full graph tracing + LLM streaming
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            kind = event.get("event")
            node_name = event.get("metadata", {}).get("langgraph_node")

            if kind == "on_chain_start" and node_name in STREAM_STAGE_LABELS and node_name not in emitted_statuses:
                emitted_statuses.add(node_name)
                yield (
                    "event: status\n"
                    f"data: {json.dumps({'stage': node_name, 'message': STREAM_STAGE_LABELS[node_name]})}\n\n"
                )

            # When generate node completes, we can get sources from state
            if kind == "on_chain_end" and event.get("name") == "rerank":
                output = event.get("data", {}).get("output", {})
                if "reranked_docs" in output:
                    sources_count = len(output["reranked_docs"])
                    source_items = _build_source_items(output["reranked_docs"])
                    if not metadata_sent:
                        metadata = {"sources": sources_count, "source_items": source_items}
                        yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"
                        metadata_sent = True

            # Stream LLM tokens - ONLY from the generate node (not rewrite node)
            if kind == "on_chat_model_stream":
                if node_name != "generate":
                    continue

                if not metadata_sent:
                    metadata = {"sources": sources_count, "source_items": source_items}
                    yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"
                    metadata_sent = True

                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = chunk.content
                    full_answer += token
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

        if not metadata_sent:
            metadata = {"sources": sources_count, "source_items": source_items}
            yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"

        logger.info(f"Streamed response: {len(full_answer)} chars, {sources_count} sources")
        yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"
    except Exception as e:
        logger.exception("Stream query failed")
        message = str(e) or repr(e)
        yield f"event: error\ndata: {json.dumps({'message': message})}\n\n"
