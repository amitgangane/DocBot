# Backend Changes for Frontend Integration

This document details all backend modifications made to support the Next.js frontend.

## Summary of Changes

| File | Change Type | Description |
|------|-------------|-------------|
| `app/main.py` | Modified | Added CORS middleware |
| `app/api/routes.py` | Modified | Added streaming endpoint + thread history endpoint |
| `app/services/rag_service.py` | Modified | Added streaming query function + cache fix |
| `app/services/generation.py` | Modified | Added async streaming generation |
| `app/services/__init__.py` | Modified | Exported new functions |

---

## 1. CORS Support (`app/main.py`)

Added CORS middleware to allow frontend (Vercel) to call backend (Render).

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Why:** Without CORS, browser blocks cross-origin requests from frontend to backend.

---

## 2. Streaming Endpoint (`app/api/routes.py`)

Added new endpoint for Server-Sent Events (SSE) streaming.

```python
from fastapi.responses import StreamingResponse

@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    """
    Query the RAG system with streaming response.
    Returns Server-Sent Events (SSE) stream.
    """
    return StreamingResponse(
        query_stream(request.question, thread_id=request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

**SSE Event Format:**
```
event: metadata
data: {"sources": 5, "rewritten_query": "..."}

event: token
data: {"token": "The"}

event: done
data: {"status": "complete"}
```

---

## 3. Thread History Endpoint (`app/api/routes.py`)

Added endpoint to retrieve conversation history for a thread.

```python
@router.get("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str):
    """Get chat history for a thread."""
    graph = get_rag_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)

    if not state.values:
        return {"messages": []}

    chat_history = state.values.get("chat_history", [])

    messages = []
    for msg in chat_history:
        messages.append({
            "role": "user" if msg.type == "human" else "assistant",
            "content": msg.content
        })

    return {"messages": messages}
```

**Why:** Frontend needs to load previous messages when user clicks on a thread in the sidebar.

---

## 4. Streaming Generation (`app/services/generation.py`)

Added async generator for streaming LLM tokens.

```python
from typing import AsyncGenerator

def get_llm(streaming: bool = False) -> ChatOpenAI:
    """Get the LLM instance."""
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        max_retries=2,
        streaming=streaming,  # Enable streaming
    )

async def generate_answer_stream(context: str, question: str) -> AsyncGenerator[str, None]:
    """Generate an answer using the LLM with streaming."""
    llm = get_llm(streaming=True)
    chain = RAG_PROMPT | llm

    async for chunk in chain.astream({"context": context, "question": question}):
        if chunk.content:
            yield chunk.content
```

**Why:** Allows tokens to be sent to frontend as they're generated, improving perceived performance.

---

## 5. Streaming Query Service (`app/services/rag_service.py`)

Added `query_stream()` function that:
1. Runs pipeline steps (rewrite, retrieve, rerank, context)
2. Streams generation tokens
3. Updates chat history after completion

```python
async def query_stream(
    question: str,
    thread_id: str = "default"
) -> AsyncGenerator[str, None]:
    """Stream RAG response with SSE format."""

    # Run pipeline steps synchronously
    result = rewrite_query_node(current_state)
    result = retrieve_node(current_state)
    result = rerank_node(current_state)
    result = build_context_node(current_state)

    # Send metadata
    yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"

    # Stream generation
    async for token in generate_answer_stream(context, rewritten_query):
        yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

    # Update memory
    graph.update_state(config, {"chat_history": new_history})

    # Send done
    yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"
```

---

## 6. Cache Fix for Chat History (`app/services/rag_service.py`)

**Problem:** When response was served from cache, chat history wasn't updated.

**Solution:** Update chat history even on cache hits.

```python
if cached is not None:
    logger.info(f"Response cache HIT")
    # Still need to save Q&A to graph state for history
    from langchain_core.messages import HumanMessage, AIMessage
    current_history = state.values.get("chat_history", []) if state.values else []
    new_history = list(current_history)
    new_history.append(HumanMessage(content=question))
    new_history.append(AIMessage(content=cached["answer"]))
    graph.update_state(config, {"chat_history": new_history})
    logger.info(f"Updated chat history with cached response")
    return cached
```

**Why:** Without this fix, cached responses wouldn't appear in conversation history when loading a thread.

---

## 7. Updated Exports (`app/services/__init__.py`)

```python
from .generation import get_llm, generate_answer, generate_answer_stream
from .rag_service import query, query_simple, query_stream
```

---

## API Changes Summary

### New Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query/stream` | Streaming query with SSE |
| GET | `/threads/{thread_id}/history` | Get conversation history |

### Modified Behavior

| Endpoint | Change |
|----------|--------|
| POST `/query` | Now updates chat history even on cache hits |

---

## Environment Variables

No new environment variables required. Existing configuration works with frontend.

---

## Testing the Changes

### Test Streaming Endpoint
```bash
curl -X POST "http://localhost:8001/query/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "What is attention?", "thread_id": "test-1"}'
```

### Test Thread History
```bash
curl "http://localhost:8001/threads/test-1/history"
```

### Test CORS
```bash
curl -X OPTIONS "http://localhost:8001/query" \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -v
```

---

## Migration Notes

1. **No database changes** - Uses existing Supabase tables via LangGraph
2. **No breaking changes** - Original `/query` endpoint still works
3. **Backwards compatible** - Existing API clients unaffected
