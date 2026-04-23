"""LangGraph nodes for the RAG pipeline."""

import tiktoken
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logger import setup_logger
from app.services.state import AgentState
from app.services.retrieval import search_similar
from app.services.reranker import rerank_documents
from app.services.generation import get_llm, RAG_PROMPT

logger = setup_logger("nodes")

# LLM for query rewriting (use smaller/faster model)
_rewrite_llm = None
# Token encoder for fast local token counting
_token_encoder = None


def get_rewrite_llm() -> ChatOpenAI:
    """Get LLM for query rewriting (singleton). Uses faster model."""
    global _rewrite_llm
    if _rewrite_llm is None:
        # Use gpt-4o-mini for rewriting - faster and cheaper
        _rewrite_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _rewrite_llm


def get_token_encoder():
    """Get tiktoken encoder for fast token counting."""
    global _token_encoder
    if _token_encoder is None:
        _token_encoder = tiktoken.encoding_for_model("gpt-4o")
    return _token_encoder


def count_tokens(messages) -> int:
    """Count tokens in messages using tiktoken (fast, local)."""
    encoder = get_token_encoder()
    total = 0
    for msg in messages:
        # Approximate token count: content + role overhead
        total += len(encoder.encode(msg.content)) + 4  # 4 tokens overhead per message
    return total


def _build_source_items(docs: list[Document]) -> list[dict]:
    """Create stable source metadata for persistence in chat history."""
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

def rewrite_query_node(state: AgentState) -> dict:
    """Rewrite query using a token-aware trimmer."""
    history = state.get("chat_history", [])

    if not history:
        logger.debug(f"No history, using original query")
        return {"rewritten_query": state["query"]}

    logger.debug(f"Rewriting query with {len(history)} history messages")

    # 1. Define the trimmer with fast local token counting
    trimmer = trim_messages(
        max_tokens=settings.REWRITE_MAX_TOKENS,
        strategy="last",           # Keep the most recent messages
        token_counter=count_tokens, # Fast local token counting with tiktoken
        start_on="human",          # Ensure the snippet doesn't start with a random AI response
        include_system=True,       # Keep your instructions if they are in history
    )

    # 2. Apply the trimmer to your chat history
    trimmed_history = trimmer.invoke(history)

    # 3. Format the trimmed messages for your prompt
    # Now you are certain history_text won't blow up your context window
    history_text = "\n".join([f"{m.type}: {m.content}" for m in trimmed_history])

    llm = get_rewrite_llm()
    prompt = f"""Given the chat history and a follow-up question, rewrite the question to be standalone.

Chat History:
{history_text}

Follow-up Question: {state["query"]}

Standalone Question:"""

    response = llm.invoke(prompt)
    logger.info(f"Query rewritten: \"{response.content[:50]}...\"")
    return {"rewritten_query": response.content}


def retrieve_node(state: AgentState) -> dict:
    """Retrieve documents using embedding similarity search."""
    query = state.get("rewritten_query") or state["query"]
    docs = search_similar(query, k=settings.RERANKER_INITIAL_K)
    logger.info(f"Retrieved {len(docs)} documents")
    return {"retrieved_docs": docs}


def rerank_node(state: AgentState) -> dict:
    """Rerank retrieved documents using cross-encoder."""
    query = state.get("rewritten_query") or state["query"]
    docs = rerank_documents(
        query,
        state["retrieved_docs"],
        top_k=settings.RETRIEVER_K
    )
    logger.info(f"Reranked to top {len(docs)} documents")
    return {"reranked_docs": docs}


def build_context_node(state: AgentState) -> dict:
    """Build context string from reranked documents."""
    context = "\n\n".join([doc.page_content for doc in state["reranked_docs"]])
    logger.debug(f"Built context: {len(context)} chars")
    return {
        "context": context,
        "source_items": _build_source_items(state["reranked_docs"]),
    }


def generate_node(state: AgentState) -> dict:
    """Generate answer using LLM (streaming-compatible)."""
    query = state.get("rewritten_query") or state["query"]

    # Use streaming=True so astream_events can capture tokens
    llm = get_llm(streaming=True)
    chain = RAG_PROMPT | llm

    response = chain.invoke({"context": state["context"], "question": query})
    logger.info(f"Generated answer: {len(response.content)} chars")
    return {"answer": response.content}


def update_memory_node(state: AgentState) -> dict:
    """Update chat history with current Q&A pair."""
    history = list(state.get("chat_history", []))
    history.append(HumanMessage(content=state["query"]))
    history.append(
        AIMessage(
            content=state["answer"],
            additional_kwargs={"source_items": state.get("source_items", [])},
        )
    )
    logger.debug(f"Updated chat history: {len(history)} messages")
    return {"chat_history": history}
