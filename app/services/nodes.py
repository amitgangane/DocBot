"""LangGraph nodes for the RAG pipeline."""

from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.state import AgentState
from app.services.retrieval import search_similar
from app.services.reranker import rerank_documents
from app.services.generation import generate_answer

# LLM for query rewriting
_rewrite_llm = None


def get_rewrite_llm() -> ChatOpenAI:
    """Get LLM for query rewriting (singleton)."""
    global _rewrite_llm
    if _rewrite_llm is None:
        _rewrite_llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
    return _rewrite_llm

def rewrite_query_node(state: AgentState) -> dict:
    """Rewrite query using a token-aware trimmer."""
    history = state.get("chat_history", [])

    if not history:
        return {"rewritten_query": state["query"]}

    # 1. Define the trimmer (can be moved outside the function for performance)
    trimmer = trim_messages(
        max_tokens=settings.REWRITE_MAX_TOKENS,
        strategy="last",           # Keep the most recent messages
        token_counter=get_rewrite_llm(), # Use the LLM to count tokens accurately
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
    return {"rewritten_query": response.content}

def retrieve_node(state: AgentState) -> dict:
    """Retrieve documents using embedding similarity search."""
    query = state.get("rewritten_query") or state["query"]
    docs = search_similar(query, k=settings.RERANKER_INITIAL_K)
    return {"retrieved_docs": docs}


def rerank_node(state: AgentState) -> dict:
    """Rerank retrieved documents using cross-encoder."""
    query = state.get("rewritten_query") or state["query"]
    docs = rerank_documents(
        query,
        state["retrieved_docs"],
        top_k=settings.RETRIEVER_K
    )
    return {"reranked_docs": docs}


def build_context_node(state: AgentState) -> dict:
    """Build context string from reranked documents."""
    context = "\n\n".join([doc.page_content for doc in state["reranked_docs"]])
    return {"context": context}


def generate_node(state: AgentState) -> dict:
    """Generate answer using LLM."""
    query = state.get("rewritten_query") or state["query"]
    answer = generate_answer(state["context"], query)
    return {"answer": answer}


def update_memory_node(state: AgentState) -> dict:
    """Update chat history with current Q&A pair."""
    history = list(state.get("chat_history", []))
    history.append(HumanMessage(content=state["query"]))
    history.append(AIMessage(content=state["answer"]))
    return {"chat_history": history}
