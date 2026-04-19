from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class AgentState(TypedDict):
    """State for the RAG agent graph."""
    query: str
    rewritten_query: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    context: str
    answer: str
    chat_history: List[BaseMessage]
