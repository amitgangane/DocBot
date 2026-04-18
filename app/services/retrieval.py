from langchain_core.vectorstores import VectorStoreRetriever

from app.db.vector_db import get_vectorstore
from app.core.config import settings


def get_retriever() -> VectorStoreRetriever:
    """Get a retriever from the vector store."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_kwargs={"k": settings.RETRIEVER_K}
    )


def search_similar(query: str, k: int = None) -> list:
    """Search for similar documents."""
    vectorstore = get_vectorstore()
    k = k or settings.RETRIEVER_K
    return vectorstore.similarity_search(query, k=k)
