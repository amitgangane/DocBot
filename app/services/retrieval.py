import time
from langchain_core.vectorstores import VectorStoreRetriever

from app.db.vector_db import get_vectorstore
from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger("retrieval")


def get_retriever() -> VectorStoreRetriever:
    """Get a retriever from the vector store."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_kwargs={"k": settings.RETRIEVER_K}
    )


def search_similar(query: str, k: int = None) -> list:
    """Search for similar documents."""
    start_time = time.time()
    vectorstore = get_vectorstore()
    k = k or settings.RETRIEVER_K
    docs = vectorstore.similarity_search(query, k=k)
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Similarity search: {len(docs)} docs in {elapsed:.0f}ms")
    return docs
