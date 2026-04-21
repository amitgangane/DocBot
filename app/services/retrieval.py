import time
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

from app.db.vector_db import get_vectorstore
from app.core.config import settings
from app.core.logger import setup_logger
from app.core.cache import cache_get, cache_set, make_cache_key

logger = setup_logger("retrieval")


def get_retriever() -> VectorStoreRetriever:
    """Get a retriever from the vector store."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_kwargs={"k": settings.RETRIEVER_K}
    )


def _docs_to_cache(docs: list[Document]) -> list[dict]:
    """Convert Documents to cacheable format."""
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]


def _cache_to_docs(cached: list[dict]) -> list[Document]:
    """Convert cached format back to Documents."""
    return [
        Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
        for d in cached
    ]


def search_similar(query: str, k: int = None) -> list:
    """Search for similar documents with caching."""
    k = k or settings.RETRIEVER_K
    cache_key = make_cache_key("retrieve", query, k)

    # Try cache first
    cached = cache_get(cache_key)
    if cached is not None:
        logger.info(f"Retrieval cache HIT: {len(cached)} docs")
        return _cache_to_docs(cached)

    # Cache miss - do actual search
    start_time = time.time()
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    elapsed = (time.time() - start_time) * 1000

    # Cache the results
    cache_set(cache_key, _docs_to_cache(docs), settings.CACHE_TTL_RETRIEVAL)

    logger.info(f"Similarity search: {len(docs)} docs in {elapsed:.0f}ms (cached)")
    return docs
