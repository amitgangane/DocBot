import time
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import setup_logger
from app.core.cache import cache_get, cache_set, make_cache_key

logger = setup_logger("reranker")

# Initialize cross-encoder model (loads once)
_reranker = None


def get_reranker() -> CrossEncoder:
    """Get or create the cross-encoder reranker singleton."""
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        logger.info("Cross-encoder model loaded")
    return _reranker


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


def _get_doc_fingerprint(documents: list[Document]) -> str:
    """Create a fingerprint of documents for cache key."""
    # Use first 100 chars of each doc's content for fingerprint
    return "|".join([doc.page_content[:100] for doc in documents])


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 5
) -> list[Document]:
    """
    Rerank documents using cross-encoder for better relevance with caching.

    Args:
        query: The search query
        documents: List of LangChain Document objects from retrieval
        top_k: Number of top documents to return after reranking

    Returns:
        List of top_k most relevant documents
    """
    if not documents:
        return []

    # Create cache key from query and document fingerprint
    doc_fingerprint = _get_doc_fingerprint(documents)
    cache_key = make_cache_key("rerank", query, doc_fingerprint, top_k)

    # Try cache first
    cached = cache_get(cache_key)
    if cached is not None:
        logger.info(f"Reranker cache HIT: {len(cached)} docs")
        return _cache_to_docs(cached)

    # Cache miss - do actual reranking
    start_time = time.time()
    reranker = get_reranker()

    # Create query-document pairs for cross-encoder
    pairs = [(query, doc.page_content) for doc in documents]

    # Get relevance scores
    scores = reranker.predict(pairs)

    # Sort by score (descending) and return top_k documents
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    result_docs = [doc for doc, _ in ranked[:top_k]]

    # Cache the results
    cache_set(cache_key, _docs_to_cache(result_docs), settings.CACHE_TTL_RERANKER)

    elapsed = (time.time() - start_time) * 1000
    logger.debug(f"Reranking scores: {[f'{s:.3f}' for _, s in ranked[:top_k]]}")
    logger.info(f"Reranked {len(documents)} -> {top_k} docs in {elapsed:.0f}ms (cached)")

    return result_docs
