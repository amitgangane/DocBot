import time
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from app.core.logger import setup_logger

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


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 5
) -> list[Document]:
    """
    Rerank documents using cross-encoder for better relevance.

    Args:
        query: The search query
        documents: List of LangChain Document objects from retrieval
        top_k: Number of top documents to return after reranking

    Returns:
        List of top_k most relevant documents
    """
    if not documents:
        return []

    start_time = time.time()
    reranker = get_reranker()

    # Create query-document pairs for cross-encoder
    pairs = [(query, doc.page_content) for doc in documents]

    # Get relevance scores
    scores = reranker.predict(pairs)

    # Sort by score (descending) and return top_k documents
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    elapsed = (time.time() - start_time) * 1000
    logger.debug(f"Reranking scores: {[f'{s:.3f}' for _, s in ranked[:top_k]]}")
    logger.info(f"Reranked {len(documents)} -> {top_k} docs in {elapsed:.0f}ms")

    return [doc for doc, _ in ranked[:top_k]]
