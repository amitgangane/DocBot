from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# Initialize cross-encoder model (loads once)
_reranker = None


def get_reranker() -> CrossEncoder:
    """Get or create the cross-encoder reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
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

    reranker = get_reranker()

    # Create query-document pairs for cross-encoder
    pairs = [(query, doc.page_content) for doc in documents]

    # Get relevance scores
    scores = reranker.predict(pairs)

    # Sort by score (descending) and return top_k documents
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked[:top_k]]
