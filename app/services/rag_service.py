from app.services.retrieval import search_similar
from app.services.reranker import rerank_documents
from app.services.generation import generate_answer
from app.core.config import settings


def query(question: str, use_reranker: bool = True) -> dict:
    """
    Main RAG pipeline: retrieve, rerank, and generate answer.

    Args:
        question: User's question
        use_reranker: Whether to use cross-encoder reranking (default: True)

    Returns:
        dict with answer and number of sources used
    """
    # Step 1: Retrieve more documents than needed (for reranking)
    initial_k = settings.RERANKER_INITIAL_K if use_reranker else settings.RETRIEVER_K
    docs = search_similar(question, k=initial_k)

    # Step 2: Rerank to get most relevant documents
    if use_reranker and docs:
        docs = rerank_documents(question, docs, top_k=settings.RETRIEVER_K)

    # Step 3: Build context from top documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 4: Generate answer
    answer = generate_answer(context, question)

    return {
        "answer": answer,
        "sources": len(docs),
    }
