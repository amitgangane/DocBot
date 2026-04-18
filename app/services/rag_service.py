from app.services.retrieval import get_retriever
from app.services.generation import generate_answer


def query(question: str) -> dict:
    """
    Main RAG pipeline: retrieve context and generate answer.

    Returns dict with answer and number of sources used.
    """
    retriever = get_retriever()

    # Get relevant context
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate answer
    answer = generate_answer(context, question)

    return {
        "answer": answer,
        "sources": len(docs),
    }
