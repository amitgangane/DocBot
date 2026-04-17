import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from embeddings.embeddings import embedding_model
from pdf_ingestion.chunking import chunk_documents


def get_retriever():
    """Automatically loads PDF, chunks it, creates vector store, and returns retriever."""
    final_chunks = chunk_documents()

    vectorstore = Chroma.from_documents(
        documents=final_chunks,
        embedding=embedding_model,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
    )

    return retriever


# if __name__ == "__main__":
#     retriever = get_retriever()
#     answer = retriever.invoke("What is multihead attention?")
#     print(answer[0].page_content)
