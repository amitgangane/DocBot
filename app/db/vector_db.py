import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings

os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

_embeddings = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL,
    api_key=settings.OPENAI_API_KEY,
)

_vectorstore = None


def get_vectorstore() -> Chroma:
    """Get or create the ChromaDB vectorstore singleton."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=_embeddings,
        )
    return _vectorstore


def get_embeddings() -> OpenAIEmbeddings:
    """Get the embeddings model."""
    return _embeddings


def get_chunk_count() -> int:
    """Get total number of chunks in the vector store."""
    vs = get_vectorstore()
    return vs._collection.count()
