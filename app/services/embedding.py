from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


def get_embedding_model() -> OpenAIEmbeddings:
    """Get the OpenAI embedding model."""
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dimensions=settings.EMBEDDING_DIMENSIONS,
    )
