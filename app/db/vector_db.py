import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings

# 1. Initialize Embeddings
_embeddings = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL,
    api_key=settings.OPENAI_API_KEY,
)

_vectorstore = None

def get_vectorstore() -> QdrantVectorStore:
    """Get or create the Qdrant Cloud vectorstore singleton."""
    global _vectorstore
    
    if _vectorstore is None:
        # 2. Connect to Qdrant Cloud
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        
        # 3. Handle the 404 Error: Check if 'RAG-app' exists, create if it doesn't
        if not client.collection_exists(settings.QDRANT_COLLECTION_NAME):
            print(f"Creating collection: {settings.QDRANT_COLLECTION_NAME}")
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIMENSIONS, # Ensure this is 1536 for OpenAI small
                    distance=models.Distance.COSINE,
                ),
            )
        
        # 4. Initialize the LangChain VectorStore
        _vectorstore = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding=_embeddings,
        )
        
    return _vectorstore

def get_embeddings() -> OpenAIEmbeddings:
    return _embeddings

def get_chunk_count() -> int:
    """Get total number of points in the cloud collection."""
    vs = get_vectorstore()
    response = vs.client.count(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        exact=True
    )
    return response.count