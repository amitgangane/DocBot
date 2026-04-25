import os
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger("vector_db")

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
        logger.info(f"Connecting to Qdrant Cloud...")
        # 2. Connect to Qdrant Cloud
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )

        # 3. Handle the 404 Error: Check if 'RAG-app' exists, create if it doesn't
        if not client.collection_exists(settings.QDRANT_COLLECTION_NAME):
            logger.info(f"Creating collection: {settings.QDRANT_COLLECTION_NAME}")
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIMENSIONS,
                    distance=models.Distance.COSINE,
                ),
            )

        # 4. Initialize the LangChain VectorStore
        _vectorstore = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding=_embeddings,
        )
        logger.info(f"Connected to Qdrant collection: {settings.QDRANT_COLLECTION_NAME}")

    return _vectorstore

def get_embeddings() -> OpenAIEmbeddings:
    return _embeddings


def get_qdrant_client() -> QdrantClient:
    """Return the underlying Qdrant client."""
    return get_vectorstore().client


def get_chunk_count() -> int:
    """Get total number of points in the cloud collection."""
    vs = get_vectorstore()
    response = vs.client.count(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        exact=True
    )
    logger.debug(f"Collection chunk count: {response.count}")
    return response.count


def list_indexed_documents() -> list[dict]:
    """Return one aggregated record per indexed document."""
    client = get_qdrant_client()
    documents: dict[str, dict] = {}
    page_numbers: dict[str, set] = defaultdict(set)
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            scroll_filter=None,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            document_id = metadata.get("document_id") or payload.get("document_id")
            if not document_id:
                continue

            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "filename": metadata.get("filename") or payload.get("filename") or "Untitled document",
                    "source_path": metadata.get("source_path") or payload.get("source_path") or "",
                    "chunk_count": 0,
                    "page_count": 0,
                }

            documents[document_id]["chunk_count"] += 1

            page_number = metadata.get("page_number") or payload.get("page_number")
            if page_number is not None:
                page_numbers[document_id].add(page_number)

        if next_offset is None:
            break

    for document_id, document in documents.items():
        document["page_count"] = len(page_numbers.get(document_id, set())) or None

    return sorted(
        documents.values(),
        key=lambda item: (item["filename"].lower(), item["document_id"]),
    )


def delete_indexed_document(document_id: str) -> int:
    """Delete all chunks belonging to a document and return deleted count."""
    client = get_qdrant_client()
    next_offset = None
    point_ids: list[models.ExtendedPointId] = []

    while True:
        points, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            scroll_filter=None,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            point_document_id = metadata.get("document_id") or payload.get("document_id")
            if point_document_id == document_id:
                point_ids.append(point.id)

        if next_offset is None:
            break

    if not point_ids:
        return 0

    client.delete(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points_selector=models.PointIdsList(points=point_ids),
        wait=True,
    )
    return len(point_ids)
