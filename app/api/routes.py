import os
import shutil
import time

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger("routes")
from app.models import (
    HealthResponse,
    IngestResponse,
    EmbedRequest,
    EmbedResponse,
    CountResponse,
    QueryRequest,
    QueryResponse,
    CacheStatsResponse,
    CacheClearResponse,
)
from app.db.vector_db import get_vectorstore, get_chunk_count
from app.services.rag_service import query as rag_query
from app.core.cache import cache_stats, cache_clear_all, invalidate_on_ingest
from ingestion.loader import load_pdf
from ingestion.chunking import chunk_documents

router = APIRouter()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", service="rag-app")


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, parse it, chunk it, and store embeddings.
    """
    start_time = time.time()
    logger.info(f"Ingesting PDF: {file.filename}")

    if not file.filename.endswith(".pdf"):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Parse PDF
    try:
        docs = load_pdf(file_path)
        logger.debug(f"Parsed {len(docs)} pages from PDF")
    except Exception as e:
        logger.error(f"Failed to parse PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")

    # Chunk documents
    try:
        chunks = chunk_documents(docs)
        logger.debug(f"Created {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Failed to chunk document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to chunk document: {str(e)}")

    # Add to vector store
    try:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks to vector store")
    except Exception as e:
        logger.error(f"Failed to embed chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to embed chunks: {str(e)}")

    # Invalidate document-dependent caches
    invalidate_on_ingest()

    elapsed = time.time() - start_time
    logger.info(f"Ingestion complete: {file.filename} ({len(chunks)} chunks) in {elapsed:.2f}s")

    return IngestResponse(
        filename=file.filename,
        status="success",
        pages_parsed=len(docs),
        chunk_count=len(chunks),
        message=f"Successfully ingested {file.filename}",
    )


@router.post("/embed", response_model=EmbedResponse)
async def embed_chunks(request: EmbedRequest):
    """
    Receive chunks and store them in the vector store.
    """
    try:
        vectorstore = get_vectorstore()
        texts = [chunk.page_content for chunk in request.chunks]
        metadatas = [chunk.metadata for chunk in request.chunks]
        vectorstore.add_texts(texts=texts, metadatas=metadatas)

        total = get_chunk_count()
        return EmbedResponse(
            status="success",
            chunks_added=len(request.chunks),
            total_chunks=total,
            message=f"Added {len(request.chunks)} chunks",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed: {str(e)}")


@router.get("/embed/count", response_model=CountResponse)
async def get_count():
    """Get total chunk count in vector store."""
    try:
        count = get_chunk_count()
        return CountResponse(total_chunks=count, status="ok")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get count: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question.
    Uses LangGraph pipeline with conversation memory.
    """
    start_time = time.time()
    logger.info(f"Query received: \"{request.question[:50]}...\" thread_id={request.thread_id}")

    try:
        result = rag_query(request.question, thread_id=request.thread_id)
        elapsed = time.time() - start_time
        logger.info(f"Query complete: {result['sources']} sources in {elapsed:.2f}s")
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get cache statistics."""
    stats = cache_stats()
    return CacheStatsResponse(**stats)


@router.post("/cache/clear", response_model=CacheClearResponse)
async def clear_cache():
    """Clear all caches."""
    cleared = cache_clear_all()
    logger.info(f"Cache cleared: {cleared} keys")
    return CacheClearResponse(status="success", keys_cleared=cleared)
