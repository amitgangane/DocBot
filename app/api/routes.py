import os
import shutil
import time
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger("routes")
from app.models import (
    HealthResponse,
    IngestResponse,
    IndexedDocumentResponse,
    DeleteDocumentResponse,
    EmbedRequest,
    EmbedResponse,
    CountResponse,
    QueryRequest,
    QueryResponse,
    CacheStatsResponse,
    CacheClearResponse,
)
from app.db.vector_db import (
    get_vectorstore,
    get_chunk_count,
    list_indexed_documents,
    delete_indexed_document,
)
from app.services.rag_service import query as rag_query, query_stream
from app.core.cache import cache_stats, cache_clear_all, invalidate_on_ingest
from ingestion.loader import load_pdf
from ingestion.chunking import chunk_documents

router = APIRouter()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


def _annotate_loaded_docs(docs: list, *, file_path: str, filename: str, document_id: str) -> list:
    """Stamp stable source metadata onto each loaded page document."""
    for page_index, doc in enumerate(docs, start=1):
        metadata = dict(getattr(doc, "metadata", {}) or {})
        metadata["document_id"] = document_id
        metadata["filename"] = filename
        metadata["source_path"] = file_path
        metadata["page_number"] = metadata.get("page_number") or metadata.get("page") or page_index
        doc.metadata = metadata
    return docs


def _annotate_chunks(chunks: list, *, document_id: str) -> list:
    """Stamp stable chunk-level metadata for downstream source display."""
    for chunk_index, chunk in enumerate(chunks, start=1):
        metadata = dict(getattr(chunk, "metadata", {}) or {})
        metadata["chunk_id"] = f"{document_id}:chunk-{chunk_index}"
        metadata["chunk_index"] = chunk_index
        chunk.metadata = metadata
    return chunks


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
        document_id = str(uuid4())
        docs = _annotate_loaded_docs(
            load_pdf(file_path),
            file_path=file_path,
            filename=file.filename,
            document_id=document_id,
        )
        logger.debug(f"Parsed {len(docs)} pages from PDF")
    except Exception as e:
        logger.error(f"Failed to parse PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")

    # Chunk documents
    try:
        chunks = _annotate_chunks(chunk_documents(docs), document_id=document_id)
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
        document_id=document_id,
        message=f"Successfully ingested {file.filename}",
    )


@router.get("/documents", response_model=list[IndexedDocumentResponse])
async def get_documents():
    """List indexed documents available in the vector store."""
    try:
        return list_indexed_documents()
    except Exception as e:
        logger.error(f"Failed to list indexed documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list indexed documents: {str(e)}")


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(document_id: str):
    """Delete an indexed document from the vector store by document id."""
    try:
        chunks_deleted = delete_indexed_document(document_id)
        if chunks_deleted == 0:
            raise HTTPException(status_code=404, detail="Document not found")

        invalidate_on_ingest()
        logger.info(f"Deleted indexed document {document_id} ({chunks_deleted} chunks)")
        return DeleteDocumentResponse(
            status="success",
            document_id=document_id,
            chunks_deleted=chunks_deleted,
            message=f"Deleted document {document_id} from the index",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete indexed document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete indexed document: {str(e)}")


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
        result = await rag_query(request.question, thread_id=request.thread_id)
        elapsed = time.time() - start_time
        logger.info(f"Query complete: {result['sources']} sources in {elapsed:.2f}s")
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            source_items=result.get("source_items", []),
        )
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    """
    Query the RAG system with streaming response.
    Returns Server-Sent Events (SSE) stream.

    Events:
    - metadata: {sources: int, rewritten_query: str}
    - token: {token: str}
    - done: {status: "complete"}
    """
    logger.info(f"Stream query received: \"{request.question[:50]}...\" thread_id={request.thread_id}")

    return StreamingResponse(
        query_stream(request.question, thread_id=request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str):
    """Get chat history for a thread."""
    from app.services.graph import get_rag_graph

    try:
        graph = get_rag_graph()
        config = {"configurable": {"thread_id": thread_id}}
        state = await graph.aget_state(config)

        if not state.values:
            return {"messages": []}

        chat_history = state.values.get("chat_history", [])

        # Convert LangChain messages to simple format
        messages = []
        for msg in chat_history:
            messages.append({
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content,
                "source_items": getattr(msg, "additional_kwargs", {}).get("source_items", []),
            })

        return {"messages": messages}
    except Exception as e:
        logger.error(f"Failed to get thread history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get thread history: {str(e)}")


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
