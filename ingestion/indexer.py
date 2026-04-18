from ingestion.loader import load_pdf
from ingestion.chunking import chunk_documents
from ingestion.embedder import embed_documents


def index_pdf(file_path: str) -> dict:
    """
    Full indexing pipeline: load PDF, chunk, and embed.

    Returns dict with indexing stats.
    """
    # Load
    docs = load_pdf(file_path)

    # Chunk
    chunks = chunk_documents(docs)

    # Embed
    count = embed_documents(chunks)

    return {
        "pages": len(docs),
        "chunks": count,
        "file": file_path,
    }
