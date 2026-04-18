from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from app.core.config import settings


def chunk_documents(docs: list) -> list:
    """
    Split documents into chunks for embedding.
    """
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    chunks = char_splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")
    return chunks
