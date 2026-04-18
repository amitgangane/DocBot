from app.db.vector_db import get_vectorstore, get_embeddings


def embed_documents(docs: list) -> int:
    """
    Embed documents and add to vector store.
    Returns the number of chunks added.
    """
    vectorstore = get_vectorstore()
    vectorstore.add_documents(docs)
    return len(docs)


def embed_texts(texts: list[str], metadatas: list[dict] = None) -> int:
    """
    Embed raw texts and add to vector store.
    Returns the number of chunks added.
    """
    vectorstore = get_vectorstore()
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    return len(texts)
