import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title       = "Embedding Service",
    description = "Receives chunks, embeds them and stores in ChromaDB",
    version     = "1.0.0"
)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
PERSIST_DIR = os.getenv("CHROMA_DIR", "/embedding-service/chroma_db")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")

os.makedirs(PERSIST_DIR, exist_ok=True)

# ─────────────────────────────────────────
# INIT EMBEDDINGS + VECTORSTORE
# ─────────────────────────────────────────
embeddings  = OpenAIEmbeddings(
    model   = "text-embedding-3-small",
    api_key = OPENAI_KEY
)

vectorstore = Chroma(
    persist_directory  = PERSIST_DIR,
    embedding_function = embeddings
)

class ChunkModel(BaseModel):
    page_content: str
    metadata    : dict


class EmbedRequest(BaseModel):
    chunks: list[ChunkModel]


class EmbedResponse(BaseModel):
    status     : str
    chunks_added: int
    total_chunks: int
    message    : str


class HealthResponse(BaseModel):
    status  : str
    service : str


class CountResponse(BaseModel):
    total_chunks: int
    status      : str

@app.get("/embed/health", response_model = HealthResponse)
async def health_check():
    """
    Confirms the embedding service is running
    and ChromaDB is accessible
    """
    try:
        count = vectorstore._collection.count()
        return HealthResponse(
            status = "ok",
            service = "embedding-service"
        )
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"ChromaDB not accessible: {str(e)}"
        )


# ─────────────────────────────────────────
# COUNT ENDPOINT
# ─────────────────────────────────────────

@app.get("/embed/count", response_model=CountResponse)
async def get_chunk_count():
    """
    Returns how many chunks are currently
    stored in the vector store.
    """
    try:
        count = vectorstore._collection.count()
        return CountResponse(
            total_chunks = count,
            status       = "ok"
        )
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Failed to get count: {str(e)}"
        )
