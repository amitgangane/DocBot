import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Embedding model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

    # LLM
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.6"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "500"))

    # Vector Store
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db")
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "8"))

    # Ingestion
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./documents")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Service URLs (for microservice mode)
    EMBEDDING_SERVICE_URL: str = os.getenv("EMBEDDING_URL", "http://localhost:8004")


settings = Settings()
