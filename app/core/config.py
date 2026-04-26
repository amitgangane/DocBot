import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Embedding model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

    # LLM
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.6"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "500"))

    # Vector Store
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "RAG-app")
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "5"))

    # Reranker
    RERANKER_INITIAL_K: int = int(os.getenv("RERANKER_INITIAL_K", "20"))

    # Query Rewriting
    REWRITE_MAX_TOKENS: int = int(os.getenv("REWRITE_MAX_TOKENS", "1000"))

    # Ingestion
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./documents")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Supabase PostgreSQL (for LangGraph checkpointer)
    SUPABASE_DB_URL: str = os.getenv("SUPABASE_DB_URL", "")
    CHECKPOINTER_POOL_MIN_SIZE: int = int(os.getenv("CHECKPOINTER_POOL_MIN_SIZE", "1"))
    CHECKPOINTER_POOL_MAX_SIZE: int = int(os.getenv("CHECKPOINTER_POOL_MAX_SIZE", "3"))
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    SUPABASE_STORAGE_BUCKET: str = os.getenv("SUPABASE_STORAGE_BUCKET", "documents")

    # Redis Cache
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL_RESPONSE: int = int(os.getenv("CACHE_TTL_RESPONSE", "3600"))  # 1 hour
    CACHE_TTL_EMBEDDING: int = int(os.getenv("CACHE_TTL_EMBEDDING", "86400"))  # 24 hours
    CACHE_TTL_RETRIEVAL: int = int(os.getenv("CACHE_TTL_RETRIEVAL", "3600"))  # 1 hour
    CACHE_TTL_RERANKER: int = int(os.getenv("CACHE_TTL_RERANKER", "3600"))  # 1 hour

    # Service URLs (for microservice mode)
    EMBEDDING_SERVICE_URL: str = os.getenv("EMBEDDING_URL", "http://localhost:8004")


settings = Settings()
