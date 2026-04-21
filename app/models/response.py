from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str


class IngestResponse(BaseModel):
    filename: str
    status: str
    pages_parsed: int
    chunk_count: int
    message: str


class EmbedResponse(BaseModel):
    status: str
    chunks_added: int
    total_chunks: int
    message: str


class CountResponse(BaseModel):
    total_chunks: int
    status: str


class QueryResponse(BaseModel):
    answer: str
    sources: int


class CacheStatsResponse(BaseModel):
    enabled: bool
    connected: bool
    memory_used: str = None
    keys: dict = None
    total_keys: int = None
    error: str = None


class CacheClearResponse(BaseModel):
    status: str
    keys_cleared: int
