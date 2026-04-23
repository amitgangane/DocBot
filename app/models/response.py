from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str


class SourceItemResponse(BaseModel):
    document_id: str
    filename: str
    source_path: str
    page_number: int | None = None
    chunk_id: str | None = None
    excerpt: str


class IngestResponse(BaseModel):
    filename: str
    status: str
    pages_parsed: int
    chunk_count: int
    message: str
    document_id: str | None = None


class IndexedDocumentResponse(BaseModel):
    document_id: str
    filename: str
    source_path: str
    chunk_count: int
    page_count: int | None = None


class DeleteDocumentResponse(BaseModel):
    status: str
    document_id: str
    chunks_deleted: int
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
    source_items: list[SourceItemResponse] = []


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
