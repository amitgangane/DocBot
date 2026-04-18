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
