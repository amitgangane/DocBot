from uuid import uuid4

from pydantic import BaseModel, Field


class ChunkModel(BaseModel):
    page_content: str
    metadata: dict


class EmbedRequest(BaseModel):
    chunks: list[ChunkModel]


class QueryRequest(BaseModel):
    question: str
    thread_id: str = Field(default_factory=lambda: f"thread-{uuid4()}")  # Session ID for conversation memory
