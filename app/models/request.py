from pydantic import BaseModel


class ChunkModel(BaseModel):
    page_content: str
    metadata: dict


class EmbedRequest(BaseModel):
    chunks: list[ChunkModel]


class QueryRequest(BaseModel):
    question: str
