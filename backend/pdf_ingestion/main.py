import os
import shutil
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from docloader import load_pdf
from chunking import chunk_documents

load_dotenv()


class IngestResponse(BaseModel):
    filename    : str
    status      : str
    pages_parsed: int
    chunk_count : int
    message     : str


class HealthResponse(BaseModel):
    status  : str
    service : str

app = FastAPI(
    title = "Ingestion Service",
    description = "Handles PDF upload, parsing and chunking",
    version     = "1.0.0"
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "documents")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_URL", "http://embedding-service:8004")

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/ingest/health",  response_model=HealthResponse)
async def heath_check():
    """
    Simple endpoint to confirm the service is running.
    Used by Docker and API gateway to check service status.
    """
    return HealthResponse(
        status  = "ok",
        service = "ingestion-service"
    )

@app.post("/ingest/upload", response_model = IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, parse it, chunk it,
    and send chunks to embedding service.
    """
    # validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code = 400,
            detail = "Only PDF files are supported"
        )
    #
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"Saved:{file_path}")

    # parse PDF
    try:
        docs = load_pdf(file_path)
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to parse PDF:{str(e)}"
        )

    # chunk documents
    try:
        chunks = chunk_documents(docs)
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to chunk document{str(e)}"
        )
    #send chunks to embedding service 
    try:
        chunks_payload = [
            {
                "page_content":chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{EMBEDDING_SERVICE_URL}/embd",
                json = {"chunks": chunks_payload}
            )
            response.raise_for_status()
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to send chunks to embedding services: {str(e)}"
        )
    
    # retrun response
    return IngestResponse(
        filename = file.filename,
        status = "Success",
        pages_parsed = len(docs),
        chunk_count = len(chunks),
        messsage = f"Successfully ingested {file.filename}"
    )