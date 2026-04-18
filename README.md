# DocBot - RAG Application

A Retrieval-Augmented Generation (RAG) application for querying PDF documents using OpenAI and ChromaDB.

## Features

- PDF upload and parsing (with image and table extraction)
- Document chunking with configurable size/overlap
- Vector embeddings using OpenAI `text-embedding-3-small`
- Persistent storage with ChromaDB
- RAG-based Q&A using GPT-4o-mini
- FastAPI REST API

## Project Structure

```
DocBot/
├── app/                        # Main application
│   ├── api/
│   │   └── routes.py           # FastAPI endpoints
│   ├── core/
│   │   └── config.py           # Centralized settings
│   ├── db/
│   │   └── vector_db.py        # ChromaDB client
│   ├── models/
│   │   ├── request.py          # Pydantic request models
│   │   └── response.py         # Pydantic response models
│   ├── services/
│   │   ├── embedding.py        # Embedding logic
│   │   ├── generation.py       # LLM calls
│   │   ├── rag_service.py      # Main RAG pipeline
│   │   └── retrieval.py        # Vector search
│   └── main.py                 # FastAPI entrypoint
│
├── ingestion/                  # Data ingestion pipeline
│   ├── loader.py               # PDF loading with PyMuPDF
│   ├── chunking.py             # Document chunking
│   ├── embedder.py             # Embedding helper
│   └── indexer.py              # Full indexing pipeline
│
├── tests/                      # Unit tests
├── workers/                    # Background jobs (future)
├── requirements.txt
├── .env
└── README.md
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/DocBot.git
cd DocBot
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key

# Optional configs
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
CHUNK_SIZE=500
CHUNK_OVERLAP=100
RETRIEVER_K=8
```

### 3. Run the application

```bash
uvicorn app.main:app --reload
```

API available at: `http://localhost:8000`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest/upload` | Upload and index a PDF |
| POST | `/embed` | Embed raw chunks |
| GET | `/embed/count` | Get total chunk count |
| POST | `/query` | Query documents with a question |

### Example: Upload PDF

```bash
curl -X POST "http://localhost:8000/ingest/upload" \
  -F "file=@document.pdf"
```

### Example: Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the transformer architecture?"}'
```

## TODO / Roadmap

- [ ] **Duplicate detection**: Add content-based hashing to prevent duplicate chunks when re-uploading same PDF (even with different filename)
- [ ] Add Redis caching for frequent queries
- [ ] Add S3/blob storage for PDFs
- [ ] Background workers for async ingestion
- [ ] Dockerfile and docker-compose setup
- [ ] Add logging module
- [ ] Add more comprehensive tests
- [ ] Support for other document types (DOCX, TXT)

## Tech Stack

- **Framework**: FastAPI
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: ChromaDB
- **PDF Parsing**: PyMuPDF4LLM
- **Orchestration**: LangChain
