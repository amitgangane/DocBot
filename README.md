# DocBot - RAG Application

A Retrieval-Augmented Generation (RAG) application for querying PDF documents using OpenAI, Qdrant Cloud, and LangGraph. Includes a modern Next.js frontend with streaming responses.

## Features

- **Backend (FastAPI)**
  - PDF upload and parsing (with image and table extraction)
  - Document chunking with configurable size/overlap
  - Vector embeddings using OpenAI `text-embedding-3-small`
  - Cross-encoder reranking for improved retrieval accuracy
  - Conversation memory with async LangGraph checkpointing
  - Supabase PostgreSQL checkpointer with in-memory fallback for local/dev use
  - Query rewriting for natural follow-up questions
  - Cloud vector storage with Qdrant
  - Redis caching (Upstash) for improved performance
  - Streaming responses via Server-Sent Events (SSE)
  - RAG-based Q&A using GPT-4o-mini

- **Frontend (Next.js)**
  - Modern chat interface with dark theme
  - Real-time token-by-token streaming responses
  - PDF upload support
  - Conversation history sidebar
  - Thread persistence across sessions

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DOCBOT ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐         ┌─────────────────────────────────────────┐   │
│   │   Next.js       │         │            FastAPI Backend              │   │
│   │   Frontend      │  HTTP   │                                         │   │
│   │   (Vercel)      │ ──────► │  ┌─────────┐    ┌─────────────────┐     │   │
│   │                 │         │  │ /query  │    │ LangGraph       │     │   │
│   │  - Chat UI      │◄─────── │  │ /query/ │───►│ Pipeline        │     │   │
│   │  - Streaming UI │   SSE   │  │ stream  │    │                 │     │   │
│   │  - PDF Upload   │   SSE   │  │ /upload │    │                 │     │   │
│   │  - Thread List  │         │  └─────────┘    └────────┬────────┘     │   │
│   └─────────────────┘         │                          │              │   │
│                               │                          ▼              │   │
│                               │  ┌──────────────────────────────────┐   │   │
│                               │  │           External Services      │   │   │
│                               │  │  ┌────────┐ ┌────────┐ ┌───────┐ │   │   │
│                               │  │  │Qdrant  │ │Supabase│ │Upstash│ │   │   │
│                               │  │  │Cloud   │ │Postgres│ │Redis  │ │   │   │
│                               │  │  └────────┘ └────────┘ └───────┘ │   │   │
│                               │  └──────────────────────────────────┘   │   │
│                               └─────────────────────────────────────────┘   │
│                                         (Render)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## LangGraph RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                      LANGGRAPH QUERY FLOW                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   User Question + thread_id                                         │
│        ↓                                                            │
│   ┌─────────────────┐                                               │
│   │  Query Rewrite  │  → Resolves pronouns using chat history       │
│   └────────┬────────┘    "How does it work?" → "How does the        │
│            ↓              Transformer work?"                        │
│   ┌─────────────────┐                                               │
│   │   Retrieval     │  → Fetch top 10 docs (embedding search)       │
│   └────────┬────────┘                                               │
│            ↓                                                        │
│   ┌─────────────────┐                                               │
│   │   Reranker      │  → Cross-encoder scores → keep top 5          │
│   └────────┬────────┘                                               │
│            ↓                                                        │
│   ┌─────────────────┐                                               │
│   │ Build Context   │  → Combine reranked docs into context         │
│   └────────┬────────┘                                               │
│            ↓                                                        │
│   ┌─────────────────┐                                               │
│   │   Generation    │  → LLM generates answer (streaming)           │
│   └────────┬────────┘                                               │
│            ↓                                                        │
│   ┌─────────────────┐                                               │
│   │ Update Memory   │  → Save Q&A to checkpointed chat history      │
│   └────────┬────────┘                                               │
│            ↓                                                        │
│       Response (streamed)                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
DocBot/
├── app/                        # FastAPI Backend
│   ├── api/
│   │   └── routes.py           # API endpoints (query, stream, upload, history)
│   ├── core/
│   │   ├── config.py           # Centralized settings
│   │   ├── cache.py            # Redis caching (Upstash)
│   │   └── logger.py           # Logging configuration
│   ├── db/
│   │   └── vector_db.py        # Qdrant Cloud client
│   ├── models/
│   │   ├── request.py          # Pydantic request models
│   │   └── response.py         # Pydantic response models
│   ├── services/
│   │   ├── state.py            # LangGraph AgentState
│   │   ├── nodes.py            # LangGraph node functions
│   │   ├── graph.py            # LangGraph builder + checkpointer lifecycle
│   │   ├── rag_service.py      # Async RAG entry point + SSE streaming
│   │   ├── embedding.py        # Embedding logic
│   │   ├── generation.py       # LLM calls
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   └── retrieval.py        # Vector search
│   └── main.py                 # FastAPI entrypoint with CORS
│
├── frontend/                   # Next.js Frontend
│   ├── src/app/
│   │   ├── page.tsx            # Main chat interface
│   │   ├── layout.tsx          # App layout
│   │   └── globals.css         # Tailwind styles
│   ├── package.json
│   ├── tailwind.config.js
│   └── .env.local              # Frontend environment variables
│
├── ingestion/                  # Data ingestion pipeline
│   ├── loader.py               # PDF loading with PyMuPDF
│   ├── chunking.py             # Document chunking
│   ├── embedder.py             # Embedding helper
│   └── indexer.py              # Full indexing pipeline
│
├── tests/                      # Unit tests
├── render.yaml                 # Render deployment config
├── requirements.txt
├── .env
└── README.md
```

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/DocBot.git
cd DocBot

# Backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

### 2. Configure environment

Create a `.env` file in the root directory:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token

# Logging
LOG_LEVEL=INFO

# LLM Settings
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.6
LLM_MAX_TOKENS=500

# Qdrant Cloud
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=RAG-app

# Supabase PostgreSQL (for LangGraph checkpointer)
SUPABASE_DB_URL=postgresql://postgres.[PROJECT-REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres

# Upstash Redis (note: rediss:// for TLS)
REDIS_URL=rediss://default:your_password@your_endpoint.upstash.io:6379
CACHE_ENABLED=true

# LangSmith (optional)
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=doc-bot-project
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8001
```

### 3. Run locally

**Terminal 1 - Backend:**
```bash
uvicorn app.main:app --reload --port 8001
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8001
- API Docs: http://localhost:8001/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest/upload` | Upload and index a PDF |
| POST | `/query` | Query documents (non-streaming) |
| POST | `/query/stream` | Query documents (streaming SSE) |
| GET | `/threads/{thread_id}/history` | Get conversation history |
| GET | `/embed/count` | Get total chunk count |
| GET | `/cache/stats` | Get cache statistics |
| POST | `/cache/clear` | Clear all caches |

### Example: Query with Streaming

```bash
curl -X POST "http://localhost:8001/query/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "What is the transformer architecture?", "thread_id": "user-123"}'
```

Response (Server-Sent Events):
```
event: metadata
data: {"sources": 5}

event: token
data: {"token": "The"}

event: token
data: {"token": " Transformer"}

event: token
data: {"token": " is"}

... (more tokens)

event: done
data: {"status": "complete"}
```

Possible error event:
```text
event: error
data: {"message": "Stream query failed"}
```

### Example: Get Thread History

```bash
curl "http://localhost:8001/threads/user-123/history"
```

Response:
```json
{
  "messages": [
    {"role": "user", "content": "What is attention?"},
    {"role": "assistant", "content": "Attention is a mechanism..."}
  ]
}
```

## Deployment

### Backend → Render

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Runtime:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables from your `.env`
6. Deploy

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → New Project
2. Import your GitHub repo
3. Set **Root Directory:** `frontend`
4. Add environment variable:
   ```
   NEXT_PUBLIC_API_URL=https://your-docbot-api.onrender.com
   ```
5. Deploy

## Conversation Memory

DocBot uses LangGraph checkpointing for conversation memory:

- **Same `thread_id`** = Same conversation (history shared)
- **Different `thread_id`** = New conversation (fresh start)
- **Frontend-generated thread IDs** = New chats automatically get a unique thread id
- **Persistence** = Conversations survive server restarts when `SUPABASE_DB_URL` is configured
- **Fallback behavior** = Without Supabase configured, history uses in-memory checkpoints only
- **Cache-aware** = Cached responses still save to chat history

The backend query path and history path both use LangGraph's async checkpoint APIs, which is required when running with the async Postgres saver.

## Caching

DocBot uses Upstash Redis (serverless) for multi-layer caching:

| Layer | Cache Key | TTL | Description |
|-------|-----------|-----|-------------|
| Response | `response:{query_hash}` | 1 hour | Full answer for repeated questions |
| Retrieval | `retrieve:{query_hash}` | 1 hour | Document search results |
| Reranker | `rerank:{query+docs_hash}` | 1 hour | Reranked document order |

**Note:** Cached responses now properly save Q&A to conversation history.

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend Framework** | FastAPI |
| **Frontend Framework** | Next.js 14 + React |
| **Styling** | Tailwind CSS |
| **Orchestration** | LangGraph |
| **LLM** | OpenAI GPT-4o-mini |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Reranker** | Cross-encoder (ms-marco-MiniLM-L6-v2) |
| **Vector Store** | Qdrant Cloud |
| **Database** | Supabase PostgreSQL |
| **Cache** | Upstash Redis |
| **PDF Parsing** | PyMuPDF4LLM |
| **Observability** | LangSmith |
| **Backend Hosting** | Render |
| **Frontend Hosting** | Vercel |

## TODO / Roadmap

- [x] Redis caching for frequent queries
- [x] Streaming responses (SSE)
- [x] Next.js frontend with chat UI
- [x] Conversation history in frontend
- [x] Upstash Redis (serverless)
- [ ] Duplicate detection (content-based hashing)
- [ ] S3/blob storage for PDFs
- [ ] Background workers for async ingestion
- [ ] Dockerfile and docker-compose setup
- [ ] Support for other document types (DOCX, TXT)
- [ ] RAGAS evaluation metrics
