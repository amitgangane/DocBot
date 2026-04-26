# DocBot - RAG Application

A Retrieval-Augmented Generation (RAG) application for querying PDF documents using OpenAI, Qdrant Cloud, and LangGraph. Includes a modern Next.js frontend with streaming responses.

## Features

- **Backend (FastAPI)**
  - PDF upload and parsing (with image and table extraction)
  - Persistent PDF storage in Supabase Storage, with local fallback for development
  - Document chunking with configurable size/overlap and low-signal chunk filtering
  - Vector embeddings using OpenAI `text-embedding-3-small`
  - Stable document metadata (`document_id`, `filename`, `source_path`, `chunk_id`) on ingestion
  - Cross-encoder reranking for improved retrieval accuracy (`top 20` retrieved → `top 5` reranked)
  - Conversation memory with async LangGraph checkpointing
  - Supabase PostgreSQL checkpointer with in-memory fallback for local/dev use
  - Query rewriting for natural follow-up questions
  - Cloud vector storage with Qdrant
  - Redis caching (Upstash) for improved performance
  - Streaming responses via Server-Sent Events (SSE)
  - Document listing and per-document deletion from Qdrant
  - RAG-based Q&A using GPT-4o-mini
  - Retrieval evaluation suite for context precision, context recall, faithfulness, and answer relevancy

- **Frontend (Next.js)**
  - Clean light-theme chat interface with a fixed left sidebar
  - Real-time token-by-token streaming responses
  - Floating PDF manager for upload and document deletion
  - Conversation history sidebar
  - Thread persistence across sessions
  - Compact source chips with expandable source details
  - Thread reloads with persisted source metadata for newer answers

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
│   │  - PDF Manager  │         │  │ /ingest │    │                 │     │   │
│   │  - Thread List  │         │  │ /documents     │                │     │   │
│   │  - Source Chips │         │  └─────────┘    └────────┬────────┘     │   │
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
│   │   Retrieval     │  → Fetch top 20 docs (embedding search)       │
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
│   │   └── routes.py           # API endpoints (query, stream, upload, documents, history)
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
│   │   ├── document_storage.py # Supabase Storage upload/delete helpers
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

# Supabase Storage (for uploaded PDFs)
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
SUPABASE_STORAGE_BUCKET=documents

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
| GET | `/documents` | List indexed PDFs available for retrieval |
| DELETE | `/documents/{document_id}` | Delete one indexed PDF from Qdrant |
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
event: status
data: {"message": "Searching the document index"}

event: metadata
data: {"sources": 5, "source_items": [{"document_id": "...", "filename": "RAG_LLM.pdf", "source_path": "./documents/RAG_LLM.pdf", "page_number": 9, "chunk_id": "...", "excerpt": "..."}]}

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

### Example: List Indexed Documents

```bash
curl "http://localhost:8001/documents"
```

Response:
```json
[
  {
    "document_id": "c1712875-8ae7-4dd9-a0bd-97af9be5aaae",
    "filename": "RAG_LLM.pdf",
    "source_path": "./documents/RAG_LLM.pdf",
    "chunk_count": 352,
    "page_count": 20
  }
]
```

### Example: Delete One Indexed Document

```bash
curl -X DELETE "http://localhost:8001/documents/c1712875-8ae7-4dd9-a0bd-97af9be5aaae"
```

Response:
```json
{
  "status": "success",
  "document_id": "c1712875-8ae7-4dd9-a0bd-97af9be5aaae",
  "chunks_deleted": 352,
  "message": "Deleted document c1712875-8ae7-4dd9-a0bd-97af9be5aaae from the index"
}
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

## Document Indexing And Sources

Each uploaded PDF is stored in Qdrant as chunked embeddings with stable metadata:

- `document_id` identifies one uploaded PDF across all chunks
- `filename` and `source_path` are used for document listing and source display
- `page_number` is used to show page references in answers
- `chunk_id` and `chunk_index` help with traceability and deletion

When Supabase Storage is configured, uploaded PDFs are persisted in the configured bucket and `source_path` stores the storage object path. The backend still uses a temporary local file only during parsing because the current PDF loader expects a filesystem path. Without storage credentials, the app falls back to local file storage for development.

This metadata powers two user-facing features:

- the `Documents` section in the frontend PDF manager
- source chips and expandable source details under assistant answers

During ingestion, the app now filters out low-signal chunks such as reference-heavy citation blobs, generic figure summaries, and low-value caption artifacts while preserving useful body text and tables. If you reset or rebuild the Qdrant collection, re-upload your PDFs so every chunk uses the latest filtering and metadata schema.

## Frontend Notes

- The chat UI uses `/query/stream` and renders tokens incrementally.
- Before answer tokens arrive, the assistant bubble can show step-by-step retrieval status.
- Source references are shown as compact chips by default, with full excerpts hidden behind `Show details`.
- The PDF manager is a floating panel that lets you upload files, inspect indexed PDFs, and delete documents from the vector store.
- Thread history now restores saved `source_items` for answers that were generated after source persistence was added.
- The current UI is intentionally minimal: one upload entry point in the composer, one `Documents` list in the upload manager, and a cleaner empty state for new chats.

## Caching

DocBot uses Upstash Redis (serverless) for multi-layer caching:

| Layer | Cache Key | TTL | Description |
|-------|-----------|-----|-------------|
| Response | `response:{query_hash}` | 1 hour | Full answer for repeated questions |
| Retrieval | `retrieve:{query_hash}` | 1 hour | Document search results |
| Reranker | `rerank:{query+docs_hash}` | 1 hour | Reranked document order |

**Note:** Cached responses now properly save Q&A to conversation history.

## Evaluation

DocBot includes a lightweight metric-based retrieval and answer evaluation suite in [test_rag.py](/Users/amit/Desktop/DocuBot/DocBot/tests/test_rag.py).

Current checks:

- `ContextPrecision` to measure whether the top reranked chunks are relevant to the question/reference pair
- `ContextRecall` to measure whether the retrieved chunks collectively cover the reference answer
- `Faithfulness` to measure whether the app's generated answer is grounded in the retrieved contexts
- `AnswerRelevancy` to measure whether the generated answer directly addresses the user question

Run the suite with:

```bash
pytest tests/test_rag.py -q
```

The current evaluation set is defined directly in `TESTSET` inside `tests/test_rag.py`.

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

## TODO

- [ ] Add more questions and edge cases to the current `TESTSET`.
- [ ] Improve metadata-aware retrieval and reranking.
- [ ] Add a few more frontend polish items like source actions and thread rename/search.
- [ ] Add broader regression coverage for thread history, document deletion, and key UI states.

## Completed

- [x] Redis caching for frequent queries
- [x] Streaming responses (SSE)
- [x] Next.js frontend with chat UI
- [x] Conversation history in frontend
- [x] Upstash Redis (serverless)
- [x] Document-level metadata on ingestion
- [x] Per-document listing and deletion in Qdrant
- [x] Source persistence for newer thread history entries
- [x] LangSmith tracing across the RAG pipeline
- [x] Metric-based retrieval and answer evaluation suite (`ContextPrecision`, `ContextRecall`, `Faithfulness`, `AnswerRelevancy`)

## Demo

**Google Drive:** https://drive.google.com/file/d/191vjtXJAwm_HMQT0eacI2lyPi1dkMtdr/view?usp=drive_link    
**LangSmith Tracing:** https://drive.google.com/file/d/1p8h4MklDDPlxdNxydNYFgzgB1KdmUvhg/view?usp=sharing
