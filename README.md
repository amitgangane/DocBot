# DocBot - RAG Application

A Retrieval-Augmented Generation (RAG) application for querying PDF documents using OpenAI, Qdrant Cloud, and LangGraph.

## Features

- PDF upload and parsing (with image and table extraction)
- Document chunking with configurable size/overlap
- Vector embeddings using OpenAI `text-embedding-3-small`
- Cross-encoder reranking for improved retrieval accuracy
- Conversation memory with Supabase PostgreSQL (or SQLite fallback)
- Query rewriting for natural follow-up questions
- Cloud vector storage with Qdrant
- Redis caching for improved performance
- RAG-based Q&A using GPT-4o-mini
- FastAPI REST API

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
│   │   Generation    │  → LLM generates answer from context          │
│   └────────┬────────┘                                               │
│            ↓                                                        │
│   ┌─────────────────┐                                               │
│   │ Update Memory   │  → Save Q&A to chat history (Supabase/SQLite)  │
│   └────────┬────────┘                                               │
│            ↓                                                        │
│       Response                                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
DocBot/
├── app/                        # Main application
│   ├── api/
│   │   └── routes.py           # FastAPI endpoints
│   ├── core/
│   │   └── config.py           # Centralized settings
│   ├── db/
│   │   └── vector_db.py        # Qdrant Cloud client
│   ├── models/
│   │   ├── request.py          # Pydantic request models
│   │   └── response.py         # Pydantic response models
│   ├── services/
│   │   ├── state.py            # LangGraph AgentState
│   │   ├── nodes.py            # LangGraph node functions
│   │   ├── graph.py            # LangGraph builder
│   │   ├── rag_service.py      # Main RAG entry point
│   │   ├── embedding.py        # Embedding logic
│   │   ├── generation.py       # LLM calls
│   │   ├── reranker.py         # Cross-encoder reranking
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
├── checkpoints.db              # SQLite fallback (when Supabase not configured)
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

# Logging
LOG_LEVEL=INFO    # DEBUG, INFO, WARNING, ERROR

# LLM Settings
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.6
LLM_MAX_TOKENS=500

# Embedding Settings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Qdrant Cloud
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=RAG-app

# Supabase PostgreSQL (for chat history)
# Use the Connection Pooler URL (not direct connection) to avoid IPv6 issues
SUPABASE_DB_URL=postgresql://postgres.[PROJECT-REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres

# Retrieval & Reranking
RERANKER_INITIAL_K=10    # Docs to fetch before reranking
RETRIEVER_K=5            # Final docs after reranking

# Chunking
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# Redis Cache
REDIS_URL=redis://localhost:6379/0
CACHE_ENABLED=true
CACHE_TTL_RESPONSE=3600      # 1 hour
CACHE_TTL_RETRIEVAL=3600     # 1 hour
CACHE_TTL_RERANKER=3600      # 1 hour
```

### 3. Run the application

```bash
uvicorn app.main:app --reload
```

API available at: `http://localhost:8000`
Swagger docs at: `http://localhost:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest/upload` | Upload and index a PDF |
| POST | `/embed` | Embed raw chunks |
| GET | `/embed/count` | Get total chunk count |
| POST | `/query` | Query documents with conversation memory |
| GET | `/cache/stats` | Get cache statistics |
| POST | `/cache/clear` | Clear all caches |

### Example: Upload PDF

```bash
curl -X POST "http://localhost:8000/ingest/upload" \
  -F "file=@document.pdf"
```

### Example: Query with Conversation Memory

```bash
# First question
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the transformer architecture?", "thread_id": "user-123"}'

# Follow-up question (same thread_id)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does it use attention?", "thread_id": "user-123"}'
```

The follow-up query automatically rewrites "it" to "the Transformer" using conversation history.

### Example Response

```json
{
  "answer": "The Transformer is a model architecture that relies entirely on self-attention mechanisms...",
  "sources": 5
}
```

## Conversation Memory

DocBot uses LangGraph with Supabase PostgreSQL for conversation memory (falls back to SQLite if not configured):

- **Same `thread_id`** = Same conversation (history shared)
- **Different `thread_id`** = New conversation (fresh start)
- **Persistence** = Conversations survive server restarts

| thread_id | Query | Behavior |
|-----------|-------|----------|
| `user-123` | "What is attention?" | New conversation |
| `user-123` | "How does it work?" | Uses previous context, rewrites query |
| `user-456` | "What is attention?" | Separate conversation |

## Observability (LangSmith)

DocBot integrates with LangSmith for full pipeline tracing.

### Setup

Add to your `.env`:

```env
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=doc-bot-project
```

### What Gets Traced

| Component | Traced Data |
|-----------|-------------|
| Graph Execution | Full LangGraph run with all nodes |
| Query Rewrite | Input/output of rewrite node |
| Retrieval | Qdrant similarity search |
| Reranker | Cross-encoder scoring |
| Generation | LLM prompts, responses, tokens, latency |
| Memory | Chat history updates |

View traces at: [smith.langchain.com](https://smith.langchain.com)

## Logging

DocBot includes a centralized logging module for debugging and monitoring.

### Configuration

Set the log level in `.env`:

```env
LOG_LEVEL=INFO    # DEBUG, INFO, WARNING, ERROR
```

### Log Output

```
2024-01-15 10:23:45 | INFO  | routes       | Query received: "What is attention?" thread_id=user-123
2024-01-15 10:23:45 | DEBUG | nodes        | Rewriting query with 4 history messages
2024-01-15 10:23:46 | INFO  | retrieval    | Similarity search: 10 docs in 234ms
2024-01-15 10:23:46 | INFO  | reranker     | Reranked 10 -> 5 docs in 89ms
2024-01-15 10:23:48 | INFO  | generation   | LLM response: 156 chars in 1.8s
2024-01-15 10:23:48 | INFO  | routes       | Query complete: 5 sources in 2.3s
```

### Logged Components

| Module | What's Logged |
|--------|---------------|
| `routes` | Request/response, timing |
| `nodes` | Pipeline stage execution |
| `retrieval` | Document search timing |
| `reranker` | Reranking scores and timing |
| `generation` | LLM response timing |
| `graph` | Checkpointer initialization |
| `vector_db` | Qdrant connection status |

## Caching

DocBot uses Redis for multi-layer caching to improve performance.

### Setup

Start Redis with Docker:

```bash
docker run -d --name docbot-redis -p 6379:6379 redis
```

### Cache Layers

| Layer | Cache Key | TTL | Description |
|-------|-----------|-----|-------------|
| Response | `response:{query_hash}` | 1 hour | Full answer for repeated questions |
| Retrieval | `retrieve:{query_hash}` | 1 hour | Document search results |
| Reranker | `rerank:{query+docs_hash}` | 1 hour | Reranked document order |

### Cache Invalidation

- **Automatic**: Caches are invalidated when new PDFs are ingested
- **Manual**: Use `POST /cache/clear` to clear all caches

### Performance Gains

| Scenario | Without Cache | With Cache |
|----------|---------------|------------|
| Repeated query | ~3-4 seconds | ~50ms |
| Same docs reranked | ~3-4 seconds | ~2.5 seconds |

### Configuration

```env
REDIS_URL=redis://localhost:6379/0
CACHE_ENABLED=true              # Set to false to disable caching
CACHE_TTL_RESPONSE=3600         # 1 hour
CACHE_TTL_RETRIEVAL=3600        # 1 hour
CACHE_TTL_RERANKER=3600         # 1 hour
```

## TODO / Roadmap

- [ ] **Duplicate detection**: Add content-based hashing to prevent duplicate chunks
- [x] Add Redis caching for frequent queries
- [ ] Add S3/blob storage for PDFs
- [ ] Background workers for async ingestion
- [ ] Dockerfile and docker-compose setup
- [x] Add logging module
- [ ] Add more comprehensive tests
- [ ] Support for other document types (DOCX, TXT)
- [ ] Add evaluation metrics (RAGAS)
- [ ] Conditional routing in LangGraph (e.g., skip rerank for simple queries)

## Tech Stack

- **Framework**: FastAPI
- **Orchestration**: LangGraph (stateful workflows)
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Reranker**: Cross-encoder (ms-marco-MiniLM-L6-v2)
- **Vector Store**: Qdrant Cloud
- **Memory**: Supabase PostgreSQL (via LangGraph checkpointer, SQLite fallback)
- **Cache**: Redis
- **PDF Parsing**: PyMuPDF4LLM
- **Observability**: LangSmith
