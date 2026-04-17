rag-app/
├── api-gateway/        # Routes requests, handles rate limiting
├── ingestion-service/  # PDF loading, chunking, metadata extraction
├── embedding-service/  # Vectorizes chunks, writes to vector DB
├── query-service/      # Retrieval + LLM call (the "hot" service)
├── auth-service/       # JWT auth, user management
└── docker-compose.yml  # Wires everything together locally