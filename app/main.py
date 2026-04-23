from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services.graph import init_checkpointer, close_checkpointer

from app.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize checkpointer once at startup
    await init_checkpointer()

    yield

    # Clean shutdown
    await close_checkpointer()

app = FastAPI(
    title="RAG Application",
    description="PDF ingestion, embedding, and RAG query service",
    version="1.0.0",
    lifespan=lifespan,  
)

# CORS middleware - allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
