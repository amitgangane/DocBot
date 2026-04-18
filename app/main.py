from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="RAG Application",
    description="PDF ingestion, embedding, and RAG query service",
    version="1.0.0",
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
