"""
FastAPI backend for semantic search.
Demonstrates: API development, REST endpoints, request/response handling
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn

from embeddings import EmbeddingPipeline
from vector_store import VectorStore


# Pydantic models for request/response validation
class DocumentInput(BaseModel):
    """Schema for adding a single document."""
    text: str = Field(..., description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata (e.g., {\"year\": 2023, \"topic\": \"cancer\"})"
    )


class BatchDocumentInput(BaseModel):
    """Schema for adding multiple documents."""
    documents: List[DocumentInput] = Field(..., description="List of documents to add")


class SearchRequest(BaseModel):
    """Schema for search queries."""
    query: str = Field(..., description="Search query text")
    n_results: int = Field(default=5, ge=1, le=50, description="Number of results")
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filter (e.g., {\"topic\": \"treatment\"})"
    )


class SearchResult(BaseModel):
    """Schema for a single search result."""
    document: str
    similarity: float
    metadata: Optional[Dict[str, Any]]
    id: str


class SearchResponse(BaseModel):
    """Schema for search response."""
    query: str
    results: List[SearchResult]
    total_results: int


class StatsResponse(BaseModel):
    """Schema for stats response."""
    collection_name: str
    document_count: int
    embedding_model: str
    embedding_dimension: int


# Initialize FastAPI app
app = FastAPI(
    title="Biomedical Semantic Search API",
    description="Semantic search across PubMed and GEO. UI at /, API docs at /docs.",
    version="1.0.0"
)

# Allow the frontend to call the API even if opened from a file:// or another origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
embedder: Optional[EmbeddingPipeline] = None
store: Optional[VectorStore] = None
geo_store: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    global embedder, store, geo_store
    print("Initializing semantic search API...")
    embedder = EmbeddingPipeline("all-MiniLM-L6-v2")
    store = VectorStore(collection_name="pubmed_abstracts", persist_directory="./chroma_data")
    geo_store = VectorStore(collection_name="geo_experiments", persist_directory="./chroma_data")
    print(f"Loaded {store.count()} papers, {geo_store.count()} experiments")
    print("API ready! Open http://localhost:8000/")


# ---------- Frontend ----------
# Serve the custom search UI from /static and make "/" return index.html.
STATIC_DIR = Path(__file__).parent / "static"

@app.get("/", include_in_schema=False)
async def homepage():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"status": "healthy", "message": "UI not installed — expected static/index.html"}

# Mount any other static assets (if you add CSS/JS files later)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- Health ----------
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


@app.get("/stats", response_model=StatsResponse, tags=["Info"])
async def get_stats():
    if store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    stats = store.get_stats()
    return StatsResponse(
        collection_name=stats["collection_name"],
        document_count=stats["document_count"],
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.embedding_dim
    )


# ---------- Documents ----------
@app.post("/documents", tags=["Documents"])
async def add_document(doc: DocumentInput):
    if store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    embedding = embedder.encode(doc.text, show_progress=False).tolist()[0]
    store.add_documents(
        documents=[doc.text],
        embeddings=[embedding],
        metadatas=[doc.metadata] if doc.metadata else None
    )
    return {"status": "success", "message": "Document added", "document_count": store.count()}


@app.post("/documents/batch", tags=["Documents"])
async def add_documents_batch(batch: BatchDocumentInput):
    if store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    texts = [doc.text for doc in batch.documents]
    metadatas = [doc.metadata for doc in batch.documents]
    embeddings = embedder.encode(texts, show_progress=False).tolist()
    store.add_documents(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas if any(metadatas) else None
    )
    return {"status": "success", "message": f"Added {len(texts)} documents", "document_count": store.count()}


# ---------- Literature search ----------
@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    if store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    if store.count() == 0:
        return SearchResponse(query=request.query, results=[], total_results=0)

    query_embedding = embedder.encode(request.query, show_progress=False).tolist()[0]
    results = store.search(
        query_embedding=query_embedding,
        n_results=request.n_results,
        where=request.filter
    )

    search_results = []
    for doc, meta, dist, id_ in zip(
        results["documents"], results["metadatas"], results["distances"], results["ids"]
    ):
        similarity = 1 - dist
        search_results.append(SearchResult(
            document=doc,
            similarity=round(similarity, 4),
            metadata=meta,
            id=id_
        ))

    return SearchResponse(query=request.query, results=search_results, total_results=len(search_results))


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_get(
    q: str = Query(..., description="Search query"),
    n: int = Query(default=5, ge=1, le=50, description="Number of results")
):
    return await search(SearchRequest(query=q, n_results=n))


# ---------- Experiments search ----------
@app.post("/search/experiments", response_model=SearchResponse, tags=["Experiments"])
async def search_experiments(request: SearchRequest):
    if geo_store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    if geo_store.count() == 0:
        return SearchResponse(query=request.query, results=[], total_results=0)

    query_embedding = embedder.encode(request.query, show_progress=False).tolist()[0]
    results = geo_store.search(
        query_embedding=query_embedding,
        n_results=request.n_results,
        where=request.filter
    )

    search_results = []
    for doc, meta, dist, id_ in zip(
        results["documents"], results["metadatas"], results["distances"], results["ids"]
    ):
        similarity = 1 - dist
        search_results.append(SearchResult(
            document=doc[:500],
            similarity=round(similarity, 4),
            metadata=meta,
            id=id_
        ))

    return SearchResponse(query=request.query, results=search_results, total_results=len(search_results))


@app.get("/search/experiments", response_model=SearchResponse, tags=["Experiments"])
async def search_experiments_get(
    q: str = Query(..., description="Search query"),
    n: int = Query(default=5, ge=1, le=50, description="Number of results")
):
    return await search_experiments(SearchRequest(query=q, n_results=n))


@app.get("/stats/experiments", tags=["Experiments"])
async def get_experiment_stats():
    if geo_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return {
        "collection_name": "geo_experiments",
        "experiment_count": geo_store.count(),
        "description": "Gene expression experiments from NCBI GEO"
    }


if __name__ == "__main__":
    print("Starting Biomedical Semantic Search API...")
    print("UI:       http://localhost:8000/")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
