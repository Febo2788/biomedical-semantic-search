"""
FastAPI backend for semantic search.
Demonstrates: API development, REST endpoints, request/response handling
"""

from fastapi import FastAPI, HTTPException, Query
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
    description="""
    A semantic search API for biomedical literature.

    This API enables:
    - **Semantic Search**: Find documents by meaning, not just keywords
    - **Document Ingestion**: Add documents with metadata
    - **Filtered Search**: Filter results by metadata fields

    Built with: FastAPI, ChromaDB, Sentence-Transformers
    """,
    version="1.0.0"
)

# Global instances (initialized on startup)
embedder: Optional[EmbeddingPipeline] = None
store: Optional[VectorStore] = None


# Additional store for GEO experiments
geo_store: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    """Initialize ML models and vector stores on startup."""
    global embedder, store, geo_store
    print("Initializing semantic search API...")
    embedder = EmbeddingPipeline("all-MiniLM-L6-v2")

    # Literature search store
    store = VectorStore(
        collection_name="biomedical_docs",
        persist_directory="./chroma_data"
    )

    # Experimental data store
    geo_store = VectorStore(
        collection_name="geo_experiments",
        persist_directory="./chroma_data"
    )

    print(f"Loaded {store.count()} papers, {geo_store.count()} experiments")
    print("API ready!")


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Biomedical Semantic Search API"}


@app.get("/stats", response_model=StatsResponse, tags=["Info"])
async def get_stats():
    """Get statistics about the search index."""
    if store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    stats = store.get_stats()
    return StatsResponse(
        collection_name=stats["collection_name"],
        document_count=stats["document_count"],
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.embedding_dim
    )


@app.post("/documents", tags=["Documents"])
async def add_document(doc: DocumentInput):
    """
    Add a single document to the search index.

    The document will be embedded and stored for semantic search.
    """
    if store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Generate embedding
    embedding = embedder.encode(doc.text, show_progress=False).tolist()[0]

    # Add to store
    store.add_documents(
        documents=[doc.text],
        embeddings=[embedding],
        metadatas=[doc.metadata] if doc.metadata else None
    )

    return {
        "status": "success",
        "message": "Document added",
        "document_count": store.count()
    }


@app.post("/documents/batch", tags=["Documents"])
async def add_documents_batch(batch: BatchDocumentInput):
    """
    Add multiple documents to the search index.

    More efficient than adding documents one by one.
    """
    if store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    texts = [doc.text for doc in batch.documents]
    metadatas = [doc.metadata for doc in batch.documents]

    # Generate embeddings in batch
    embeddings = embedder.encode(texts, show_progress=False).tolist()

    # Add to store
    store.add_documents(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas if any(metadatas) else None
    )

    return {
        "status": "success",
        "message": f"Added {len(texts)} documents",
        "document_count": store.count()
    }


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Perform semantic search.

    Returns documents ranked by semantic similarity to the query.
    Optionally filter by metadata fields.
    """
    if store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if store.count() == 0:
        return SearchResponse(
            query=request.query,
            results=[],
            total_results=0
        )

    # Generate query embedding
    query_embedding = embedder.encode(request.query, show_progress=False).tolist()[0]

    # Search
    results = store.search(
        query_embedding=query_embedding,
        n_results=request.n_results,
        where=request.filter
    )

    # Format results
    search_results = []
    for doc, meta, dist, id_ in zip(
        results["documents"],
        results["metadatas"],
        results["distances"],
        results["ids"]
    ):
        similarity = 1 - dist  # Convert distance to similarity
        search_results.append(SearchResult(
            document=doc,
            similarity=round(similarity, 4),
            metadata=meta,
            id=id_
        ))

    return SearchResponse(
        query=request.query,
        results=search_results,
        total_results=len(search_results)
    )


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_get(
    q: str = Query(..., description="Search query"),
    n: int = Query(default=5, ge=1, le=50, description="Number of results")
):
    """
    Perform semantic search (GET method for simple queries).

    Example: /search?q=cancer treatment&n=10
    """
    request = SearchRequest(query=q, n_results=n)
    return await search(request)


# ============================================================
# EXPERIMENTAL DATA SEARCH (GEO)
# ============================================================

@app.post("/search/experiments", response_model=SearchResponse, tags=["Experiments"])
async def search_experiments(request: SearchRequest):
    """
    Search GEO experimental datasets by semantic similarity.

    Find gene expression experiments, RNA-seq studies, and other
    experimental data based on natural language descriptions.

    Example queries:
    - "breast cancer drug resistance"
    - "CRISPR screen in lung cancer"
    - "single cell RNA-seq tumor microenvironment"
    """
    if geo_store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if geo_store.count() == 0:
        return SearchResponse(
            query=request.query,
            results=[],
            total_results=0
        )

    # Generate query embedding
    query_embedding = embedder.encode(request.query, show_progress=False).tolist()[0]

    # Search experiments
    results = geo_store.search(
        query_embedding=query_embedding,
        n_results=request.n_results,
        where=request.filter
    )

    # Format results
    search_results = []
    for doc, meta, dist, id_ in zip(
        results["documents"],
        results["metadatas"],
        results["distances"],
        results["ids"]
    ):
        similarity = 1 - dist
        search_results.append(SearchResult(
            document=doc[:500],  # Truncate long descriptions
            similarity=round(similarity, 4),
            metadata=meta,
            id=id_
        ))

    return SearchResponse(
        query=request.query,
        results=search_results,
        total_results=len(search_results)
    )


@app.get("/search/experiments", response_model=SearchResponse, tags=["Experiments"])
async def search_experiments_get(
    q: str = Query(..., description="Search query"),
    n: int = Query(default=5, ge=1, le=50, description="Number of results")
):
    """
    Search experiments (GET method).

    Example: /search/experiments?q=BRCA1 expression&n=10
    """
    request = SearchRequest(query=q, n_results=n)
    return await search_experiments(request)


@app.get("/stats/experiments", tags=["Experiments"])
async def get_experiment_stats():
    """Get statistics about the experimental data index."""
    if geo_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "collection_name": "geo_experiments",
        "experiment_count": geo_store.count(),
        "description": "Gene expression experiments from NCBI GEO"
    }


if __name__ == "__main__":
    print("Starting Biomedical Semantic Search API...")
    print("API docs will be available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
