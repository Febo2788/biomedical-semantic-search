"""
Vector database using ChromaDB for semantic search.
Demonstrates: Vector databases, similarity search, metadata filtering
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os


class VectorStore:
    """
    Vector store for storing and querying document embeddings.
    Uses ChromaDB as the underlying vector database.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name for the document collection
            persist_directory: Directory to persist data (None for in-memory)
        """
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        self.collection_name = collection_name
        print(f"Vector store initialized. Collection: {collection_name}")

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents with their embeddings to the store.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            ids: Optional list of unique IDs (auto-generated if not provided)
            metadatas: Optional list of metadata dicts for each document
        """
        if ids is None:
            # Generate IDs based on current count
            start_id = self.collection.count()
            ids = [f"doc_{start_id + i}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Added {len(documents)} documents to collection")

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: List[str] = ["documents", "metadatas", "distances"]
    ) -> Dict[str, Any]:
        """
        Search for similar documents using a query embedding.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {"year": {"$gte": 2020}})
            include: What to include in results

        Returns:
            Dict with ids, documents, metadatas, distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=include
        )

        # Flatten single-query results
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        print(f"Deleted collection: {self.collection_name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
        }


if __name__ == "__main__":
    from embeddings import EmbeddingPipeline

    print("=" * 50)
    print("Testing Vector Store")
    print("=" * 50)

    # Initialize embedding pipeline and vector store
    embedder = EmbeddingPipeline("all-MiniLM-L6-v2")
    store = VectorStore(collection_name="test_collection")

    # Sample biomedical documents with metadata
    documents = [
        "BRCA1 mutations significantly increase the risk of breast and ovarian cancer.",
        "p53 is a tumor suppressor gene commonly mutated in various cancers.",
        "EGFR inhibitors are targeted therapies used in lung cancer treatment.",
        "Immunotherapy has revolutionized treatment for melanoma patients.",
        "CRISPR-Cas9 enables precise genome editing for disease research.",
        "Single-cell RNA sequencing reveals cellular heterogeneity in tumors.",
        "Machine learning algorithms can predict patient outcomes from genomic data.",
        "The gut microbiome influences response to cancer immunotherapy."
    ]

    metadatas = [
        {"topic": "genetics", "cancer_type": "breast", "year": 2020},
        {"topic": "genetics", "cancer_type": "general", "year": 2019},
        {"topic": "treatment", "cancer_type": "lung", "year": 2021},
        {"topic": "treatment", "cancer_type": "melanoma", "year": 2022},
        {"topic": "technology", "cancer_type": "general", "year": 2023},
        {"topic": "technology", "cancer_type": "general", "year": 2022},
        {"topic": "technology", "cancer_type": "general", "year": 2021},
        {"topic": "treatment", "cancer_type": "general", "year": 2023}
    ]

    # Generate embeddings
    print("\n1. Generating embeddings for documents...")
    embeddings = embedder.encode(documents, show_progress=False).tolist()
    print(f"   Generated {len(embeddings)} embeddings")

    # Add to vector store
    print("\n2. Adding documents to vector store...")
    store.add_documents(documents, embeddings, metadatas=metadatas)
    print(f"   Store stats: {store.get_stats()}")

    # Basic semantic search
    print("\n3. Semantic search (no filter):")
    query = "What genetic mutations cause cancer?"
    query_embedding = embedder.encode(query, show_progress=False).tolist()[0]
    results = store.search(query_embedding, n_results=3)

    print(f"   Query: '{query}'")
    print("\n   Top 3 results:")
    for doc, meta, dist in zip(results["documents"], results["metadatas"], results["distances"]):
        similarity = 1 - dist  # ChromaDB returns distance, convert to similarity
        print(f"   [{similarity:.4f}] {doc[:60]}...")
        print(f"            Metadata: {meta}")

    # Filtered search
    print("\n4. Filtered search (topic='treatment'):")
    results = store.search(
        query_embedding,
        n_results=3,
        where={"topic": "treatment"}
    )
    print("\n   Results with treatment topic:")
    for doc, meta, dist in zip(results["documents"], results["metadatas"], results["distances"]):
        similarity = 1 - dist
        print(f"   [{similarity:.4f}] {doc[:60]}...")

    # Year filter
    print("\n5. Filtered search (year >= 2022):")
    results = store.search(
        query_embedding,
        n_results=3,
        where={"year": {"$gte": 2022}}
    )
    print("\n   Results from 2022 or later:")
    for doc, meta, dist in zip(results["documents"], results["metadatas"], results["distances"]):
        similarity = 1 - dist
        print(f"   [{similarity:.4f}] {doc[:60]}... (year: {meta['year']})")

    print("\n" + "=" * 50)
    print("Vector store test PASSED!")
    print("=" * 50)
