"""
Biomedical Semantic Search System - Main Entry Point

This demonstrates a complete semantic search pipeline for biomedical literature:
1. Embedding generation with Hugging Face transformers
2. Vector storage with ChromaDB
3. REST API with FastAPI
4. Retrieval quality evaluation

Run the API server:
    python main.py serve

Run full demo:
    python main.py demo
"""

import argparse
import sys


def run_demo():
    """Run a complete demonstration of all components."""
    print("=" * 70)
    print("BIOMEDICAL SEMANTIC SEARCH - FULL DEMONSTRATION")
    print("=" * 70)

    # 1. Test embedding pipeline
    print("\n" + "=" * 70)
    print("STEP 1: Embedding Pipeline")
    print("=" * 70)

    from embeddings import EmbeddingPipeline

    embedder = EmbeddingPipeline("all-MiniLM-L6-v2")

    test_texts = [
        "BRCA1 mutations increase breast cancer risk.",
        "Deep learning models predict drug response.",
        "The weather is nice today."
    ]

    print("\nTesting semantic similarity:")
    query = "Genetic factors in cancer"
    similarities = embedder.similarity(query, test_texts)

    print(f"Query: '{query}'")
    for text, sim in sorted(zip(test_texts, similarities), key=lambda x: x[1], reverse=True):
        print(f"  [{sim:.4f}] {text}")

    # 2. Test vector store
    print("\n" + "=" * 70)
    print("STEP 2: Vector Database")
    print("=" * 70)

    from vector_store import VectorStore

    store = VectorStore(
        collection_name="pubmed_abstracts",
        persist_directory="./chroma_data"
    )

    print(f"Collection: {store.collection_name}")
    print(f"Documents indexed: {store.count()}")

    if store.count() > 0:
        print("\nSample search:")
        query = "cancer immunotherapy response"
        query_emb = embedder.encode(query, show_progress=False).tolist()[0]
        results = store.search(query_emb, n_results=3)

        print(f"Query: '{query}'")
        for doc, meta, dist in zip(results["documents"], results["metadatas"], results["distances"]):
            sim = 1 - dist
            title = meta.get("title", "N/A")[:50]
            print(f"  [{sim:.3f}] {title}...")

    # 3. Test evaluation metrics
    print("\n" + "=" * 70)
    print("STEP 3: Retrieval Evaluation")
    print("=" * 70)

    from evaluation import RetrievalEvaluator, create_synthetic_evaluation_set

    if store.count() > 0:
        evaluator = RetrievalEvaluator(embedder, store)
        eval_queries = create_synthetic_evaluation_set(store)

        if eval_queries:
            result = evaluator.evaluate_dataset(eval_queries, k=5)
            print(f"\nEvaluation on {result.num_queries} queries (K=5):")
            print(f"  Precision@5: {result.precision_at_k:.4f}")
            print(f"  Recall@5:    {result.recall_at_k:.4f}")
            print(f"  MRR:         {result.mrr:.4f}")
            print(f"  NDCG@5:      {result.ndcg_at_k:.4f}")
    else:
        print("No documents indexed. Run 'python ingest_pubmed.py' first.")

    # 4. API info
    print("\n" + "=" * 70)
    print("STEP 4: REST API")
    print("=" * 70)

    print("\nAPI Endpoints:")
    print("  GET  /          - Health check")
    print("  GET  /stats     - Index statistics")
    print("  POST /documents - Add single document")
    print("  POST /documents/batch - Add multiple documents")
    print("  POST /search    - Semantic search")
    print("  GET  /search?q= - Simple search")

    print("\nTo start the API server:")
    print("  python main.py serve")
    print("\nAPI documentation available at:")
    print("  http://localhost:8000/docs")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


def run_server():
    """Start the FastAPI server."""
    import uvicorn
    from api import app

    print("Starting Biomedical Semantic Search API...")
    print("API docs: http://localhost:8000/docs")
    print("Press Ctrl+C to stop")

    uvicorn.run(app, host="0.0.0.0", port=8000)


def main():
    parser = argparse.ArgumentParser(
        description="Biomedical Semantic Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo    - Run full demonstration
  python main.py serve   - Start API server

For data ingestion:
  python ingest_pubmed.py
        """
    )

    parser.add_argument(
        "command",
        choices=["demo", "serve"],
        help="Command to run"
    )

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "serve":
        run_server()


if __name__ == "__main__":
    main()
