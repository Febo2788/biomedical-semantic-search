"""
Retrieval quality evaluation module.
Implements standard IR metrics to evaluate semantic search performance.
Demonstrates: Evaluation metrics, retrieval quality assessment
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from embeddings import EmbeddingPipeline
from vector_store import VectorStore


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    precision_at_k: float
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_k: float  # Normalized Discounted Cumulative Gain
    k: int
    num_queries: int


class RetrievalEvaluator:
    """
    Evaluates retrieval quality using standard IR metrics.

    Metrics implemented:
    - Precision@K: Fraction of retrieved docs that are relevant
    - Recall@K: Fraction of relevant docs that were retrieved
    - MRR: Mean Reciprocal Rank (1/rank of first relevant doc)
    - NDCG@K: Normalized Discounted Cumulative Gain
    """

    def __init__(
        self,
        embedder: EmbeddingPipeline,
        store: VectorStore
    ):
        """
        Initialize evaluator.

        Args:
            embedder: Embedding pipeline for encoding queries
            store: Vector store to evaluate
        """
        self.embedder = embedder
        self.store = store

    def precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K.

        Precision@K = (# relevant in top-K) / K

        Args:
            retrieved_ids: IDs of retrieved documents (ordered by rank)
            relevant_ids: IDs of documents marked as relevant
            k: Number of top results to consider

        Returns:
            Precision score (0-1)
        """
        if k == 0:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_ids))

        return relevant_in_top_k / k

    def recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K.

        Recall@K = (# relevant in top-K) / (total # relevant)

        Args:
            retrieved_ids: IDs of retrieved documents (ordered by rank)
            relevant_ids: IDs of documents marked as relevant
            k: Number of top results to consider

        Returns:
            Recall score (0-1)
        """
        if len(relevant_ids) == 0:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_ids))

        return relevant_in_top_k / len(relevant_ids)

    def reciprocal_rank(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate Reciprocal Rank.

        RR = 1 / (rank of first relevant document)

        Args:
            retrieved_ids: IDs of retrieved documents (ordered by rank)
            relevant_ids: IDs of documents marked as relevant

        Returns:
            Reciprocal rank score (0-1)
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)

        return 0.0

    def dcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Discounted Cumulative Gain at K.

        DCG@K = sum(rel_i / log2(i + 1)) for i in 1..K

        Args:
            retrieved_ids: IDs of retrieved documents
            relevant_ids: IDs of relevant documents
            k: Number of results to consider

        Returns:
            DCG score
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = 1 if doc_id in relevant_ids else 0
            dcg += rel / np.log2(i + 2)  # +2 because log2(1) = 0

        return dcg

    def ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.

        NDCG@K = DCG@K / IDCG@K

        Args:
            retrieved_ids: IDs of retrieved documents
            relevant_ids: IDs of relevant documents
            k: Number of results to consider

        Returns:
            NDCG score (0-1)
        """
        dcg = self.dcg_at_k(retrieved_ids, relevant_ids, k)

        # Ideal DCG: all relevant docs ranked first
        ideal_k = min(k, len(relevant_ids))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_query(
        self,
        query: str,
        relevant_ids: List[str],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate retrieval for a single query.

        Args:
            query: Search query text
            relevant_ids: IDs of documents relevant to this query
            k: Number of results to evaluate

        Returns:
            Dict with all metrics
        """
        # Get query embedding and search
        query_embedding = self.embedder.encode(query, show_progress=False).tolist()[0]
        results = self.store.search(query_embedding, n_results=k)
        retrieved_ids = results["ids"]

        return {
            "precision_at_k": self.precision_at_k(retrieved_ids, relevant_ids, k),
            "recall_at_k": self.recall_at_k(retrieved_ids, relevant_ids, k),
            "mrr": self.reciprocal_rank(retrieved_ids, relevant_ids),
            "ndcg_at_k": self.ndcg_at_k(retrieved_ids, relevant_ids, k)
        }

    def evaluate_dataset(
        self,
        queries_and_relevance: List[Tuple[str, List[str]]],
        k: int = 10
    ) -> EvaluationResult:
        """
        Evaluate retrieval across multiple queries.

        Args:
            queries_and_relevance: List of (query, relevant_ids) tuples
            k: Number of results to evaluate

        Returns:
            EvaluationResult with averaged metrics
        """
        all_metrics = []

        for query, relevant_ids in queries_and_relevance:
            metrics = self.evaluate_query(query, relevant_ids, k)
            all_metrics.append(metrics)

        # Average across all queries
        avg_precision = np.mean([m["precision_at_k"] for m in all_metrics])
        avg_recall = np.mean([m["recall_at_k"] for m in all_metrics])
        avg_mrr = np.mean([m["mrr"] for m in all_metrics])
        avg_ndcg = np.mean([m["ndcg_at_k"] for m in all_metrics])

        return EvaluationResult(
            precision_at_k=avg_precision,
            recall_at_k=avg_recall,
            mrr=avg_mrr,
            ndcg_at_k=avg_ndcg,
            k=k,
            num_queries=len(queries_and_relevance)
        )


def create_synthetic_evaluation_set(store: VectorStore) -> List[Tuple[str, List[str]]]:
    """
    Create a synthetic evaluation dataset based on document metadata.

    Uses keyword matching to establish pseudo-relevance judgments.
    In production, you'd have human-labeled relevance judgments.
    """
    # Get all documents with their metadata
    all_docs = store.collection.get(include=["metadatas", "documents"])

    evaluation_queries = []

    # Query 1: Immunotherapy
    query1 = "immunotherapy cancer treatment response prediction"
    relevant1 = [
        id_ for id_, meta, doc in zip(all_docs["ids"], all_docs["metadatas"], all_docs["documents"])
        if meta and ("immunotherap" in doc.lower() or "immune" in doc.lower())
    ]
    if relevant1:
        evaluation_queries.append((query1, relevant1))

    # Query 2: Machine learning genomics
    query2 = "machine learning genomic analysis prediction"
    relevant2 = [
        id_ for id_, meta, doc in zip(all_docs["ids"], all_docs["metadatas"], all_docs["documents"])
        if meta and ("machine learning" in doc.lower() or "deep learning" in doc.lower())
    ]
    if relevant2:
        evaluation_queries.append((query2, relevant2))

    # Query 3: Biomarkers
    query3 = "biomarker discovery cancer prognosis"
    relevant3 = [
        id_ for id_, meta, doc in zip(all_docs["ids"], all_docs["metadatas"], all_docs["documents"])
        if meta and ("biomarker" in doc.lower() or "prognostic" in doc.lower())
    ]
    if relevant3:
        evaluation_queries.append((query3, relevant3))

    # Query 4: Single cell sequencing
    query4 = "single cell RNA sequencing tumor analysis"
    relevant4 = [
        id_ for id_, meta, doc in zip(all_docs["ids"], all_docs["metadatas"], all_docs["documents"])
        if meta and ("single-cell" in doc.lower() or "single cell" in doc.lower())
    ]
    if relevant4:
        evaluation_queries.append((query4, relevant4))

    return evaluation_queries


if __name__ == "__main__":
    print("=" * 60)
    print("Retrieval Quality Evaluation")
    print("=" * 60)

    # Initialize components
    embedder = EmbeddingPipeline("all-MiniLM-L6-v2")
    store = VectorStore(
        collection_name="pubmed_abstracts",
        persist_directory="./chroma_data"
    )

    if store.count() == 0:
        print("No documents in store. Run ingest_pubmed.py first.")
        exit(1)

    print(f"\nDocuments in store: {store.count()}")

    # Create evaluator
    evaluator = RetrievalEvaluator(embedder, store)

    # Create synthetic evaluation set
    print("\nCreating evaluation dataset...")
    eval_queries = create_synthetic_evaluation_set(store)
    print(f"Created {len(eval_queries)} evaluation queries")

    for query, relevant in eval_queries:
        print(f"  - '{query[:40]}...' ({len(relevant)} relevant docs)")

    # Evaluate at different K values
    print("\n" + "-" * 60)
    print("Evaluation Results")
    print("-" * 60)

    for k in [3, 5, 10]:
        result = evaluator.evaluate_dataset(eval_queries, k=k)
        print(f"\n@ K={k}:")
        print(f"  Precision@{k}: {result.precision_at_k:.4f}")
        print(f"  Recall@{k}:    {result.recall_at_k:.4f}")
        print(f"  MRR:           {result.mrr:.4f}")
        print(f"  NDCG@{k}:      {result.ndcg_at_k:.4f}")

    # Detailed per-query analysis
    print("\n" + "-" * 60)
    print("Per-Query Analysis (K=5)")
    print("-" * 60)

    for query, relevant_ids in eval_queries:
        metrics = evaluator.evaluate_query(query, relevant_ids, k=5)
        print(f"\nQuery: '{query[:50]}...'")
        print(f"  Relevant docs: {len(relevant_ids)}")
        print(f"  P@5: {metrics['precision_at_k']:.3f} | "
              f"R@5: {metrics['recall_at_k']:.3f} | "
              f"MRR: {metrics['mrr']:.3f} | "
              f"NDCG: {metrics['ndcg_at_k']:.3f}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
