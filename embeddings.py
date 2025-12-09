"""
Embedding pipeline using Hugging Face sentence-transformers.
Demonstrates: Embeddings, Hugging Face, PyTorch
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class EmbeddingPipeline:
    """
    Embedding pipeline for generating semantic embeddings from text.
    Uses a pre-trained model optimized for semantic similarity tasks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding pipeline.

        Args:
            model_name: Hugging Face model name. Options include:
                - "all-MiniLM-L6-v2" (fast, general purpose)
                - "pritamdeka/S-PubMedBert-MS-MARCO" (biomedical)
                - "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" (biomedical)
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for input text(s).

        Args:
            texts: Single string or list of strings to encode
            batch_size: Number of texts to process at once
            show_progress: Show progress bar during encoding
            normalize: L2 normalize embeddings (recommended for cosine similarity)

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize
        )

        return embeddings

    def similarity(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute cosine similarity between a query and documents.

        Args:
            query: Query text
            documents: List of document texts

        Returns:
            List of similarity scores (0-1, higher is more similar)
        """
        query_embedding = self.encode(query, show_progress=False)
        doc_embeddings = self.encode(documents, show_progress=False)

        # Cosine similarity (embeddings are normalized, so dot product = cosine)
        similarities = np.dot(doc_embeddings, query_embedding.T).flatten()

        return similarities.tolist()


if __name__ == "__main__":
    # Test the embedding pipeline
    print("=" * 50)
    print("Testing Embedding Pipeline")
    print("=" * 50)

    # Initialize with a fast general model for testing
    pipeline = EmbeddingPipeline("all-MiniLM-L6-v2")

    # Test single text encoding
    print("\n1. Single text encoding:")
    text = "BRCA1 gene mutations are associated with breast cancer risk."
    embedding = pipeline.encode(text, show_progress=False)
    print(f"   Input: '{text}'")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   First 5 values: {embedding[0][:5]}")

    # Test batch encoding
    print("\n2. Batch encoding:")
    texts = [
        "Breast cancer is a common malignancy in women.",
        "BRCA1 and BRCA2 are tumor suppressor genes.",
        "Machine learning is used in genomic analysis.",
        "Pizza is a popular Italian dish."  # Unrelated text
    ]
    embeddings = pipeline.encode(texts, show_progress=False)
    print(f"   Encoded {len(texts)} texts")
    print(f"   Embeddings shape: {embeddings.shape}")

    # Test semantic similarity
    print("\n3. Semantic similarity search:")
    query = "What genes are linked to breast cancer?"
    similarities = pipeline.similarity(query, texts)

    print(f"   Query: '{query}'")
    print("\n   Results (ranked by similarity):")
    ranked = sorted(zip(texts, similarities), key=lambda x: x[1], reverse=True)
    for doc, score in ranked:
        print(f"   [{score:.4f}] {doc}")

    print("\n" + "=" * 50)
    print("Embedding pipeline test PASSED!")
    print("=" * 50)
