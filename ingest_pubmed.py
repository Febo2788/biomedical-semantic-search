"""
PubMed data ingestion script.
Fetches biomedical abstracts from PubMed and indexes them for semantic search.
Demonstrates: API integration, data processing, batch operations
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import time

from embeddings import EmbeddingPipeline
from vector_store import VectorStore


class PubMedFetcher:
    """
    Fetches abstracts from PubMed using the E-utilities API.
    Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25499/
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: str = "felix.borrego02@gmail.com", api_key: str = "745f2b98fc7384ec80b6c74e4636087a2508"):
        """
        Initialize the fetcher.

        Args:
            email: Email for NCBI API (required by their terms of service)
            api_key: NCBI API key for higher rate limits (10 req/sec)
        """
        self.email = email
        self.api_key = api_key

    def search(self, query: str, max_results: int = 100) -> List[str]:
        """
        Search PubMed and return PMIDs.

        Args:
            query: Search query (e.g., "breast cancer BRCA1")
            max_results: Maximum number of results to return

        Returns:
            List of PubMed IDs (PMIDs)
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
            "api_key": self.api_key
        }

        response = requests.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
        response.raise_for_status()

        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])

        return pmids

    def fetch_abstracts(self, pmids: List[str], batch_size: int = 200) -> List[Dict[str, Any]]:
        """
        Fetch full records for given PMIDs with batching for large requests.

        Args:
            pmids: List of PubMed IDs
            batch_size: Number of PMIDs to fetch per request (max ~200 recommended)

        Returns:
            List of article records with title, abstract, metadata
        """
        if not pmids:
            return []

        all_articles = []

        # Process in batches
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]

            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "email": self.email,
                "api_key": self.api_key
            }

            try:
                response = requests.get(f"{self.BASE_URL}/efetch.fcgi", params=params)
                response.raise_for_status()

                # Parse XML response
                root = ET.fromstring(response.content)

                for article in root.findall(".//PubmedArticle"):
                    record = self._parse_article(article)
                    if record and record.get("abstract"):
                        all_articles.append(record)

                # Rate limiting between batches (10 req/sec with API key)
                if i + batch_size < len(pmids):
                    time.sleep(0.1)

            except Exception as e:
                print(f"   Error fetching batch {i//batch_size + 1}: {e}")
                time.sleep(1)
                continue

        return all_articles

    def _parse_article(self, article: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse a single article from XML."""
        try:
            # Get PMID
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None

            # Get title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""

            # Get abstract
            abstract_parts = []
            for abstract_text in article.findall(".//AbstractText"):
                if abstract_text.text:
                    label = abstract_text.get("Label", "")
                    if label:
                        abstract_parts.append(f"{label}: {abstract_text.text}")
                    else:
                        abstract_parts.append(abstract_text.text)
            abstract = " ".join(abstract_parts)

            # Get publication year
            year_elem = article.find(".//PubDate/Year")
            year = int(year_elem.text) if year_elem is not None and year_elem.text else None

            # Get journal
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""

            # Get keywords/MeSH terms
            keywords = []
            for mesh in article.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    keywords.append(mesh.text)

            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
                "keywords": keywords[:10]  # Limit to 10 keywords
            }
        except Exception as e:
            print(f"Error parsing article: {e}")
            return None


def ingest_pubmed_data(
    query: str,
    max_articles: int = 50,
    collection_name: str = "pubmed_abstracts"
) -> Dict[str, Any]:
    """
    Ingest PubMed articles into the vector store.

    Args:
        query: PubMed search query
        max_articles: Maximum articles to fetch
        collection_name: Name for the ChromaDB collection

    Returns:
        Statistics about the ingestion
    """
    print(f"Starting PubMed ingestion for query: '{query}'")
    print(f"Max articles: {max_articles}")
    print("-" * 50)

    # Initialize components
    fetcher = PubMedFetcher()
    embedder = EmbeddingPipeline("all-MiniLM-L6-v2")
    store = VectorStore(
        collection_name=collection_name,
        persist_directory="./chroma_data"
    )

    # Search PubMed
    print("\n1. Searching PubMed...")
    pmids = fetcher.search(query, max_results=max_articles)
    print(f"   Found {len(pmids)} articles")

    if not pmids:
        return {"status": "no_results", "articles_found": 0}

    # Fetch abstracts
    print("\n2. Fetching abstracts...")
    articles = fetcher.fetch_abstracts(pmids)
    print(f"   Retrieved {len(articles)} articles with abstracts")

    if not articles:
        return {"status": "no_abstracts", "articles_found": len(pmids)}

    # Prepare documents for indexing
    print("\n3. Preparing documents...")
    documents = []
    metadatas = []
    ids = []

    for article in articles:
        # Combine title and abstract for richer semantic content
        text = f"{article['title']}. {article['abstract']}"
        documents.append(text)

        metadata = {
            "pmid": article["pmid"] or "",
            "title": article["title"] or "",
            "year": article["year"] if article["year"] else 0,
            "journal": article["journal"] or "",
            "keywords": ", ".join(article["keywords"]) if article["keywords"] else ""
        }
        metadatas.append(metadata)
        ids.append(f"pmid_{article['pmid']}")

    # Generate embeddings
    print("\n4. Generating embeddings...")
    embeddings = embedder.encode(documents, batch_size=16, show_progress=True).tolist()

    # Add to vector store
    print("\n5. Adding to vector store...")
    store.add_documents(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    stats = {
        "status": "success",
        "query": query,
        "articles_found": len(pmids),
        "articles_indexed": len(documents),
        "collection_name": collection_name,
        "total_documents": store.count()
    }

    print("\n" + "=" * 50)
    print("Ingestion Complete!")
    print(f"  Query: {query}")
    print(f"  Articles indexed: {len(documents)}")
    print(f"  Total in collection: {store.count()}")
    print("=" * 50)

    return stats


def demo_search(collection_name: str = "pubmed_abstracts"):
    """Demo semantic search on ingested data."""
    print("\n" + "=" * 50)
    print("Demo: Semantic Search on PubMed Abstracts")
    print("=" * 50)

    embedder = EmbeddingPipeline("all-MiniLM-L6-v2")
    store = VectorStore(
        collection_name=collection_name,
        persist_directory="./chroma_data"
    )

    if store.count() == 0:
        print("No documents in collection. Run ingestion first.")
        return

    queries = [
        "What genetic mutations increase cancer risk?",
        "How does immunotherapy work for cancer treatment?",
        "Machine learning applications in oncology"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)

        query_embedding = embedder.encode(query, show_progress=False).tolist()[0]
        results = store.search(query_embedding, n_results=3)

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"],
            results["metadatas"],
            results["distances"]
        )):
            similarity = 1 - dist
            title = meta.get("title", "N/A")[:60]
            year = meta.get("year", "N/A")
            print(f"\n  {i+1}. [{similarity:.3f}] {title}...")
            print(f"     Year: {year} | PMID: {meta.get('pmid', 'N/A')}")


def large_scale_ingestion(target_articles: int = 100000):
    """
    Large-scale ingestion of PubMed data.
    Fetches from multiple biomedical topic areas to build a comprehensive corpus.

    Args:
        target_articles: Target number of articles to ingest
    """
    # Diverse biomedical queries to build a comprehensive corpus
    queries = [
        # Cancer research
        "cancer genomics",
        "cancer immunotherapy",
        "cancer drug resistance",
        "cancer biomarkers",
        "breast cancer treatment",
        "lung cancer therapy",
        "leukemia treatment",
        "melanoma immunotherapy",
        "pancreatic cancer",
        "prostate cancer genomics",
        # Genomics & Genetics
        "CRISPR gene editing",
        "gene expression profiling",
        "single cell RNA sequencing",
        "whole genome sequencing",
        "epigenetics cancer",
        "transcriptomics analysis",
        "proteomics biomarkers",
        "metabolomics disease",
        # Drug Discovery
        "drug discovery machine learning",
        "drug target identification",
        "pharmacogenomics",
        "clinical trials oncology",
        "precision medicine",
        "targeted therapy",
        # AI/ML in Biomedicine
        "machine learning cancer diagnosis",
        "deep learning medical imaging",
        "artificial intelligence drug discovery",
        "neural networks genomics",
        "computational biology",
        "bioinformatics analysis",
        # Specific diseases
        "Alzheimer disease genetics",
        "Parkinson disease treatment",
        "diabetes molecular mechanisms",
        "cardiovascular disease biomarkers",
        "autoimmune disease therapy",
        "infectious disease genomics",
        # Molecular Biology
        "protein structure prediction",
        "molecular pathway analysis",
        "cell signaling cancer",
        "apoptosis mechanisms",
        "tumor microenvironment",
        "stem cell therapy",
        # Clinical Research
        "clinical biomarkers",
        "disease prognosis prediction",
        "treatment response prediction",
        "survival analysis cancer",
        "patient stratification",
        "personalized medicine",
    ]

    articles_per_query = max(target_articles // len(queries), 500)
    total_indexed = 0

    print("=" * 70)
    print(f"LARGE-SCALE PUBMED INGESTION")
    print(f"Target: {target_articles:,} articles")
    print(f"Queries: {len(queries)}")
    print(f"Articles per query: {articles_per_query}")
    print("=" * 70)

    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Processing query: '{query}'")
        try:
            stats = ingest_pubmed_data(
                query=query,
                max_articles=articles_per_query,
                collection_name="pubmed_abstracts"
            )
            total_indexed = stats.get("total_documents", total_indexed)
            print(f"   Total indexed so far: {total_indexed:,}")

            # Rate limiting - be nice to NCBI servers
            time.sleep(0.5)

        except Exception as e:
            print(f"   Error with query '{query}': {e}")
            time.sleep(2)
            continue

    print("\n" + "=" * 70)
    print(f"LARGE-SCALE INGESTION COMPLETE")
    print(f"Total articles indexed: {total_indexed:,}")
    print("=" * 70)

    return total_indexed


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--large":
        # Large-scale ingestion mode
        target = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
        large_scale_ingestion(target_articles=target)
    else:
        # Default: smaller ingestion for testing
        stats = ingest_pubmed_data(
            query="cancer genomics machine learning",
            max_articles=100,
            collection_name="pubmed_abstracts"
        )
        print(f"\nIngestion stats: {stats}")

        # Demo search
        time.sleep(1)
        demo_search("pubmed_abstracts")
