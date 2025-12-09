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

    def __init__(self, email: str = "user@example.com"):
        """
        Initialize the fetcher.

        Args:
            email: Email for NCBI API (required by their terms of service)
        """
        self.email = email

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
            "email": self.email
        }

        response = requests.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
        response.raise_for_status()

        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])

        return pmids

    def fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch full records for given PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of article records with title, abstract, metadata
        """
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email
        }

        response = requests.get(f"{self.BASE_URL}/efetch.fcgi", params=params)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.content)
        articles = []

        for article in root.findall(".//PubmedArticle"):
            record = self._parse_article(article)
            if record and record.get("abstract"):  # Only include articles with abstracts
                articles.append(record)

        return articles

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
            "pmid": article["pmid"],
            "title": article["title"],
            "year": article["year"],
            "journal": article["journal"],
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


if __name__ == "__main__":
    # Example: Ingest cancer genomics articles
    stats = ingest_pubmed_data(
        query="cancer genomics machine learning",
        max_articles=30,
        collection_name="pubmed_abstracts"
    )
    print(f"\nIngestion stats: {stats}")

    # Demo search
    time.sleep(1)  # Brief pause
    demo_search("pubmed_abstracts")
