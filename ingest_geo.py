"""
GEO (Gene Expression Omnibus) data ingestion script.
Fetches experimental sample metadata for semantic search.
Demonstrates: Multimodal data integration, experimental data search
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import re
import time

from embeddings import EmbeddingPipeline
from vector_store import VectorStore


class GEOFetcher:
    """
    Fetches sample metadata from NCBI GEO database.
    Documentation: https://www.ncbi.nlm.nih.gov/geo/info/geo_paccess.html
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: str = "felix.borrego02@gmail.com", api_key: str = "745f2b98fc7384ec80b6c74e4636087a2508"):
        self.email = email
        self.api_key = api_key

    def search_datasets(self, query: str, max_results: int = 50) -> List[str]:
        """
        Search GEO DataSets and return GSE IDs.

        Args:
            query: Search query (e.g., "breast cancer RNA-seq")
            max_results: Maximum datasets to return

        Returns:
            List of GSE accession IDs
        """
        params = {
            "db": "gds",
            "term": f"{query} AND gse[Entry Type]",
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
            "api_key": self.api_key
        }

        response = requests.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
        response.raise_for_status()

        data = response.json()
        gds_ids = data.get("esearchresult", {}).get("idlist", [])

        return gds_ids

    def fetch_dataset_info(self, gds_ids: List[str], batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch detailed info for GEO datasets with batching.

        Args:
            gds_ids: List of GDS IDs from search
            batch_size: Number of IDs to fetch per request (max ~100 to avoid URL length issues)

        Returns:
            List of dataset records
        """
        if not gds_ids:
            return []

        all_datasets = []

        # Process in batches to avoid 414 URI Too Long errors
        for i in range(0, len(gds_ids), batch_size):
            batch = gds_ids[i:i + batch_size]

            params = {
                "db": "gds",
                "id": ",".join(batch),
                "retmode": "xml",
                "email": self.email,
                "api_key": self.api_key
            }

            try:
                response = requests.get(f"{self.BASE_URL}/esummary.fcgi", params=params)
                response.raise_for_status()

                root = ET.fromstring(response.content)

                for doc in root.findall(".//DocSum"):
                    record = self._parse_docsum(doc)
                    if record:
                        all_datasets.append(record)

                # Rate limiting (10 req/sec with API key)
                if i + batch_size < len(gds_ids):
                    time.sleep(0.1)

            except Exception as e:
                print(f"   Error fetching batch {i//batch_size + 1}: {e}")
                time.sleep(1)
                continue

        return all_datasets

    def _parse_docsum(self, doc: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse a DocSum element from GEO."""
        try:
            record = {"id": None, "accession": None, "title": None,
                     "summary": None, "organism": None, "samples": None,
                     "platform": None, "experiment_type": None}

            # Get ID
            id_elem = doc.find("Id")
            if id_elem is not None:
                record["id"] = id_elem.text

            # Parse Items
            for item in doc.findall(".//Item"):
                name = item.get("Name", "")
                text = item.text or ""

                if name == "Accession":
                    record["accession"] = text
                elif name == "title":
                    record["title"] = text
                elif name == "summary":
                    record["summary"] = text
                elif name == "taxon":
                    record["organism"] = text
                elif name == "n_samples":
                    record["samples"] = text
                elif name == "GPL":
                    record["platform"] = text
                elif name == "gdsType":
                    record["experiment_type"] = text

            return record if record["accession"] else None

        except Exception as e:
            print(f"Error parsing DocSum: {e}")
            return None


def create_sample_descriptions(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create searchable descriptions from GEO dataset metadata.

    In a real system, you'd fetch individual sample (GSM) metadata.
    Here we create rich descriptions from dataset-level info.
    """
    samples = []

    for ds in datasets:
        if not ds.get("title") or not ds.get("summary"):
            continue

        # Create a rich text description for embedding
        description_parts = []

        if ds.get("title"):
            description_parts.append(ds["title"])

        if ds.get("summary"):
            # Clean and truncate summary
            summary = ds["summary"][:500]
            description_parts.append(summary)

        if ds.get("organism"):
            description_parts.append(f"Organism: {ds['organism']}")

        if ds.get("experiment_type"):
            description_parts.append(f"Type: {ds['experiment_type']}")

        description = ". ".join(description_parts)

        samples.append({
            "accession": ds["accession"],
            "title": ds["title"],
            "description": description,
            "organism": ds.get("organism", ""),
            "experiment_type": ds.get("experiment_type", ""),
            "n_samples": ds.get("samples", ""),
            "platform": ds.get("platform", "")
        })

    return samples


def ingest_geo_data(
    query: str,
    max_datasets: int = 50,
    collection_name: str = "geo_experiments"
) -> Dict[str, Any]:
    """
    Ingest GEO experimental data into the vector store.

    Args:
        query: GEO search query
        max_datasets: Maximum datasets to fetch
        collection_name: ChromaDB collection name

    Returns:
        Ingestion statistics
    """
    print(f"Starting GEO ingestion for query: '{query}'")
    print(f"Max datasets: {max_datasets}")
    print("-" * 50)

    # Initialize components
    fetcher = GEOFetcher()
    embedder = EmbeddingPipeline("all-MiniLM-L6-v2")
    store = VectorStore(
        collection_name=collection_name,
        persist_directory="./chroma_data"
    )

    # Search GEO
    print("\n1. Searching GEO DataSets...")
    gds_ids = fetcher.search_datasets(query, max_results=max_datasets)
    print(f"   Found {len(gds_ids)} datasets")

    if not gds_ids:
        return {"status": "no_results", "datasets_found": 0}

    # Fetch dataset details
    print("\n2. Fetching dataset metadata...")
    datasets = fetcher.fetch_dataset_info(gds_ids)
    print(f"   Retrieved {len(datasets)} dataset records")

    # Create sample descriptions
    print("\n3. Creating searchable descriptions...")
    samples = create_sample_descriptions(datasets)
    print(f"   Created {len(samples)} sample descriptions")

    if not samples:
        return {"status": "no_samples", "datasets_found": len(gds_ids)}

    # Prepare for indexing
    documents = [s["description"] for s in samples]
    metadatas = [{
        "accession": s["accession"] or "",
        "title": (s["title"][:200] if s["title"] else ""),
        "organism": s["organism"] or "",
        "experiment_type": s["experiment_type"] or "",
        "n_samples": s["n_samples"] or ""
    } for s in samples]
    ids = [f"geo_{s['accession']}" for s in samples]

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
        "datasets_found": len(gds_ids),
        "datasets_indexed": len(samples),
        "collection_name": collection_name,
        "total_documents": store.count()
    }

    print("\n" + "=" * 50)
    print("GEO Ingestion Complete!")
    print(f"  Query: {query}")
    print(f"  Datasets indexed: {len(samples)}")
    print(f"  Total in collection: {store.count()}")
    print("=" * 50)

    return stats


def demo_experiment_search(collection_name: str = "geo_experiments"):
    """Demo semantic search on GEO experimental data."""
    print("\n" + "=" * 60)
    print("Demo: Semantic Search on Gene Expression Experiments")
    print("=" * 60)

    embedder = EmbeddingPipeline("all-MiniLM-L6-v2")
    store = VectorStore(
        collection_name=collection_name,
        persist_directory="./chroma_data"
    )

    if store.count() == 0:
        print("No experiments in collection. Run ingestion first.")
        return

    print(f"\nExperiments indexed: {store.count()}")

    # Example queries a scientist might ask
    queries = [
        "breast cancer drug resistance RNA sequencing",
        "CRISPR knockout screen in cancer cells",
        "immunotherapy response gene expression",
        "tumor microenvironment single cell analysis"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print("-" * 60)

        query_embedding = embedder.encode(query, show_progress=False).tolist()[0]
        results = store.search(query_embedding, n_results=3)

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"],
            results["metadatas"],
            results["distances"]
        )):
            similarity = 1 - dist
            accession = meta.get("accession", "N/A")
            title = meta.get("title", "N/A")[:60]
            organism = meta.get("organism", "N/A")
            exp_type = meta.get("experiment_type", "N/A")

            # Clean title for display (handle unicode)
            clean_title = title.encode('ascii', 'ignore').decode('ascii')
            print(f"\n  {i+1}. [{similarity:.3f}] {accession}")
            print(f"     Title: {clean_title}...")
            print(f"     Organism: {organism} | Type: {exp_type}")


def large_scale_geo_ingestion(target_datasets: int = 50000):
    """
    Large-scale ingestion of GEO experimental data.
    Fetches from multiple research areas to build a comprehensive experiment database.

    Args:
        target_datasets: Target number of datasets to ingest
    """
    # Diverse experimental queries
    queries = [
        # Cancer types
        "breast cancer RNA-seq",
        "lung cancer gene expression",
        "leukemia transcriptome",
        "melanoma expression profiling",
        "pancreatic cancer microarray",
        "prostate cancer genomics",
        "colorectal cancer RNA-seq",
        "ovarian cancer expression",
        "glioblastoma transcriptome",
        "lymphoma gene expression",
        # Technology types
        "single cell RNA-seq",
        "bulk RNA sequencing",
        "microarray expression",
        "ChIP-seq histone",
        "ATAC-seq chromatin",
        "methylation array",
        "proteomics mass spectrometry",
        "metabolomics",
        # Research areas
        "drug resistance cancer",
        "immunotherapy response",
        "tumor microenvironment",
        "metastasis gene expression",
        "cancer stem cells",
        "drug treatment response",
        "CRISPR screen",
        "knockout gene expression",
        # Model systems
        "patient derived xenograft",
        "cell line expression",
        "organoid RNA-seq",
        "mouse tumor model",
        "human primary tumor",
        # Specific pathways/genes
        "BRCA1 BRCA2 expression",
        "TP53 mutation expression",
        "KRAS signaling",
        "MYC amplification",
        "immune checkpoint",
        "PD-1 PD-L1",
        # Clinical
        "treatment naive cancer",
        "chemotherapy resistance",
        "radiation response",
        "targeted therapy response",
        "clinical trial expression",
        # Other diseases
        "Alzheimer disease brain",
        "Parkinson disease expression",
        "diabetes gene expression",
        "cardiovascular disease",
        "autoimmune disease expression",
        "infectious disease response",
    ]

    datasets_per_query = max(target_datasets // len(queries), 200)
    total_indexed = 0

    print("=" * 70)
    print(f"LARGE-SCALE GEO INGESTION")
    print(f"Target: {target_datasets:,} datasets")
    print(f"Queries: {len(queries)}")
    print(f"Datasets per query: {datasets_per_query}")
    print("=" * 70)

    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Processing query: '{query}'")
        try:
            stats = ingest_geo_data(
                query=query,
                max_datasets=datasets_per_query,
                collection_name="geo_experiments"
            )
            total_indexed = stats.get("total_documents", total_indexed)
            print(f"   Total indexed so far: {total_indexed:,}")

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"   Error with query '{query}': {e}")
            time.sleep(2)
            continue

    print("\n" + "=" * 70)
    print(f"LARGE-SCALE GEO INGESTION COMPLETE")
    print(f"Total datasets indexed: {total_indexed:,}")
    print("=" * 70)

    return total_indexed


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--large":
        # Large-scale ingestion mode
        target = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
        large_scale_geo_ingestion(target_datasets=target)
    else:
        # Default: smaller ingestion for testing
        stats = ingest_geo_data(
            query="cancer gene expression RNA-seq",
            max_datasets=100,
            collection_name="geo_experiments"
        )
        print(f"\nIngestion stats: {stats}")

        # Demo search
        time.sleep(1)
        demo_experiment_search("geo_experiments")
