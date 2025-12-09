"""
Test script for the FastAPI backend.
Uses FastAPI's TestClient to test endpoints without running a server.
"""

from fastapi.testclient import TestClient
from api import app


def test_api():
    """Test all API endpoints."""
    print("=" * 50)
    print("Testing FastAPI Backend")
    print("=" * 50)

    with TestClient(app) as client:
        # 1. Health check
        print("\n1. Testing health endpoint...")
        response = client.get("/")
        assert response.status_code == 200
        print(f"   Response: {response.json()}")

        # 2. Get initial stats
        print("\n2. Testing stats endpoint...")
        response = client.get("/stats")
        assert response.status_code == 200
        stats = response.json()
        print(f"   Initial stats: {stats}")

        # 3. Add a single document
        print("\n3. Testing single document addition...")
        response = client.post("/documents", json={
            "text": "BRCA1 mutations are linked to hereditary breast cancer.",
            "metadata": {"topic": "genetics", "year": 2023}
        })
        assert response.status_code == 200
        print(f"   Response: {response.json()}")

        # 4. Add batch of documents
        print("\n4. Testing batch document addition...")
        documents = [
            {
                "text": "Checkpoint inhibitors have transformed cancer immunotherapy.",
                "metadata": {"topic": "treatment", "year": 2022}
            },
            {
                "text": "Single-cell sequencing reveals tumor heterogeneity.",
                "metadata": {"topic": "technology", "year": 2023}
            },
            {
                "text": "CAR-T cell therapy shows promise for blood cancers.",
                "metadata": {"topic": "treatment", "year": 2023}
            },
            {
                "text": "Machine learning predicts drug response from genomic features.",
                "metadata": {"topic": "technology", "year": 2022}
            }
        ]
        response = client.post("/documents/batch", json={"documents": documents})
        assert response.status_code == 200
        print(f"   Response: {response.json()}")

        # 5. Updated stats
        print("\n5. Checking updated stats...")
        response = client.get("/stats")
        stats = response.json()
        print(f"   Stats after adding documents: {stats}")

        # 6. Semantic search (POST)
        print("\n6. Testing semantic search (POST)...")
        response = client.post("/search", json={
            "query": "What treatments are available for cancer?",
            "n_results": 3
        })
        assert response.status_code == 200
        results = response.json()
        print(f"   Query: '{results['query']}'")
        print(f"   Found {results['total_results']} results:")
        for r in results["results"]:
            print(f"   [{r['similarity']:.4f}] {r['document'][:50]}...")

        # 7. Semantic search (GET)
        print("\n7. Testing semantic search (GET)...")
        response = client.get("/search", params={
            "q": "genetic mutations in cancer",
            "n": 2
        })
        assert response.status_code == 200
        results = response.json()
        print(f"   Query: '{results['query']}'")
        print(f"   Results:")
        for r in results["results"]:
            print(f"   [{r['similarity']:.4f}] {r['document'][:50]}...")

        # 8. Filtered search
        print("\n8. Testing filtered search...")
        response = client.post("/search", json={
            "query": "cancer research",
            "n_results": 5,
            "filter": {"topic": "treatment"}
        })
        assert response.status_code == 200
        results = response.json()
        print(f"   Query with filter (topic=treatment):")
        for r in results["results"]:
            print(f"   [{r['similarity']:.4f}] {r['document'][:50]}... | {r['metadata']}")

    print("\n" + "=" * 50)
    print("API tests PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    test_api()
