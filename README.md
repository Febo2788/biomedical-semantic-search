# Biomedical Semantic Search System

A **multimodal semantic search system** for biomedical data that enables retrieval based on **meaning rather than metadata**. Scientists can search both literature AND experimental data using natural language queries.

Built to demonstrate AI/ML engineering skills for computational sciences roles in drug discovery and biomedical research.

## Features

- **Multimodal Search**: Query both papers AND experimental datasets
- **Semantic Search**: Find data by meaning using neural embeddings
- **Vector Database**: Efficient similarity search with ChromaDB
- **REST API**: Production-ready FastAPI backend
- **PubMed Integration**: Real biomedical literature
- **GEO Integration**: Gene expression experiments from NCBI
- **Retrieval Evaluation**: Standard IR metrics (Precision, Recall, MRR, NDCG)

## Technologies

| Component | Technology |
|-----------|------------|
| Embeddings | Sentence-Transformers, Hugging Face |
| Vector DB | ChromaDB |
| ML Framework | PyTorch |
| API | FastAPI, Uvicorn |
| Literature Data | PubMed E-utilities API |
| Experimental Data | NCBI GEO API |

## Project Structure

```
biomedical-semantic-search/
├── embeddings.py      # Hugging Face embedding pipeline
├── vector_store.py    # ChromaDB vector database
├── api.py             # FastAPI REST endpoints (multimodal)
├── ingest_pubmed.py   # PubMed literature ingestion
├── ingest_geo.py      # GEO experimental data ingestion
├── evaluation.py      # Retrieval quality metrics
├── main.py            # Main entry point
├── test_api.py        # API tests
├── requirements.txt   # Python dependencies
└── chroma_data/       # Persisted vector database
    ├── pubmed_abstracts/   # Literature embeddings
    └── geo_experiments/    # Experiment embeddings
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (use Python 3.12 x64)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers chromadb fastapi uvicorn requests
```

### 2. Ingest Data

```bash
# Small-scale ingestion for testing
python ingest_pubmed.py
python ingest_geo.py

# Large-scale ingestion (builds full dataset)
python ingest_pubmed.py --large 100000
python ingest_geo.py --large 50000
```

The large-scale ingestion fetches from 48+ diverse biomedical queries to build a comprehensive corpus.

### 3. Run Demo

```bash
python main.py demo
```

### 4. Start API Server

```bash
python main.py serve
```

API documentation: http://localhost:8000/docs

## API Endpoints

### Literature Search
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/search` | Semantic search on papers |
| GET | `/search?q=` | Simple paper search |
| GET | `/stats` | Paper index statistics |

### Experimental Data Search
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/search/experiments` | Search GEO experiments |
| GET | `/search/experiments?q=` | Simple experiment search |
| GET | `/stats/experiments` | Experiment index statistics |

### Document Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents` | Add single document |
| POST | `/documents/batch` | Add multiple documents |
| GET | `/` | Health check |

### Example Search Request

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What genes are linked to breast cancer?", "n_results": 5}'
```

## Dataset Statistics

| Data Source | Documents Indexed |
|-------------|-------------------|
| PubMed Abstracts | 64,753 |
| GEO Experiments | 10,462 |
| **Total** | **75,215** |

## Evaluation Results

Evaluated using synthetic relevance judgments (keyword-based pseudo-labels):

| Metric | K=5 | Description |
|--------|-----|-------------|
| Precision@K | 1.00 | All top-5 results contain target keywords |
| Recall@K | 0.0007 | 5 retrieved out of ~7,000 matching docs |
| MRR | 1.00 | First result is always relevant |
| NDCG@K | 1.00 | Perfect ranking within top-5 |

*Note: High precision is expected with keyword-based relevance on a large corpus. For production use, human relevance judgments would provide more meaningful evaluation.*

## Key Concepts Demonstrated

### 1. Embedding Pipeline (`embeddings.py`)
- Load pre-trained transformer models from Hugging Face
- Generate dense vector representations of text
- Batch processing for efficiency
- L2 normalization for cosine similarity

### 2. Vector Database (`vector_store.py`)
- Store embeddings with ChromaDB
- HNSW index for fast approximate nearest neighbor search
- Metadata filtering (by year, topic, etc.)
- Persistent storage

### 3. Semantic Search (`api.py`)
- Query encoding → vector similarity search
- Distance to similarity conversion
- Filtered search by metadata
- RESTful API design

### 4. Retrieval Evaluation (`evaluation.py`)
- Precision@K: Fraction of retrieved docs that are relevant
- Recall@K: Fraction of relevant docs retrieved
- MRR: Mean Reciprocal Rank
- NDCG: Normalized Discounted Cumulative Gain


## Author

Felix Borrego
MS Biostatistics, UMass Amherst
[GitHub](https://github.com/Febo2788)

## License

MIT License - Free for educational and research use.
