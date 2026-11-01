# IR-Arena: Hybrid Search and Retrieval Evaluation Platform

IR-Arena benchmarks search strategies (lexical BM25, semantic kNN, Reciprocal Rank Fusion, and Weighted Score Fusion) on German IT job postings collected from the Bundesagentur für Arbeit API.

## Requirements

* Docker and Docker Compose
* Python 3.10 or newer
* uv package manager

## Quickstart

### 1. Start Elasticsearch and Kibana

Start the local Elasticsearch 8.15.0 and Kibana instances:

```bash
docker compose up -d
```

Verify that the cluster is healthy:

```bash
curl http://localhost:9200/_cluster/health
```

Elasticsearch listens on `http://localhost:9200` and Kibana on `http://localhost:5601`. Security and SSL are disabled for local development.

### 2. Install dependencies

Install the project and development dependencies using uv:

```bash
uv sync
```

### 3. Fetch vacancies

Collect job vacancies from the public Arbeitsagentur API:

```bash
uv run python scripts/fetch_jobs.py --limit 1500
```

This queries multiple German IT search profiles, filters non-target roles, fetches vacancy details concurrently, and writes normalized records to `data/jobs.jsonl`.

### 4. Generate embeddings and index documents

Index the vacancy dataset into Elasticsearch with dense vectors:

```bash
uv run python scripts/index_jobs.py
```

This uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions) to compute embeddings and indexes documents in batches into the `jobs` index.

### 5. Run the evaluation benchmark

Run the 50-query benchmark across all search strategies:

```bash
uv run python scripts/evaluate.py --k 10
```

The script evaluates each query against graded ground-truth judgments in `data/ground_truth.json` and outputs aggregate metrics.

### 6. Start the API server

Run the FastAPI application:

```bash
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

## Benchmark Results

Evaluation on 50 curated queries against 1,310 indexed job listings (k=10):

| Strategy | NDCG@10 | MRR | Recall@10 | Precision@10 | Mean Latency (ms) | p50 (ms) | p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| hybrid_rrf | 0.2407 | 0.5317 | 0.2167 | 0.2420 | 149.89 | 133.94 | 303.28 |
| hybrid_weighted | 0.2232 | 0.4893 | 0.2014 | 0.2260 | 115.43 | 87.16 | 207.94 |
| bm25 | 0.2141 | 0.4070 | 0.2018 | 0.2260 | 143.53 | 100.38 | 297.23 |
| knn | 0.1602 | 0.3588 | 0.1668 | 0.1860 | 52.99 | 37.87 | 78.65 |

Hybrid Reciprocal Rank Fusion achieves the highest ranking quality, improving MRR by 30.6% over BM25 and 48.2% over vector kNN.

## Search API Endpoints

All search endpoints accept POST requests with JSON payload containing `query` and optional filters (`city`, `is_remote`, `skills`):

* `POST /search/bm25`: BM25 lexical search using a custom German analyzer and title/skill field boosting.
* `POST /search/knn`: Vector search using cosine similarity on 384-dimensional dense embeddings.
* `POST /search/hybrid`: Concurrent BM25 and kNN retrieval merged with Reciprocal Rank Fusion (k=60).
* `GET /health`: Cluster health check returning Elasticsearch connectivity status.

Example request:

```bash
curl -X POST http://localhost:8000/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "Python FastAPI", "city": "Berlin", "limit": 10}'
```

## Project Layout

```
IR-Arena/
├── docker-compose.yml           # Elasticsearch 8.15.0 and Kibana
├── pyproject.toml               # Project metadata, dependencies, and tool settings
├── .env.example                 # Environment variable templates
├── data/
│   ├── jobs.jsonl               # Ingested and normalized vacancies
│   ├── ground_truth.json        # 50 benchmark queries with relevance judgments
│   └── evaluation_results.json  # Full metric breakdown per strategy and query
├── scripts/
│   ├── fetch_jobs.py            # CLI for Arbeitsagentur data collection
│   ├── index_jobs.py            # CLI for embedding and Elasticsearch bulk indexing
│   └── evaluate.py              # CLI for running IR evaluation benchmarks
├── src/
│   ├── main.py                  # FastAPI application entrypoint and lifespan
│   ├── clients/                 # Arbeitsagentur API client
│   ├── core/                    # Settings and Elasticsearch connection pool
│   ├── repositories/            # Elasticsearch repository and index mapping
│   ├── routers/                 # API route handlers and dependencies
│   ├── schemas/                 # Pydantic v2 domain schemas (jobs, search, evaluation)
│   └── services/
│       ├── embedder.py          # SentenceTransformer text embedding service
│       ├── search_service.py    # Search orchestration service
│       ├── evaluation/          # IRMetricsCalculator and SearchEvaluator
│       ├── fusion/              # ReciprocalRankFusion and WeightedScoreFusion
│       └── ingestion/           # Skill extraction, job filtering, and normalization
└── tests/                       # Unit and integration test suite
```

## Running Tests

Run the full test suite with pytest:

```bash
uv run pytest
```
