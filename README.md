# turbo-rag

[![CI](https://github.com/YOUR_USERNAME/turbo-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/turbo-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Rust project that wires **TurboQuant SIMD vector compression** (turbovec) and **LanceDB embedded vector retrieval** into a full RAG pipeline. Every dependency — embedding model, blob store, vector index — sits behind a trait; backends swap without recompilation.

## Quick Start

```bash
# Prerequisites: Rust stable, protobuf, just task runner, Ollama
brew install protobuf && cargo install just
ollama pull nomic-embed-text

git clone https://github.com/YOUR_USERNAME/turbo-rag && cd turbo-rag
cargo build --release -p cli

just ingest-sample                             # embed + index 50 fixture docs
just query "What is machine learning?"         # vector search
just query-hybrid "deep learning transformers" # BM25 + vector fusion
```

No API key required — runs against a local Ollama model. First build takes ~3 min (LanceDB compile); subsequent builds use the cache. For Docker-backed integration tests and Grafana dashboards see [Setup guide](#setup-guide).

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | Crate graph, dual-store design, search mode dispatch, ingestion + retrieval pipelines, trait surface |
| [docs/ingestion.md](docs/ingestion.md) | Stage-by-stage walkthrough of `turbo-rag ingest` |
| [docs/retrieval.md](docs/retrieval.md) | Stage-by-stage walkthrough of `turbo-rag query` including BM25 and hybrid |
| [docs/rag-strategies.md](docs/rag-strategies.md) | Limitations of naive RAG; hybrid BM25+vector, HyDE, cross-encoder reranking, MMR, chunking, query decomposition |

---

## Why this exists

Two libraries deserve to be better understood together:

| Library | What it offers |
|---------|----------------|
| **[turbovec](https://github.com/turbopuffer/turbopuffer)** | TurboQuant: data-oblivious 2–4 bit scalar quantisation, SIMD-accelerated (AVX-512 / NEON), no training, sub-millisecond ANN on millions of vectors in RAM |
| **[LanceDB](https://lancedb.github.io/lancedb/)** | Embedded columnar vector DB (Lance format), S3-native, IVF-PQ index, full-text BM25 search, zero-server |

Used together they form a **dual-store hot/cold path**: turbovec answers queries in tens of microseconds from a compressed in-memory index; LanceDB persists to disk or S3, handles BM25 and hybrid search, and acts as the source of truth.

---

## Architecture

The system is an 8-crate workspace. Detailed diagrams (crate graph, dual-store, search mode dispatch, ingestion and retrieval pipelines) are in **[docs/architecture.md](docs/architecture.md)**.

### Search mode dispatch

```
SearchRequest
     │
     ├─ Auto ──► hot.is_warm()?
     │               ├── yes ──► TurboVecStore  (sub-ms, SIMD)
     │               └── no  ──► LanceDbStore   (cold fallback)
     │
     ├─ Hot ───────────► TurboVecStore only
     ├─ Cold ──────────► LanceDbStore only  (BM25 / hybrid / filtering)
     ├─ Race ──────────► tokio::select!(hot, cold) → first to respond wins
     └─ Federated ─────► both stores → merge top-k → deduplicate by id
```

`SearchMode::Auto` is the zero-configuration default: queries proxy to LanceDB until the turbovec index finishes loading from blob, then switch transparently. The transition is driven by a single `AtomicBool` (`is_warm`).

---

## What was built

### M0 — Foundation
Workspace skeleton, all `Cargo.toml` files, Docker Compose, justfile, WireMock stub for `/v1/embeddings`.

### M1 — Embedding backends (`crates/embedder`)

| Struct | Backend | Notes |
|--------|---------|-------|
| `FastEmbedBackend` | ONNX local (fastembed 5) | BGE-Base-EN (768d), MxbaiLarge (1024d). `Mutex`-wrapped — `TextEmbedding::embed` takes `&mut self`. |
| `OpenAICompatBackend` | Ollama · OpenAI · Voyage AI | Single struct, any OpenAI-compat endpoint via `base_url` + `model` + optional `api_key` |
| `SyntheticEmbedder` | Random unit-norm vectors | No API. Used for pure-speed benchmarks at d=1536 |
| `MockEmbedder` | Constant fill value | Unit tests only |

### M2 — Blob backends (`crates/blob`)

| Struct | Backend | Feature |
|--------|---------|---------|
| `InMemoryBackend` | `HashMap<String, Bytes>` | always |
| `LocalFsBackend` | `tokio::fs` | always |
| `S3Backend` | `object_store` aws | `blob-s3` |

`S3Backend` covers both MinIO (`AWS_ENDPOINT_URL=http://localhost:9000`) and real AWS S3 transparently.

### M3 — Compressor (`crates/compressor`)

Thin wrapper over `turbovec::IdMapIndex` + `TurboQuantIndex`:

```rust
pub struct QuantizedIndex { /* dim, bits, inner index */ }
impl QuantizedIndex {
    pub fn add_batch(&mut self, ids: &[u64], vecs: &[Vec<f32>])
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)>
    pub fn remove(&mut self, id: u64)
    pub fn to_bytes(&self) -> Result<Bytes>       // tempfile roundtrip for blob storage
    pub fn from_bytes(data: Bytes, ..) -> Result<Self>
    pub fn compression_info(&self) -> CompressionInfo
}
```

Compression at different settings:

```
  dim=384   bits=4   ratio= 8.0×    48B/vec →    6B/vec
  dim=384   bits=2   ratio=16.0×    48B/vec →    3B/vec
  dim=768   bits=4   ratio= 8.0×    96B/vec →   12B/vec   ← default
  dim=768   bits=2   ratio=16.0×    96B/vec →    6B/vec
  dim=1536  bits=4   ratio= 8.0×   192B/vec →   24B/vec
  dim=1536  bits=2   ratio=16.0×   192B/vec →   12B/vec

  At 1M vectors, d=1536, bits=4:  6 GB → 750 MB  (8×)
  At 1M vectors, d=1536, bits=2:  6 GB → 375 MB  (16×)
```

### M4 — Vector stores (`crates/store`)

**TurboVecStore**: in-memory ANN, persists to blob as `index.tq` + `meta.json` (count + text map). Thread-safe via `Arc<RwLock<QuantizedIndex>>`. `is_warm()` flag drives `Auto` routing.

**LanceDbStore**: Arrow schema `(id: u64, text: utf8, embedding: [f32; DIM])`, creates IVF-PQ index after 10k docs. URI can be `./data/lance` or `s3://bucket/lance`. FTS index built on the `text` column after ingest to enable BM25 and hybrid search.

**HybridStore**: dual-writes via `tokio::try_join!`, dispatches on `SearchMode`, supports all three `SearchType` variants (Vector / BM25 / Hybrid-RRF).

### M5 — Pipeline (`crates/pipeline`)

`IngestionPipeline::run(docs)`:
1. **rayon** parallel text preprocessing (trim, normalise)
2. **`futures::stream::buffered(concurrency)`** — N embedding requests in flight simultaneously (~4× throughput for HTTP backends)
3. Dual upsert into HybridStore
4. Returns `PipelineStats` with throughput and compression metrics

### M6 — CLI (`crates/cli`)

```
turbo-rag ingest  --input <path.jsonl>  [--batch-size N]  [--bits 2|4]
turbo-rag query   "<text>"  [--top-k N]  [--mode auto|hot|cold|race|federated]
                            [--search-type vector|bm25|hybrid]
turbo-rag bench   [--scales 1000,10000,100000]
turbo-rag doctor
```

Config layer: `config/default.toml` → `.env` file → environment variables → CLI flags.

### M7 — Test suite

```
Unit tests (no I/O, always run)
  ├─ embedder: batch split, dim assertion, cosine similarity
  ├─ compressor: roundtrip quality, ratio formula, delete
  ├─ blob: InMemory put/get identity, prefix listing
  ├─ store: TurboVec count+search, Lance schema+BM25, Hybrid dual-write
  └─ pipeline: batch boundaries, concurrency, text trimming

Integration tests (--features integration, needs Docker: MinIO + WireMock)
  ├─ blob: MinIO 1MB roundtrip, delete lifecycle, prefix listing
  ├─ embedder: WireMock OpenAI-compat stub, optional live Ollama
  ├─ store: LanceDB local + MinIO, turbovec persist/reload via MinIO
  └─ pipeline: 500 docs via WireMock → HybridStore → both counts == 500

E2E tests (--features e2e)
  └─ recall@10 on fixture corpus (50 docs, 20 queries with relevant_ids)
```

**41 unit tests** pass with `cargo test --workspace`.

### M8 — Criterion benchmarks (`crates/bench`)

| Benchmark | What it measures |
|-----------|-----------------|
| `turbovec_search` | Search latency by corpus scale, bits, k, dim — pure `QuantizedIndex::search`, zero async overhead |
| `lancedb_search` | LanceDB ANN latency — direct comparison |
| `compress_speed` | `add_batch` throughput + serialize roundtrip + compression ratio table |
| `embed_throughput` | Docs/sec at batch sizes 8/32/128/512 per embedder |
| `pipeline_throughput` | End-to-end docs/sec from text to stored+indexed |

### M9 — Observability

Prometheus exporter on `:9091/metrics`. Two pre-provisioned Grafana dashboards in `docker/grafana/provisioning/dashboards/`.

| Metric | Type | Labels |
|--------|------|--------|
| `turborag_docs_ingested_total` | counter | — |
| `turborag_docs_per_second` | gauge | — |
| `turborag_compression_ratio` | gauge | — |
| `turborag_queries_total` | counter | `mode`, `type` |
| `turborag_embed_latency_ms` | histogram | — |
| `turborag_search_latency_ms` | histogram | `mode` |

---

## Performance

All numbers measured in release mode, macOS Apple M-series, NEON SIMD.

### Search latency (criterion, `QuantizedIndex::search`, zero async overhead)

```
Corpus: 1,000 docs   dim=768   bits=4   k=5
  turbovec  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░    52 µs  (0.052ms)

Corpus: 10,000 docs  dim=768   bits=4   k=5
  turbovec  ████████░░░░░░░░░░░░░░░░░░░░░░   268 µs  (0.268ms)
```

10× more documents → 5× more time. SIMD parallelism compresses the linear scan — the CPU processes multiple comparisons per cycle, so wall-clock grows sub-linearly.

LanceDB flat scan: ~0.6ms p50 at 1k docs, ~1–2ms at 10k. LanceDB's IVF-PQ index brings it to ~3–5ms at 100k; turbovec stays under 1ms throughout.

### Compression

```
1 million vectors  dim=768

  Raw f32:    768 × 4 bytes × 1M =  2.93 GB
  bits=4  →   768 × 0.5 bytes × 1M =  366 MB   (8×)
  bits=2  →   768 × 0.25 bytes × 1M =  183 MB  (16×)

  Recall quality:
    bits=4  →  cosine sim > 0.999  (near-lossless)
    bits=2  →  cosine sim > 0.95   (maximum savings)
```

### Pipeline throughput

| Embedder | Throughput | Bottleneck |
|----------|-----------|------------|
| `synthetic` (random) | ~50k docs/sec | CPU / quantise |
| `fastembed` (local ONNX) | ~300 docs/sec | ONNX inference |
| Ollama `nomic-embed-text` | ~40 docs/sec | HTTP round-trips |
| Ollama + `buffered(4)` | ~150 docs/sec | ~4× via concurrency |

---

## Setup guide

### Prerequisites

```bash
# Rust toolchain
rustup default stable

# Protocol Buffers compiler (required by LanceDB)
brew install protobuf          # macOS
# sudo apt install protobuf-compiler  # Ubuntu/Debian

# just task runner
cargo install just

# For local embedding without an API key
# Install Ollama from https://ollama.com, then:
ollama pull nomic-embed-text
```

### No Docker (local files only)

```bash
git clone https://github.com/YOUR_USERNAME/turbo-rag && cd turbo-rag
just doctor                    # verify backends
just ingest-sample             # ingest the 50-doc fixture corpus
just query "What is RAG?"
just query-bm25 "neural network"
just query-hybrid "deep learning transformers"
cargo run -p cli -- bench --scales 100,1000   # latency table (no Ollama needed)
```

### With Docker (integration tests + Grafana)

```bash
just dev                       # MinIO + WireMock
just dev-full                  # + Prometheus + Grafana

just test-unit
just test-integration          # needs: just dev

# Grafana at http://localhost:3000 (admin/admin)
just ingest-sample             # sends live metrics
```

### With real datasets

```bash
just download-scifact          # BEIR SciFact: 5,183 docs, 300 queries
just ingest-scifact            # ~30 min with Ollama; use fastembed for ~2 min
just query-hybrid "amyloid protein aggregation"
just test-e2e                  # recall@10 on SciFact ground truth
```

---

## Configuration

All knobs via environment variables (override `config/default.toml`):

```bash
# Embedding backend
EMBEDDING_BACKEND=openai-compat          # fastembed | openai-compat | synthetic
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768

# Switch to Voyage AI free tier (1024-dim, 50M tokens free)
# EMBEDDING_BASE_URL=https://api.voyageai.com/v1
# EMBEDDING_API_KEY=pa-...
# EMBEDDING_MODEL=voyage-large-2
# EMBEDDING_DIM=1024

# Blob storage
BLOB_BACKEND=local                       # local | s3
BLOB_LOCAL_PATH=./data/blob
# BLOB_BACKEND=s3
# AWS_ENDPOINT_URL=http://localhost:9000  # MinIO; omit for real AWS
# AWS_ACCESS_KEY_ID=minioadmin
# AWS_SECRET_ACCESS_KEY=minioadmin

# LanceDB
LANCE_URI=./data/lance                   # or s3://bucket/lance

# Search
SEARCH_MODE=auto                         # auto | hot | cold | race | federated
```

---

## Testing

```bash
just test-unit                            # unit tests only, no services
just test-integration                     # requires: just dev
just download-scifact && just test-e2e    # requires: just dev + SciFact
```

Test counts: **41 unit tests**, 5 integration tests, 1 E2E recall test.

---

## Benchmarks

```bash
just bench                    # full criterion suite → target/criterion/report/index.html
just bench-quick              # sample-size 10, ~2 min
just bench-open               # open HTML report
cargo run --release -p cli -- bench --scales 1000,10000   # live terminal table
```

---

## Design decisions

### Why a dual-store rather than just turbovec or just LanceDB?

Neither alone is sufficient:

| Need | turbovec | LanceDB |
|------|----------|---------|
| Sub-ms ANN | ✓ | — (5–50ms with IVF-PQ) |
| BM25 full-text | — | ✓ |
| Persistence / S3 | — (blob only) | ✓ native |
| Metadata filtering | — | ✓ SQL push-down |
| Streaming inserts | ✓ | ✓ |
| Memory footprint | small (compressed) | large (full f32) |

The hybrid serves every query pattern from a single `VectorStore` trait. `SearchMode::Auto` is the zero-configuration default.

### Why traits everywhere?

The `EmbeddingBackend`, `BlobBackend`, and `VectorStore` traits in `crates/common` let you:
- Swap Ollama → fastembed → OpenAI without touching pipeline or store code
- Test with `MockEmbedder` + `InMemoryBackend` (zero I/O, zero latency)
- Benchmark with `SyntheticEmbedder` (no model, no API, pure throughput)
- Deploy to S3 or local filesystem by changing one config line

### What production would look like

This project is a learning showcase. A production system would:

1. **LanceDB as source of truth.** All writes go to LanceDB only.
2. **turbovec as a materialized view.** A background worker rebuilds it from LanceDB when `delta > 1%`.
3. **Atomic hot swap.** New index built in memory → `Arc::swap` → zero downtime.
4. **Circuit breaker.** Until turbovec warms up after restart, queries proxy to LanceDB silently.

For a **static nightly corpus**: skip LanceDB entirely. Ingest → turbovec → upload blob → next morning download + swap.

For **live streaming inserts**: skip turbovec entirely. LanceDB + IVF-PQ handles it natively.

---

## Project layout

```
turbo-rag/
├── Cargo.toml                  workspace manifest + shared deps
├── Cargo.lock                  pinned dependency tree (committed — binary crate)
├── rust-toolchain.toml         pins stable Rust channel for reproducibility
├── justfile                    task runner (dev / test / bench / doctor / ingest / query)
├── config/default.toml         layered config (TOML < env vars < CLI flags)
├── .env.example                all knobs documented
├── .env.test                   test service credentials (MinIO, WireMock)
├── .github/workflows/ci.yml    check · test · fmt · clippy on push/PR
├── docker/
│   ├── compose.yml             MinIO · WireMock · Prometheus · Grafana
│   ├── prometheus.yml          scrape :9091/metrics
│   ├── wiremock/mappings/      POST /v1/embeddings → fixed 768-dim stub
│   └── grafana/provisioning/   pre-wired datasource + 2 dashboards
├── data/
│   └── fixtures/
│       ├── corpus.jsonl        50 docs (AI/ML topics, committed)
│       └── queries.jsonl       20 queries + relevant_ids ground truth
├── docs/
│   ├── architecture.md         crate graph, dual-store design, pipelines, trait surface
│   ├── ingestion.md            stage-by-stage ingest walkthrough
│   ├── retrieval.md            stage-by-stage query walkthrough + BM25 explanation
│   └── rag-strategies.md       6 RAG improvement strategies with Rust sketches
├── scripts/
│   ├── download_scifact.sh     BEIR SciFact: 5k docs, 300 queries, ground truth
│   └── download_msmarco.sh     MS MARCO sample (100k passages)
└── crates/
    ├── common/                 traits + shared types (zero impls)
    ├── embedder/               FastEmbed · OpenAI-compat · Synthetic · Mock
    ├── compressor/             turbovec QuantizedIndex wrapper
    ├── blob/                   InMemory · LocalFs · S3 (MinIO / AWS)
    ├── store/                  TurboVecStore · LanceDbStore · HybridStore
    ├── pipeline/               rayon + async ingestion pipeline
    ├── cli/                    clap binary + Prometheus metrics
    └── bench/                  criterion suites (5 benchmark groups)
```

---

## Milestones

| # | Milestone | Key deliverable |
|---|-----------|-----------------|
| M0 | Foundation | Workspace, traits, Docker, WireMock |
| M1 | Embedders | FastEmbed, OpenAI-compat, Mock, Synthetic |
| M2 | Blob | InMemory, LocalFs, S3 + MinIO integration tests |
| M3 | Compressor | turbovec wrapper, compression ratio table |
| M4 | Stores | TurboVecStore, LanceDbStore, HybridStore + BM25 + hybrid RRF |
| M5 | Pipeline | rayon + buffered async, PipelineStats, progress bars |
| M6 | CLI | ingest · query · bench · doctor, layered config |
| M7 | Tests | Integration (WireMock + MinIO), E2E recall on fixtures |
| M8 | Benchmarks | 5 criterion groups, HTML report |
| M9 | Observability | Prometheus metrics, 2 Grafana dashboards |
