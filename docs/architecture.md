# Architecture

## Crate graph

```
                        ┌─────────┐
                        │ common  │  traits + shared types only
                        └────┬────┘
           ┌─────────────────┼─────────────────────┐
           ▼                 ▼                     ▼
      ┌──────────┐    ┌────────────┐        ┌──────────┐
      │ embedder │    │ compressor │        │   blob   │
      └────┬─────┘    └─────┬──────┘        └────┬─────┘
           │                │                    │
           └────────────────┼────────────────────┘
                            ▼
                       ┌─────────┐
                       │  store  │  TurboVecStore · LanceDbStore · HybridStore
                       └────┬────┘
                            │
                       ┌────┴────┐
                       │pipeline │  rayon preprocessing + buffered async embed
                       └────┬────┘
                            │
                       ┌────┴────┐
                       │   cli   │  ingest · query · bench · doctor
                       └─────────┘
                            │
                       ┌────┴────┐
                       │  bench  │  criterion suites (separate binary, same workspace)
                       └─────────┘
```

`common` has zero external runtime deps — only `serde`, `async-trait`, `thiserror`. Every impl crate depends on `common`; nothing depends upward.

---

## Dual-store design

Every ingest goes to both stores simultaneously via `tokio::try_join!`. Every query is routed based on `SearchMode`.

```
                         ┌─────────────────────────────────────────────┐
                         │               HybridStore                   │
                         │                                             │
  EmbeddedDoc[] ─────────┼──► tokio::try_join!(hot.upsert, cold.upsert)│
                         │                   │              │          │
                         │                   ▼              ▼          │
                         │          TurboVecStore    LanceDbStore      │
                         │          ┌───────────┐   ┌──────────────┐  │
                         │          │QuantizedIdx│   │ Arrow schema │  │
                         │          │ 4-bit SIMD │   │  IVF-PQ idx  │  │
                         │          │  ~0.08ms   │   │  ~0.61ms     │  │
                         │          │  in-memory │   │  disk / S3   │  │
                         │          └─────┬──────┘   └──────┬───────┘  │
                         │                └──────┬───────────┘          │
                         │           ┌───────────┴──────────┐           │
                         │           │  SearchMode dispatch  │           │
                         │           └──────────┬───────────┘           │
                         └──────────────────────┼────────────────────── ┘
                                                ▼
                                         ScoredDoc[]
```

---

## Search mode dispatch

```
SearchRequest
     │
     ├─ Auto ──► hot.is_warm()?
     │               ├── yes ──► TurboVecStore  (sub-ms, SIMD)
     │               └── no  ──► LanceDbStore   (cold fallback)
     │
     ├─ Hot ───────────► TurboVecStore only
     │
     ├─ Cold ──────────► LanceDbStore only  (BM25 / hybrid / filtering)
     │
     ├─ Race ──────────► tokio::select!(hot, cold) → first to respond wins
     │                   (turbovec wins >97% of the time once warm)
     │
     └─ Federated ─────► both stores → merge top-k → deduplicate by id
```

| Mode | Store | Best for |
|------|-------|----------|
| `auto` | turbovec if warm, LanceDB otherwise | Default — zero config |
| `hot` | turbovec only | Pure SIMD speed benchmarks |
| `cold` | LanceDB only | BM25, metadata filters, hybrid |
| `race` | First to respond | Measuring race win rate |
| `federated` | Both merged | Maximum recall diversity |

---

## Search type (orthogonal to mode)

| Type | Implementation | Score column | Requires |
|------|---------------|--------------|---------|
| `vector` | ANN cosine via turbovec / LanceDB IVF-PQ | cosine similarity | query embedding |
| `bm25` | LanceDB FTS (tantivy-backed inverted index) | raw BM25 float | `--mode cold` + query text |
| `hybrid` | LanceDB `execute_hybrid` (vector + BM25, internal RRF) | RRF rank score | `--mode cold` + embedding + query text |

---

## Ingestion pipeline

```
corpus.jsonl
     │
     ▼
 [load_jsonl]  ─────────────────────────────────────► Vec<Document>
                                                            │
                                               rayon::par_iter  (Phase 1)
                                               text trim + preprocess
                                                            │
                                          futures::stream::buffered(4)  (Phase 2)
                                          N concurrent embed HTTP calls
                                                            │
                                               embedder.embed_batch()
                                                            │
                                                     Vec<EmbeddedDoc>
                                                            │
                                          ┌─────────────────┴─────────────────┐
                                 TurboVecStore.upsert()          LanceDbStore.upsert()
                                 (Arc<RwLock<QuantizedIndex>>)   (RecordBatchIterator)
                                          │
                                     flush() ──► blob/turbovec/main/index.tq
                                               └► blob/turbovec/main/meta.json
                                          │
                                ensure_fts_index() ──► tantivy inverted index on `text`
```

Phase 1 (rayon) handles CPU-bound preprocessing in parallel. Phase 2 uses `buffered(4)` to keep 4 HTTP embedding requests in flight simultaneously, giving ~4× throughput over sequential embedding for network-bound backends (Ollama, OpenAI).

---

## Retrieval pipeline

```
"What causes Alzheimer's disease?"
     │
     ▼  Stage 0 — cold startup
     │     TurboVecStore::load_from_blob()
     │     ├── blob.get("turbovec/main/index.tq")  → QuantizedIndex in RAM
     │     └── blob.get("turbovec/main/meta.json") → HashMap<u64, String>
     │     LanceDbStore::new()  → open manifest (lazy, no vectors loaded)
     │
     ▼  Stage 1 — embed query
     │     POST /v1/embeddings → Vec<f32> len=768
     │
     ▼  Stage 2 — build SearchRequest
     │     { query_embedding: &[f32], k: 5, search_type: Vector }
     │
     ▼  Stage 3 — HybridStore dispatch (Auto)
     │     hot.is_warm()? → true → TurboVecStore
     │
     ▼  Stage 4 — SIMD scan
     │     4-bit quantize query → scan 5183 vectors (1.9 MB in L3 cache)
     │     NEON/AVX-512 inner products → top-k min-heap → Vec<(id, score)>
     │
     ▼  Stage 5 — text enrichment
     │     HashMap<u64, String> lookup for each result id
     │
     ▼  Stage 6 — output + metrics
           render table, emit Prometheus counters/histograms
```

Full stage-by-stage walkthroughs: [docs/ingestion.md](ingestion.md) · [docs/retrieval.md](retrieval.md)

---

## Cold startup time

| Step | Data | Approx time |
|------|------|-------------|
| Load `index.tq` from local blob | 2.5 MB | ~5ms |
| Deserialise `QuantizedIndex` | 5183 × 384 bytes = 1.9 MB | ~2ms |
| Load `meta.json` text map | 5183 entries | ~8ms |
| Open LanceDB manifest | manifest only (lazy) | ~1ms |
| **Total cold start** | | **~16ms** |

After cold start, queries are sub-millisecond (turbovec) or ~1–2ms (LanceDB flat scan).

---

## Trait surface (`crates/common`)

```rust
#[async_trait]
pub trait EmbeddingBackend: Send + Sync {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    async fn embed_one(&self, text: &str) -> Result<Vec<f32>>;  // default impl
    fn dim(&self) -> usize;
}

#[async_trait]
pub trait BlobBackend: Send + Sync {
    async fn put(&self, key: &str, data: Bytes) -> Result<()>;
    async fn get(&self, key: &str) -> Result<Bytes>;
    async fn list_prefix(&self, prefix: &str) -> Result<Vec<String>>;
    async fn delete(&self, key: &str) -> Result<()>;
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn upsert(&self, docs: &[EmbeddedDoc]) -> Result<()>;
    async fn search(&self, request: &SearchRequest<'_>) -> Result<Vec<ScoredDoc>>;
    async fn delete(&self, id: u64) -> Result<()>;
    async fn doc_count(&self) -> Result<usize>;
}
```

Swapping backends requires no code changes — only a config line:

| Config value | Implementation | Notes |
|-------------|---------------|-------|
| `embedding.backend = "fastembed"` | ONNX local | BGE-Base-EN, no API key |
| `embedding.backend = "openai-compat"` | HTTP (Ollama / OpenAI / Voyage) | set `base_url` + `model` |
| `embedding.backend = "synthetic"` | Random unit vectors | benchmarks only |
| `blob.backend = "local"` | `tokio::fs` | dev default |
| `blob.backend = "s3"` | `object_store` aws | MinIO or real S3 |
