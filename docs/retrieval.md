# Retrieval pipeline

This document traces what happens when you run `turbo-rag query "What causes Alzheimer's disease?"`.

---

## Overview

```
"What causes Alzheimer's disease?"
     │
     ▼  Stage 0 ── Cold startup: reconstruct in-memory state
     │               ├── Load turbovec index from blob (index.tq → RAM)
     │               ├── Load text map from blob (meta.json → HashMap)
     │               └── Open LanceDB manifest (lazy — no vectors loaded yet)
     │
     ▼  Stage 1 ── Embed the query
     │               POST /v1/embeddings → [f32; 768]
     │
     ▼  Stage 2 ── Build SearchRequest
     │               { query_embedding: &[f32], k: 5, search_type: Vector }
     │
     ▼  Stage 3 ── HybridStore dispatch (SearchMode::Auto)
     │               hot.is_warm()? → true → route to TurboVecStore
     │
     ▼  Stage 4 ── Turbovec SIMD search
     │               4-bit scan of 5183 vectors → Vec<(id, score)>
     │
     ▼  Stage 5 ── Text enrichment
     │               HashMap lookup: id → text for each result
     │
     ▼  Stage 6 ── Print + metrics
                     Render result table, emit Prometheus counters/histograms
```

---

## Stage 0 — Cold startup: reconstruct state

**Code:** `backend::build_store()`

Every invocation of `turbo-rag query` starts from scratch — there is no persistent server. The in-memory state is rebuilt from disk on every run.

```
TurboVecStore::with_blob("main", blob, dim=768, bits=4)
  → empty QuantizedIndex, empty texts HashMap, warm=false

LanceDbStore::new("./data/lance", "docs", 768)
  → lancedb::connect("./data/lance").execute().await
  → reads the current manifest (which version is active)
  → does NOT load vectors — Lance is lazy / memory-mapped

load_strategy = "from-blob":
  hot.load_from_blob()
    ├── blob.get("turbovec/main/index.tq")
    │     → tokio::fs::read("data/blob/turbovec/main/index.tq")  ~2.5 MB
    ├── QuantizedIndex::from_bytes(bytes, 768, 4)
    │     ├── write bytes to NamedTempFile
    │     ├── IdMapIndex::load(tempfile_path)   ← turbovec file API
    │     ├── delete tempfile
    │     └── index is now in RAM: 5183 vectors × 384 bytes = ~2 MB
    ├── blob.get("turbovec/main/meta.json")
    │     → reads JSON: {"count":5183, "dim":768, "bits":4, "texts":{...}}
    │     → populates HashMap<u64, String> with 5183 id→text entries  ~9 MB
    └── warm.store(true, SeqCst)
```

After this stage: turbovec is hot in RAM, LanceDB is open but cold on disk.

---

## Stage 1 — Embed the query

**Code:** `commands::query()` lines 63–65, `EmbeddingBackend::embed_one()`

```rust
let embedding = embedder.embed_one(&text).await?;
```

`embed_one` is a default method on the `EmbeddingBackend` trait — it wraps `embed_batch` with a single-element slice:

```rust
async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
    let mut batch = self.embed_batch(&[text.to_string()]).await?;
    batch.pop()
}
```

This triggers one HTTP round-trip to Ollama:

```
POST http://localhost:11434/v1/embeddings
{
  "model": "nomic-embed-text",
  "input": ["What causes Alzheimer's disease?"]
}

← 200 OK
{
  "data": [{ "embedding": [0.0231, -0.1142, 0.0887, ..., -0.0341] }]
}
```

The result is a `Vec<f32>` of length 768 — the geometric representation of the query's meaning in the same embedding space as the documents. The timer `t0` wraps this call and gives you `embed_ms` in the output.

The critical property: the query and all documents were embedded using the **same model** (`nomic-embed-text`). This is what makes cosine similarity meaningful — both live in the same vector space, so "closeness" corresponds to semantic relatedness.

---

## Stage 2 — Build the SearchRequest

**Code:** `commands::query()` lines 67–72

```rust
let search_req = SearchRequest::vector(&embedding, top_k);
```

`SearchRequest` is the unified input type for all vector stores:

```rust
pub struct SearchRequest<'a> {
    pub query_embedding: &'a [f32],   // borrowed — no copy of the 768 floats
    pub query_text: Option<&'a str>,  // None for vector search; set for bm25/hybrid
    pub k: usize,                     // 5
    pub search_type: SearchType,      // Vector | Bm25 | Hybrid { rrf_k }
    pub filter: Option<MetadataFilter>, // None — no metadata filtering
}
```

The embedding is **borrowed** for the lifetime of the request — the `'a` lifetime means the 768 floats live on the stack of `query()` and are not copied into the struct.

---

## Stage 3 — HybridStore dispatch

**Code:** `hybrid.rs::HybridStore::search()`

```rust
match &self.mode {
    SearchMode::Auto => {
        if self.hot.is_warm() {   // reads AtomicBool — one CPU instruction
            self.hot.search(request).await
        } else {
            self.cold.search(request).await
        }
    }
    SearchMode::Hot       => self.hot.search(request).await,
    SearchMode::Cold      => self.cold.search(request).await,
    SearchMode::Race      => tokio::select! {
        r = self.hot.search(request)  => r,
        r = self.cold.search(request) => r,
    },
    SearchMode::Federated(k) => { /* run both, merge, deduplicate */ }
}
```

In `Auto` mode, `is_warm()` reads the `AtomicBool` set during `load_from_blob()`. Since the blob was loaded in Stage 0, this is `true`, and the request goes to turbovec.

**What each mode means in practice:**

| Mode | Which store answers | When to use |
|------|---------------------|-------------|
| `auto` | turbovec (if warm), else LanceDB | Default — zero config |
| `hot` | turbovec only | Benchmarking pure SIMD speed |
| `cold` | LanceDB only | When you need BM25 / metadata filters |
| `race` | Whichever responds first (turbovec wins >99%) | Benchmarking; measuring race win rate |
| `federated` | Both — results merged and deduplicated | When diversity matters more than speed |

---

## Stage 4 — Turbovec SIMD search

**Code:** `TurboVecStore::search()` → `QuantizedIndex::search()` → `IdMapIndex::search()`

```rust
let hits = self.index.read().await.search(request.query_embedding, request.k);
```

`self.index.read().await` acquires a **shared read lock** — multiple concurrent queries can hold read locks simultaneously; only upserts require an exclusive write lock.

Inside `QuantizedIndex::search`:

```rust
pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
    let (scores, ids) = self.id_map.search(query, k);
    ids.into_iter().zip(scores).collect()
}
```

Inside turbovec's `IdMapIndex::search`, three things happen:

**1. Quantize the query on the fly**

The 768 f32 query values are scalar-quantized to 4 bits using the same per-dimension bounds that were recorded during ingestion:

```
query[dim_i] → round((query[dim_i] - min[dim_i]) / (max[dim_i] - min[dim_i]) × 15)
```

The result is a 384-byte compact query vector (768 dimensions × 4 bits ÷ 8).

**2. SIMD linear scan**

The quantized query is compared against all 5,183 stored document vectors using SIMD instructions (NEON on Apple Silicon, AVX-512 on x86-64). The CPU computes approximate inner products on 4-bit values, processing multiple comparisons per instruction cycle.

```
Data scanned: 5,183 vectors × 384 bytes = ~1.9 MB
              → fits entirely in L3 cache on most CPUs

Criterion benchmark results (release mode, NEON SIMD, no async overhead):
  1,000 docs  →  52 µs  (0.052ms)
  10,000 docs → 268 µs  (0.268ms)
  Scaling: 10× docs → ~5× time  (SIMD compresses the linear growth)
```

This is a **brute-force scan** — every document is compared to the query. At 5,183 documents turbovec beats LanceDB's indexed search because index overhead (IVF cluster assignment, PQ decompression) costs more than a direct SIMD scan at this scale. LanceDB's IVF-PQ index only pays off beyond ~50,000–100,000 documents.

**3. Top-k selection**

turbovec maintains a running min-heap of size k, returning the top 5 by score:

```
[(id: 4983,  score: 0.847),
 (id: 18238, score: 0.831),
 (id: 2349,  score: 0.819),
 (id: 7621,  score: 0.804),
 (id: 11023, score: 0.791)]
```

The scores are **approximate** cosine similarities. The 4-bit quantization introduces a small error — in practice, recall@10 on standard benchmarks stays above 0.99.

---

## Stage 5 — Text enrichment

**Code:** `TurboVecStore::search()` lines 140–150

turbovec only stores vectors and IDs. The text comes from the `HashMap<u64, String>` that was loaded from `meta.json` in Stage 0:

```rust
let texts = self.texts.read().await;   // shared read lock — concurrent queries OK
Ok(hits
    .into_iter()
    .map(|(id, score)| ScoredDoc {
        text: texts.get(&id).cloned().unwrap_or_default(),
        id,
        score,
        source: SearchSource::TurboVec,
        metadata: Default::default(),
    })
    .collect())
```

Five HashMap lookups in a 5,183-entry map takes nanoseconds. The result is:

```rust
vec![
    ScoredDoc { id: 4983,  score: 0.847, text: "Microstructural development...", source: TurboVec },
    ScoredDoc { id: 18238, score: 0.831, text: "Amyloid beta deposits...",        source: TurboVec },
    ScoredDoc { id: 2349,  score: 0.819, text: "Tau protein...",                  source: TurboVec },
    ScoredDoc { id: 7621,  score: 0.804, text: "Neuroinflammation...",            source: TurboVec },
    ScoredDoc { id: 11023, score: 0.791, text: "Apolipoprotein E genotype...",    source: TurboVec },
]
```

---

## Stage 6 — Output and metrics

**Code:** `commands::query()` lines 80–104

The timer `t1` (started just before `store.search()`) stops here, giving `search_ms`. The result is printed and Prometheus metrics are emitted:

```
Query : "What causes Alzheimer's disease?"
Mode  : auto  |  Type: vector  |  Top-5
Times : embed 24ms  search 1ms  total 25ms

#    id         score    text
────────────────────────────────────────────────────────────────────────
1    4983       0.8470   Microstructural development of human newborn ce
2    18238      0.8310   Amyloid beta deposits are a hallmark of AD...
3    2349       0.8190   Tau protein hyperphosphorylation leads to...
4    7621       0.8040   Neuroinflammation plays a key role in...
5    11023      0.7910   Apolipoprotein E genotype is the strongest...

Source: Some(TurboVec)
```

Prometheus counters updated:
- `turborag_queries_total{mode="auto", type="vector"}` += 1
- `turborag_embed_latency_ms` histogram: record 24ms
- `turborag_search_latency_ms{mode="auto"}` histogram: record 1ms

---

## What changes with `--mode cold` (LanceDB path)

Stage 3 routes to `LanceDbStore::search()` instead:

```rust
table.query()
    .nearest_to(request.query_embedding)   // 768 f32 values
    .limit(5)
    .execute()
    .await
```

LanceDB performs an **IVF-PQ** search (with an index) or a flat scan (without). The table opened in Stage 0 is memory-mapped — Lance pages in the data it needs from disk rather than loading everything upfront.

After scanning, results come back as Arrow `RecordBatch` objects. These are unpacked column by column in `extract_scored_docs`:

```rust
fn extract_scored_docs(batch: &RecordBatch, source: &SearchSource) -> Result<Vec<ScoredDoc>> {
    let ids   = batch.column_by_name("id")   // UInt64Array
    let texts = batch.column_by_name("text") // StringArray
    // score is assigned as a rank: 1.0, 0.8, 0.6, 0.4, 0.2
    // (LanceDB does not return cosine scores directly in this API version)
}
```

Note: the score in cold-mode results is a **rank-based approximation** (1.0 for rank 1, 0.8 for rank 2, etc.), not a true cosine similarity. This is a current limitation — LanceDB 0.26's query API does not surface the raw distance scores in the returned batch.

---

## BM25 — implemented

BM25 and hybrid search are fully implemented. After an ingest the FTS index is built automatically.

### What BM25 actually is

BM25 (Best Match 25) scores documents by **term frequency and rarity**. For the query `"tau protein"`:

```
score(doc, query) = Σ  IDF(term) × TF(term, doc) × (k1 + 1)
                   term          ─────────────────────────────────
                                 TF(term, doc) + k1 × (1 - b + b × |doc|/avgdl)
```

Where:
- `IDF(term)` = log((N - df + 0.5) / (df + 0.5)) — how rare the term is across all docs
- `TF(term, doc)` = how many times the term appears in this document
- `k1 = 1.2`, `b = 0.75` — tuning parameters
- `|doc|/avgdl` — normalises for document length

BM25 excels at **exact keyword matches** — acronyms, proper nouns, identifiers, rare technical terms. Vector search excels at **semantic relatedness** — paraphrases, synonyms, conceptual similarity.

### What the real implementation would look like

LanceDB supports BM25 via a full-text search index. The implementation requires:

**1. Create an FTS index on the `text` column after ingestion:**

```rust
// In LanceDbStore, after upsert when doc_count > some threshold:
table.create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
     .execute()
     .await?;
```

**2. Query with full-text search:**

```rust
SearchType::Bm25 => {
    let query_text = request.query_text
        .ok_or(TurboError::Store("BM25 requires query_text".into()))?;
    table
        .query()
        .full_text_search(FullTextSearchQuery::new(query_text.to_string()))
        .limit(request.k)
        .execute()
        .await?
}
```

**3. Hybrid search with Reciprocal Rank Fusion:**

```rust
SearchType::Hybrid { rrf_k } => {
    // Run both searches, then fuse rankings
    let vector_results = table.query().nearest_to(query_vec).limit(k * 2).execute().await?;
    let bm25_results   = table.query().full_text_search(query_text).limit(k * 2).execute().await?;

    // RRF score: Σ 1 / (rrf_k + rank_i)
    // A document ranked 1st in both lists scores ~2 × 1/(60+1) = 0.033
    // vs a document ranked 1st in one list only: ~0.016
    rrf_merge(vector_results, bm25_results, rrf_k, k)
}
```

Hybrid with RRF consistently outperforms either alone on standard benchmarks by 5–15% NDCG. It catches documents that are semantically related but use different vocabulary (vector wins) and documents that contain the exact query terms but are topically broader (BM25 wins).
