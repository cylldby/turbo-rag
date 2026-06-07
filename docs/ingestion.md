# Ingestion pipeline

This document walks through every stage that runs when you execute `turbo-rag ingest`.

---

## Overview

```
corpus.jsonl
     │
     ▼  Stage 0 ── Config + backend init
     │
     ▼  Stage 1 ── Load & parse JSONL  (handles both our format and BEIR format)
     │
     ▼  Stage 2 ── Parallel preprocessing  (rayon, all CPU cores)
     │
     ▼  Stage 3 ── Concurrent embedding    (HTTP, buffered(4) async)
     │                   │
     │     ┌─────────────┘  (64 texts per batch, 4 batches in flight)
     │     ▼
     │  embed_batch() → Vec<Vec<f32>>
     │
     ▼  Stage 4 ── Dual-store upsert      (tokio::try_join! — both stores written simultaneously)
     │     ├── TurboVecStore: SIMD quantise + append to QuantizedIndex + update texts HashMap
     │     └── LanceDbStore:  Arrow RecordBatch → append to Lance columnar file
     │
     ▼  Stage 5 ── Stats + flush
           ├── Print PipelineStats (docs, elapsed, throughput, compression ratio)
           ├── Emit Prometheus metrics
           └── TurboVecStore.flush() → write index.tq + meta.json to blob
```

---

## Stage 0 — Config + backend construction

**Code:** `commands::ingest()` lines 15–30, `backend::build_store()`

Before touching any data, the CLI reconstructs all backends from config:

```
config::load()
  reads: config/default.toml
  merges: .env file (if present)
  merges: environment variables (EMBEDDING__BACKEND=..., etc.)
  forces: load_strategy = "lazy"   ← ingest never loads the previous index;
                                      it overwrites it at the end
```

Then it constructs:

| Object | What | Cost |
|--------|------|------|
| `OpenAICompatBackend` | HTTP client + struct with URL/model/dim | Zero — no network call |
| `LocalFsBackend` | Resolves `./data/blob` path | Zero |
| `TurboVecStore` | Empty `QuantizedIndex` in RAM | Zero |
| `LanceDbStore` | Opens Lance manifest file, reads current version | Fast disk read |

`TurboVecStore` starts with `warm = false`. Because `load_strategy = "lazy"`, `load_from_blob()` is **not** called — the previous index is ignored until after the new one is built.

---

## Stage 1 — Load and parse JSONL

**Code:** `commands::load_jsonl()`

The file is read line by line. Each line is parsed through a flexible deserializer that handles two formats simultaneously:

**Our fixture format:**
```json
{"id": 1, "text": "Machine learning is...", "metadata": {"category": "ai"}}
```

**BEIR format (SciFact, MS MARCO, etc.):**
```json
{"_id": "4983", "title": "Microstructural development...", "text": "Alterations of...", "metadata": {}}
```

The deserializer:
1. Resolves `id`: tries numeric `id` → string `id` → string `_id` → falls back to line number
2. Concatenates `title + ". " + text` if both fields are present (BEIR datasets always have a title)
3. Keeps `metadata` as-is (empty `{}` for BEIR is fine)

The entire corpus ends up as `Vec<Document>` in RAM before any embedding starts. For SciFact (5,183 docs) this takes milliseconds.

---

## Stage 2 — Parallel preprocessing

**Code:** `pipeline::run()` lines 59–64

```rust
let preprocessed: Vec<(u64, String, _)> = docs
    .par_iter()
    .map(|d| (d.id, d.text.trim().to_string(), d.metadata.clone()))
    .collect();
```

rayon distributes this across **all CPU cores** simultaneously. The work is purely CPU-bound: trimming whitespace from each document's text. For 5,183 documents averaging 1,900 characters each, this finishes in tens of milliseconds.

This phase exists separately from embedding for a reason: embedding is network I/O-bound (waiting for Ollama). Preprocessing is CPU-bound. Separating them means the CPU work doesn't serialise with the network waits.

---

## Stage 3 — Concurrent embedding

**Code:** `pipeline::run()` lines 66–99

This stage is where nearly all the wall-clock time is spent.

### Batching

The 5,183 preprocessed documents are split into **chunks of `batch_size=64`**. For SciFact: 80 full batches of 64 + 1 partial batch of 23 = **81 HTTP requests total**.

### Concurrency with `buffered(4)`

```rust
stream::iter(chunks)
    .map(|chunk| async move {
        embedder.embed_batch(&texts).await  // one HTTP POST per chunk
    })
    .buffered(4)   // ← keeps 4 requests in flight simultaneously
    .collect()
    .await
```

`buffered(4)` is from the `futures` crate. It works like a sliding window:

```
time →

Without buffered:
  batch 1:  [POST ──── response]
  batch 2:                      [POST ──── response]
  batch 3:                                          [POST ──── response]
  Total: 81 × 24ms = 1,944ms

With buffered(4):
  batch 1:  [POST ──── response]
  batch 2:    [POST ──── response]
  batch 3:      [POST ──── response]
  batch 4:        [POST ──── response]
  batch 5:                [POST ──── response]
  ...
  Total: ≈ (81 ÷ 4) × 24ms = ~486ms
```

For fastembed (local ONNX), `buffered(1)` is optimal — the ONNX runtime already uses all CPU cores internally, and sending concurrent requests would just queue inside the same process.

### What one HTTP roundtrip looks like

```
POST http://localhost:11434/v1/embeddings
{
  "model": "nomic-embed-text",
  "input": [
    "Microstructural development of human newborn cerebral white matter...",
    "Adenosine is a signaling molecule...",
    // ... 62 more texts
  ]
}

← 200 OK
{
  "data": [
    { "embedding": [0.0231, -0.1142, 0.0887, ...] },  // 768 floats
    { "embedding": [0.0519, -0.0334, 0.1201, ...] },
    // ... 63 more
  ]
}
```

Each response contains 64 × 768 = **49,152 floats** ≈ 192 KB of JSON. The floats are decoded by `serde_json` and zipped with the original `(id, text, metadata)` tuples to produce `Vec<EmbeddedDoc>`.

The progress bar updates once per completed batch:
```
embed  [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░] 3200/5183 docs  12 docs/sec
```

---

## Stage 4 — Dual-store upsert

**Code:** `pipeline::run()` lines 103–122, `HybridStore::upsert()`

Each completed embedding batch is written to **both stores at the same time**:

```rust
tokio::try_join!(self.hot.upsert(docs), self.cold.upsert(docs))?;
```

`try_join!` drives both futures on the same tokio thread, interleaving them at each `.await` point. If either fails, the whole join fails immediately.

### Into TurboVecStore (hot path)

```
docs[0..63]
  ├── extract ids:    [4983, 14, 21047, ...]           Vec<u64>
  ├── extract vecs:   [[0.023, -0.11, ...], ...]       Vec<Vec<f32>>
  │
  ├── QuantizedIndex::add_batch(ids, vecs)
  │     └── for each (id, vec):
  │           turbovec::IdMapIndex::add_with_ids(vec, &[id])
  │             ── per-dimension min/max computed or updated
  │             ── 768 f32 values → 768 × 4 bits = 384 bytes  (scalar quantisation)
  │             ── stored in the SIMD-friendly layout inside IdMapIndex
  │
  ├── texts HashMap: insert { 4983 → "Microstructural...", 14 → "Adenosine...", ... }
  │
  ├── count += 64    (AtomicUsize, relaxed ordering — just a counter)
  └── warm.store(true, SeqCst)    ← set on first upsert; subsequent ones are no-ops
```

The 4-bit scalar quantisation happens here. Each f32 value is mapped to the nearest of 16 levels (2⁴) using the observed min/max for that dimension across the whole index. This is **data-oblivious** — no training or calibration needed — which is why turbovec can quantise incrementally as documents arrive.

### Into LanceDbStore (cold path, simultaneous)

```
docs[0..63]
  ├── make_batch():
  │     ids column:        UInt64Array  [4983, 14, ...]        8 bytes × 64 rows
  │     texts column:      StringArray  ["Microstructural...", ...]
  │     embeddings column: FixedSizeListArray  [768 × f32 × 64 rows]  = 196,608 bytes
  │     → Arrow RecordBatch (columnar, zero-copy layout)
  │
  ├── RecordBatchIterator wraps it for streaming
  │
  └── lancedb::Table::add(reader).execute().await
        ── serialises to Lance columnar format (bit-packed, dictionary-encoded)
        ── appends a new data fragment file: data/lance/docs.lance/data/<uuid>.lance
        ── writes a transaction log entry: _transactions/<version>.txn
        ── does NOT rewrite existing fragments (append-only)
```

LanceDB stores the full f32 embeddings (not quantised). This is intentional: LanceDB is the source of truth for exact vectors, supporting BM25 and SQL filtering. turbovec is the fast approximate query layer.

### Progress bar

```
store  [████████████░░░░░░░░░░░░░░░░░░░░░░░] 3264/5183 docs
```

---

## Stage 5 — Stats + flush

**Code:** `commands::ingest()` lines 36–48

### Stats

After all 81 batches have been embedded and stored:

```
─── Ingestion Stats ──────────────────────────────────
  Documents  : 5183
  Elapsed    : 487.3s
  Throughput : 10 docs/sec
  Original   : 15.2 MB      ← 5183 × 768 × 4 bytes (raw f32)
  Compressed : 1.9 MB       ← 5183 × 768 × 0.5 bytes (4-bit)
  Ratio      : 8.0×
──────────────────────────────────────────────────────
```

The compression ratio is calculated analytically (`compression_ratio(dim=768, bits=4) = 32/4 = 8.0`), not by measuring the actual file size. The actual `index.tq` on disk is slightly smaller due to turbovec's internal layout overhead at small corpus sizes.

### Flush

```rust
store.hot.flush().await?
```

This is the only time the turbovec index touches disk during an ingest run:

```
QuantizedIndex::to_bytes()
  ├── NamedTempFile::new()   — creates e.g. /tmp/.tmpXXXXX
  ├── IdMapIndex::write("/tmp/.tmpXXXXX")   — turbovec writes its binary format
  ├── std::fs::read("/tmp/.tmpXXXXX")   — read the bytes back
  └── tempfile is dropped and deleted

LocalFsBackend::put("turbovec/main/index.tq", bytes)
  └── tokio::fs::write("data/blob/turbovec/main/index.tq", bytes)

texts_snapshot = { 4983 → "Microstructural...", ... }   // all 5183 entries
LocalFsBackend::put("turbovec/main/meta.json", ...)
  └── tokio::fs::write("data/blob/turbovec/main/meta.json",
        json!({ "count": 5183, "dim": 768, "bits": 4, "texts": {...} }))
```

The tempfile round-trip exists because turbovec's API only accepts file paths for serialization, not byte slices. Writing to a tempfile and reading it back is the bridge between turbovec's file API and the blob backend's bytes API.

After the flush, disk contains:

```
data/blob/turbovec/main/
  index.tq    ~2.5 MB   TurboQuant 4-bit SIMD index
  meta.json   ~9.3 MB   count + dim + bits + full id→text map

data/lance/docs.lance/
  data/        ~3 MB    Arrow columnar fragments (full f32, text, ids)
  _versions/            version manifests
  _transactions/        write-ahead log entries
```

LanceDB data was committed incrementally during Stage 4. The turbovec blob is written once at the end, atomically.
