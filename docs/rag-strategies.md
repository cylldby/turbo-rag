# RAG strategies

The current system implements **naive RAG**: embed query → nearest-neighbour search → return results. This works well as a baseline but leaves significant recall and precision on the table. This document explains the limitations, the strategies that address them, and how each would be implemented in this codebase.

---

## What naive RAG gets wrong

```
User query → embed → ANN search → top-k docs → [LLM generation]
```

Three structural problems:

**1. The vocabulary gap.** Embedding models map meaning to geometry, but they can miss exact term matches. If a user asks about "ApoE4" and a document says "Apolipoprotein E ε4 allele", the cosine similarity may be low even though they refer to the same thing. BM25 would catch it; vector search may not.

**2. Single-shot retrieval.** A query like "How does the treatment for Alzheimer's compare to Parkinson's?" needs information from multiple sources. One retrieval pass returns documents about one disease or the other, rarely both in the right proportions.

**3. Query-document mismatch.** Queries are short questions; documents are long answers. Embedding a question like "What causes neurodegeneration?" and a paper abstract about neurodegeneration produces vectors that may not be geometrically close, because a question and its answer look different in embedding space. The model optimises for sentence similarity, not question-answer similarity.

---

## Strategy 1: Hybrid search (BM25 + vector)

**Impact: high. Effort: low. Status: implemented.**

Run both vector search and BM25 in parallel, then fuse the rankings with Reciprocal Rank Fusion.

```
query
  ├──[embed]──► vector search    top-20 by cosine  ─────┐
  └──[BM25]───► keyword search   top-20 by TF-IDF  ─────┤
                                                         ▼
                                                    RRF merge
                                                         │
                                                    top-5 final
```

**RRF formula:**

```
score_rrf(doc) = Σ  1 / (k + rank_in_list_i)
               lists i
```

With `k=60` (the standard constant), a document ranked 1st in both lists scores `1/61 + 1/61 = 0.033`. A document ranked 1st in only one list scores `1/61 = 0.016`. Documents that appear in both lists are strongly promoted.

**Why it works:** BM25 wins on rare terms, acronyms, proper nouns, identifiers. Vector search wins on paraphrases, synonyms, conceptual proximity. Hybrid catches what either alone misses. Typical gains: +5–15% NDCG@10 on standard benchmarks.

**Implementation in this codebase:**

The `SearchType::Hybrid { rrf_k }` variant and `SearchRequest::hybrid()` constructor already exist in `common`. The only missing piece is the actual LanceDB FTS index and query in `lance_store.rs`:

```rust
// 1. After ingest, create FTS index (add to LanceDbStore::upsert or as a separate method):
pub async fn ensure_fts_index(&self) -> Result<()> {
    let table = self.table().await?;
    table
        .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
        .execute()
        .await
        .map_err(|e| TurboError::Store(e.to_string()))
}

// 2. In LanceDbStore::search, replace the stub:
SearchType::Bm25 => {
    let text = request.query_text.ok_or(TurboError::Store("bm25 requires query_text".into()))?;
    table
        .query()
        .full_text_search(FullTextSearchQuery::new(text.to_string()))
        .limit(request.k)
        .execute()
        .await
        .map_err(|e| TurboError::Store(e.to_string()))?
}

SearchType::Hybrid { rrf_k } => {
    let text = request.query_text.ok_or(TurboError::Store("hybrid requires query_text".into()))?;
    // Run both in parallel
    let (vec_stream, bm25_stream) = tokio::join!(
        table.query().nearest_to(request.query_embedding)?.limit(request.k * 3).execute(),
        table.query().full_text_search(FullTextSearchQuery::new(text.to_string())).limit(request.k * 3).execute()
    );
    let vec_docs  = collect_docs(vec_stream?,  &SearchSource::LanceDb)?;
    let bm25_docs = collect_docs(bm25_stream?, &SearchSource::LanceDb)?;
    rrf_merge(vec_docs, bm25_docs, *rrf_k, request.k)
}
```

```rust
// rrf_merge can live in common/src/lib.rs:
pub fn rrf_merge(a: Vec<ScoredDoc>, b: Vec<ScoredDoc>, k: usize, top_n: usize) -> Vec<ScoredDoc> {
    let mut scores: HashMap<u64, f32> = HashMap::new();
    for (rank, doc) in a.iter().enumerate() {
        *scores.entry(doc.id).or_insert(0.0) += 1.0 / (k + rank + 1) as f32;
    }
    for (rank, doc) in b.iter().enumerate() {
        *scores.entry(doc.id).or_insert(0.0) += 1.0 / (k + rank + 1) as f32;
    }
    let mut all: Vec<ScoredDoc> = a.into_iter().chain(b).collect();
    all.dedup_by_key(|d| d.id);  // keep first occurrence (vector result)
    all.sort_by(|x, y| scores[&y.id].partial_cmp(&scores[&x.id]).unwrap());
    all.truncate(top_n);
    all
}
```

---

## Strategy 2: HyDE — Hypothetical Document Embeddings

**Impact: high for factual Q&A. Effort: medium. Status: not implemented.**

Instead of embedding the question directly, ask an LLM to generate a *hypothetical answer*, then embed that.

```
"What causes Alzheimer's disease?"
         │
         ▼ LLM (e.g. ollama/llama3)
"Alzheimer's disease is caused by accumulation of amyloid-beta plaques
 and tau protein tangles, leading to neuronal death. Risk factors include
 APOE4 genotype, age, and family history..."
         │
         ▼ embed_one()
    [f32; 768]   ← now in "answer space", not "question space"
         │
         ▼ ANN search
    top-5 real documents
```

**Why it works:** Questions and answers have different geometric properties in embedding space. "What causes X?" and a paper explaining X may be distant vectors. "X is caused by Y and Z" and a paper about X are close vectors. By generating a hypothetical answer first, you move the query into the same part of the space as the real answers.

HyDE typically improves recall@10 by 10–30% on factual retrieval tasks with no changes to the index.

**Implementation:**

Add a `HydeEmbedder` wrapper in `crates/embedder/src/hyde.rs`:

```rust
pub struct HydeEmbedder {
    llm_client: reqwest::Client,
    llm_url: String,
    llm_model: String,
    inner: Arc<dyn EmbeddingBackend>,
}

impl HydeEmbedder {
    pub fn new(llm_url: &str, llm_model: &str, inner: Arc<dyn EmbeddingBackend>) -> Self { ... }
}

#[async_trait]
impl EmbeddingBackend for HydeEmbedder {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // For each query, generate a hypothetical answer, then embed the answer
        let hypotheticals = futures::stream::iter(texts)
            .map(|q| self.generate_hypothetical(q))
            .buffered(4)
            .collect::<Vec<_>>()
            .await;
        self.inner.embed_batch(&hypotheticals.into_iter().flatten().collect::<Vec<_>>()).await
    }

    async fn generate_hypothetical(&self, question: &str) -> Result<String> {
        // POST to ollama /api/generate
        let prompt = format!(
            "Write a short factual answer (2-3 sentences) to this question. \
             Answer directly without preamble.\n\nQuestion: {question}\n\nAnswer:"
        );
        // call ollama generate API, return the generated text
    }
}
```

Wire it up in `backend::build_embedder()` with a config flag:

```toml
[embedding]
hyde = true          # off by default
hyde_model = "llama3.2"
```

HyDE adds one LLM round-trip per query (~200–500ms with a local 8B model). For latency-sensitive applications, it can run in parallel with the standard embedding call and pick whichever returns a better result.

---

## Strategy 3: Re-ranking with a cross-encoder

**Impact: very high precision gains. Effort: medium. Status: not implemented.**

The current system uses a **bi-encoder** (embed query separately, embed docs separately, compare vectors). Bi-encoders are fast but imprecise — the interaction between query and document happens only at the cosine similarity step.

A **cross-encoder** takes (query, document) as a pair and models their interaction directly:

```
current:
  embed(query) → [f32; 768]
  embed(doc)   → [f32; 768]
  cosine(q, d) → score             ← no interaction, just geometry

cross-encoder:
  model([QUERY] "alzheimer causes" [SEP] "Tau protein...") → relevance score
```

The cross-encoder reads both at once, so it can model subtle relevance signals that bi-encoders miss.

**Two-stage pipeline:**

```
query
  │
  ▼ turbovec ANN (fast, high recall)
  top-50 candidates
  │
  ▼ cross-encoder re-rank (slower, high precision)
  top-5 final results
```

The ANN stage provides recall (get the right documents in the candidate set). The re-ranker provides precision (put the most relevant ones first).

**Implementation:**

Add a `Reranker` trait to `common`:

```rust
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Score each (query, document) pair. Returns scores in the same order as docs.
    async fn rerank(&self, query: &str, docs: &[ScoredDoc]) -> Result<Vec<f32>>;
}
```

Implement with Ollama's reranking endpoint or Cohere Rerank API:

```rust
pub struct OllamaReranker {
    client: reqwest::Client,
    base_url: String,
    model: String,  // e.g. "bge-reranker-v2-m3" via ollama
}

#[async_trait]
impl Reranker for OllamaReranker {
    async fn rerank(&self, query: &str, docs: &[ScoredDoc]) -> Result<Vec<f32>> {
        // POST /api/rerank
        // { "model": "bge-reranker-v2-m3", "query": "...", "documents": [...] }
        // returns relevance scores for each doc
    }
}
```

Wire it into the query command:

```rust
// In commands::query(), after store.search() returns top-50:
if let Some(reranker) = &reranker {
    let scores = reranker.rerank(&text, &results).await?;
    results.iter_mut().zip(scores).for_each(|(doc, s)| doc.score = s);
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results.truncate(top_k);
}
```

Cross-encoder re-ranking adds 50–200ms depending on model size and number of candidates, but typically improves NDCG@5 by 20–40% over bi-encoder retrieval alone.

---

## Strategy 4: MMR — Maximum Marginal Relevance

**Impact: medium (diversity). Effort: low. Status: not implemented but trivial to add.**

Standard ANN search can return five documents that all say essentially the same thing. MMR explicitly trades off relevance against redundancy.

**Algorithm:** pick results one at a time. Each pick maximises:

```
score_mmr(doc) = λ × cosine(doc, query) - (1-λ) × max_j cosine(doc, already_selected_j)
```

- `λ = 1.0` → pure relevance (standard ANN behaviour)
- `λ = 0.5` → equal weight to relevance and diversity  
- `λ = 0.0` → pure diversity (picks maximally different documents)

**Implementation:** pure Rust, no new deps. `cosine_similarity` already exists in `common`:

```rust
// Add to common/src/lib.rs:
pub fn mmr_select(
    candidates: Vec<ScoredDoc>,
    query_embedding: &[f32],
    doc_embeddings: &HashMap<u64, Vec<f32>>,  // needed for doc-doc similarity
    lambda: f32,
    k: usize,
) -> Vec<ScoredDoc> {
    let mut selected: Vec<&ScoredDoc> = Vec::new();
    let mut remaining: Vec<&ScoredDoc> = candidates.iter().collect();

    while selected.len() < k && !remaining.is_empty() {
        let next = remaining.iter().max_by(|a, b| {
            let rel_a = cosine_similarity(doc_embeddings[&a.id].as_slice(), query_embedding);
            let rel_b = cosine_similarity(doc_embeddings[&b.id].as_slice(), query_embedding);

            let red_a = selected.iter()
                .map(|s| cosine_similarity(&doc_embeddings[&a.id], &doc_embeddings[&s.id]))
                .fold(0.0f32, f32::max);
            let red_b = selected.iter()
                .map(|s| cosine_similarity(&doc_embeddings[&b.id], &doc_embeddings[&s.id]))
                .fold(0.0f32, f32::max);

            let score_a = lambda * rel_a - (1.0 - lambda) * red_a;
            let score_b = lambda * rel_b - (1.0 - lambda) * red_b;
            score_a.partial_cmp(&score_b).unwrap()
        }).copied().unwrap();

        selected.push(next);
        remaining.retain(|d| d.id != next.id);
    }
    selected.into_iter().cloned().collect()
}
```

The catch: MMR needs the embedding of each candidate document, not just their text. Currently `ScoredDoc` doesn't carry the embedding vector. The turbovec store would need to expose a way to retrieve vectors by ID, or the embeddings need to be stored alongside results.

Add `--lambda` as a CLI flag: `turbo-rag query "..." --mmr-lambda 0.7`.

---

## Strategy 5: Chunking — retrieve at sentence level, return at paragraph level

**Impact: high for long documents. Effort: medium. Status: not implemented.**

Currently each SciFact document (title + abstract, ~1900 chars) is stored as one unit. The problem: a long document may contain the relevant sentence in paragraph 3, but the cosine similarity is diluted by the irrelevant content in paragraphs 1, 2, and 4.

**Parent-child chunking:**

```
document (1900 chars)
  ├── chunk 0 (400 chars, 50-char overlap with chunk 1)  ← stored as embedding unit
  ├── chunk 1 (400 chars, 50-char overlap with chunk 2)  ← stored as embedding unit
  ├── chunk 2 (400 chars, 50-char overlap with chunk 3)  ← stored as embedding unit
  └── chunk 3 (300 chars)                                ← stored as embedding unit
```

Retrieve at chunk level (better cosine match), return the parent document (full context).

**Implementation:** add a `Chunker` to `crates/pipeline/src/chunker.rs`:

```rust
pub struct Chunker {
    pub chunk_size: usize,    // in characters, e.g. 400
    pub overlap: usize,       // in characters, e.g. 50
}

impl Chunker {
    pub fn chunk(&self, doc: &Document) -> Vec<Document> {
        let text = &doc.text;
        let mut chunks = Vec::new();
        let mut start = 0;
        let mut chunk_idx = 0u64;

        while start < text.len() {
            let end = (start + self.chunk_size).min(text.len());
            // Find a sentence boundary near end to avoid mid-sentence splits
            let end = find_sentence_boundary(text, end);

            let mut metadata = doc.metadata.clone();
            metadata.insert("parent_id".into(), doc.id.to_string());
            metadata.insert("chunk_idx".into(), chunk_idx.to_string());

            chunks.push(Document {
                id: doc.id * 1000 + chunk_idx,   // stable chunk ID
                text: text[start..end].to_string(),
                metadata,
            });
            start = end.saturating_sub(self.overlap);
            chunk_idx += 1;
        }
        chunks
    }
}
```

Then in the query command, after retrieving top-k chunks, **deduplicate by `parent_id`** and return the full parent document text:

```rust
// After search returns chunk-level results:
let parent_ids: Vec<u64> = results.iter()
    .filter_map(|r| r.metadata.get("parent_id")?.parse().ok())
    .collect::<IndexSet<_>>()  // deduplicated, order-preserving
    .into_iter().take(top_k).collect();

// Fetch full parent documents from LanceDB by ID
let parent_docs = store.cold.get_by_ids(&parent_ids).await?;
```

This requires a `get_by_ids` method on `LanceDbStore` (a SQL `WHERE id IN (...)` query), which is straightforward to add.

---

## Strategy 6: Query decomposition for multi-hop questions

**Impact: high for complex questions. Effort: high. Status: not implemented.**

For a question like "Compare the mechanisms of Alzheimer's and Parkinson's disease", a single retrieval pass returns documents about one or the other, not a structured comparison. Query decomposition breaks it into sub-queries:

```
"Compare mechanisms of Alzheimer's and Parkinson's"
         │
         ▼ LLM decomposition
   ["What is the mechanism of Alzheimer's disease?",
    "What is the mechanism of Parkinson's disease?"]
         │
         ▼ parallel retrieval for each sub-query
   results_1: top-5 for Alzheimer's
   results_2: top-5 for Parkinson's
         │
         ▼ merge, deduplicate, rank
   top-10 combined
```

**Implementation:** add a `QueryDecomposer` to the embedder crate:

```rust
pub struct QueryDecomposer {
    client: reqwest::Client,
    llm_url: String,
    model: String,
}

impl QueryDecomposer {
    pub async fn decompose(&self, query: &str) -> Result<Vec<String>> {
        let prompt = format!(
            "Break this question into 1-3 simpler sub-questions for document retrieval. \
             Return one question per line, nothing else.\n\nQuestion: {query}"
        );
        // call ollama, parse response lines into Vec<String>
        // if only one sub-question returned (or same as input), return vec![query]
    }
}
```

Then in `commands::query()`:

```rust
let sub_queries = decomposer.decompose(&text).await?;
let mut all_results = Vec::new();
for sub_q in &sub_queries {
    let emb = embedder.embed_one(sub_q).await?;
    let req = SearchRequest::vector(&emb, top_k);
    let mut r = store.search(&req).await?;
    all_results.append(&mut r);
}
// deduplicate by id, re-rank by score
all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
all_results.dedup_by_key(|d| d.id);
all_results.truncate(top_k);
```

---

## Strategy comparison

| Strategy | Recall gain | Latency cost | Implementation effort | Best for |
|----------|------------|--------------|----------------------|----------|
| **Hybrid BM25+vector** | +5–15% NDCG | +10–30ms | Low — FTS index + RRF | Exact terms, identifiers, rare words |
| **HyDE** | +10–30% recall | +200–500ms (LLM call) | Medium — new embedder wrapper | Factual Q&A, knowledge retrieval |
| **Cross-encoder rerank** | +20–40% NDCG@5 | +50–200ms | Medium — new Reranker trait | Precision-critical applications |
| **MMR** | 0% recall, ↑ diversity | Negligible | Low — pure Rust, no deps | Avoid redundant results |
| **Chunking** | +10–20% for long docs | Negligible | Medium — Chunker + parent fetch | Long documents, paragraph retrieval |
| **Query decomposition** | +20–40% multi-hop | +400ms (LLM call) | High — LLM + parallel retrieval | Complex, multi-part questions |

**Recommended implementation order for this codebase:**

1. **Hybrid BM25+vector** — highest impact per line of code, infrastructure already exists
2. **Cross-encoder rerank** — add `Reranker` trait, implement with a local Ollama model
3. **Chunking** — add `Chunker` to pipeline, `get_by_ids` to LanceDbStore
4. **HyDE** — add `HydeEmbedder` wrapper, config flag to toggle it
5. **MMR** — add to `common`, wire up `--mmr-lambda` CLI flag

Decomposition has the highest complexity for the gain; implement it only if multi-hop questions are a primary use case.
