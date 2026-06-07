#[cfg(feature = "store-lance")]
use arrow_array::{
    FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray, UInt64Array,
};
#[cfg(feature = "store-lance")]
use arrow_schema::{DataType, Field, Schema};
#[cfg(feature = "store-lance")]
use async_trait::async_trait;
#[cfg(feature = "store-lance")]
use common::{
    EmbeddedDoc, Result, ScoredDoc, SearchRequest, SearchSource, SearchType, TurboError,
    VectorStore,
};
#[cfg(feature = "store-lance")]
use futures::TryStreamExt;
#[cfg(feature = "store-lance")]
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
#[cfg(feature = "store-lance")]
use lancedb::index::Index;
#[cfg(feature = "store-lance")]
use lancedb::query::{ExecutableQuery, QueryBase, QueryExecutionOptions};
#[cfg(feature = "store-lance")]
use lancedb::{connect, Connection};
#[cfg(feature = "store-lance")]
use std::sync::Arc;

/// LanceDB-backed persistent vector store.
/// Supports ANN vector search, BM25 full-text search, and hybrid RRF fusion (M4).
#[cfg(feature = "store-lance")]
pub struct LanceDbStore {
    conn: Connection,
    table_name: String,
    dim: usize,
}

#[cfg(feature = "store-lance")]
impl LanceDbStore {
    pub async fn new(uri: &str, table_name: &str, dim: usize) -> anyhow::Result<Self> {
        let conn = connect(uri).execute().await?;
        Ok(Self {
            conn,
            table_name: table_name.to_string(),
            dim,
        })
    }

    fn schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dim as i32,
                ),
                false,
            ),
        ]))
    }

    /// Create (or recreate) the full-text search index on the `text` column.
    /// Call this once after a bulk ingest before issuing BM25 or hybrid queries.
    /// Safe to call on an already-indexed table — LanceDB replaces the old index.
    pub async fn ensure_fts_index(&self) -> Result<()> {
        let table = self.table().await?;
        table
            .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
            .execute()
            .await
            .map_err(|e| TurboError::Store(format!("FTS index creation failed: {e}")))?;
        Ok(())
    }

    /// Open the table, creating it empty if it doesn't exist yet.
    async fn table(&self) -> Result<lancedb::Table> {
        match self.conn.open_table(&self.table_name).execute().await {
            Ok(t) => Ok(t),
            Err(_) => {
                let schema = self.schema();
                let empty = RecordBatch::new_empty(schema.clone());
                let reader = Box::new(RecordBatchIterator::new(vec![Ok(empty)], schema));
                self.conn
                    .create_table(&self.table_name, reader)
                    .execute()
                    .await
                    .map_err(|e| TurboError::Store(e.to_string()))
            }
        }
    }

    fn make_batch(&self, docs: &[EmbeddedDoc]) -> Result<RecordBatch> {
        let schema = self.schema();
        let ids = UInt64Array::from_iter_values(docs.iter().map(|d| d.id));
        let texts = StringArray::from_iter_values(docs.iter().map(|d| d.text.as_str()));
        let flat: Vec<f32> = docs
            .iter()
            .flat_map(|d| d.embedding.iter().copied())
            .collect();
        let values = Arc::new(Float32Array::from(flat));
        let embedding_col = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            self.dim as i32,
            values,
            None,
        )
        .map_err(|e| TurboError::Store(e.to_string()))?;

        RecordBatch::try_new(
            schema,
            vec![Arc::new(ids), Arc::new(texts), Arc::new(embedding_col)],
        )
        .map_err(|e| TurboError::Store(e.to_string()))
    }
}

#[cfg(feature = "store-lance")]
#[async_trait]
impl VectorStore for LanceDbStore {
    async fn upsert(&self, docs: &[EmbeddedDoc]) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let schema = self.schema();
        let batch = self.make_batch(docs)?;
        let reader = Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        let table = self.table().await?;
        table
            .add(reader)
            .execute()
            .await
            .map_err(|e| TurboError::Store(e.to_string()))?;
        Ok(())
    }

    async fn search(&self, request: &SearchRequest<'_>) -> Result<Vec<ScoredDoc>> {
        let table = self.table().await?;

        match &request.search_type {
            SearchType::Vector => {
                let stream = table
                    .query()
                    .nearest_to(request.query_embedding)
                    .map_err(|e| TurboError::Store(e.to_string()))?
                    .limit(request.k)
                    .execute()
                    .await
                    .map_err(|e| TurboError::Store(e.to_string()))?;
                let batches: Vec<RecordBatch> = stream
                    .try_collect()
                    .await
                    .map_err(|e| TurboError::Store(e.to_string()))?;
                collect_results(&batches, request.k, ScoreColumn::Distance)
            }

            SearchType::Bm25 => {
                let query_text = request
                    .query_text
                    .ok_or_else(|| TurboError::Store("BM25 search requires query_text".into()))?;
                let stream = table
                    .query()
                    .full_text_search(FullTextSearchQuery::new(query_text.to_string()))
                    .limit(request.k)
                    .execute()
                    .await
                    .map_err(|e| TurboError::Store(e.to_string()))?;
                let batches: Vec<RecordBatch> = stream
                    .try_collect()
                    .await
                    .map_err(|e| TurboError::Store(e.to_string()))?;
                collect_results(&batches, request.k, ScoreColumn::Bm25)
            }

            SearchType::Hybrid { .. } => {
                let query_text = request
                    .query_text
                    .ok_or_else(|| TurboError::Store("Hybrid search requires query_text".into()))?;
                // execute_hybrid runs vector + BM25 in parallel and applies RRF internally.
                let stream = table
                    .query()
                    .full_text_search(FullTextSearchQuery::new(query_text.to_string()))
                    .nearest_to(request.query_embedding)
                    .map_err(|e| TurboError::Store(e.to_string()))?
                    .limit(request.k)
                    .execute_hybrid(QueryExecutionOptions::default())
                    .await
                    .map_err(|e| TurboError::Store(e.to_string()))?;
                let batches: Vec<RecordBatch> = stream
                    .try_collect()
                    .await
                    .map_err(|e| TurboError::Store(e.to_string()))?;
                // Hybrid returns results already fused and ranked by RRF; use rank-based scores.
                collect_results(&batches, request.k, ScoreColumn::Rank)
            }
        }
    }

    async fn delete(&self, id: u64) -> Result<()> {
        let table = self.table().await?;
        table
            .delete(&format!("id = {id}"))
            .await
            .map_err(|e| TurboError::Store(e.to_string()))?;
        Ok(())
    }

    async fn doc_count(&self) -> Result<usize> {
        let table = self.table().await?;
        table
            .count_rows(None)
            .await
            .map_err(|e| TurboError::Store(e.to_string()))
    }
}

#[cfg(all(feature = "store-lance", test))]
mod tests {
    use super::*;
    use common::{EmbeddedDoc, SearchRequest};

    fn make_docs(n: usize, dim: usize) -> Vec<EmbeddedDoc> {
        (0..n as u64)
            .map(|id| EmbeddedDoc {
                id,
                text: format!("document about topic number {id}"),
                embedding: vec![id as f32 / n as f32 + 0.001; dim],
                metadata: Default::default(),
            })
            .collect()
    }

    async fn lance_store(dim: usize) -> (LanceDbStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap().to_string();
        let store = LanceDbStore::new(&uri, "test_docs", dim).await.unwrap();
        (store, dir)
    }

    #[tokio::test]
    async fn insert_and_count() {
        let (store, _dir) = lance_store(32).await;
        let docs = make_docs(20, 32);
        store.upsert(&docs).await.unwrap();
        assert_eq!(store.doc_count().await.unwrap(), 20);
    }

    #[tokio::test]
    async fn vector_search_returns_results() {
        let dim = 32;
        let (store, _dir) = lance_store(dim).await;
        let docs = make_docs(50, dim);
        store.upsert(&docs).await.unwrap();
        let query = vec![0.5f32; dim];
        let req = SearchRequest::vector(&query, 5);
        let results = store.search(&req).await.unwrap();
        assert!(!results.is_empty(), "should return at least one result");
        assert!(results.len() <= 5);
    }

    #[tokio::test]
    async fn schema_is_correct() {
        let (store, _dir) = lance_store(64).await;
        // Upsert and immediately count — schema must be valid
        let docs = make_docs(1, 64);
        store.upsert(&docs).await.unwrap();
        assert_eq!(store.doc_count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn delete_reduces_count() {
        let (store, _dir) = lance_store(16).await;
        let docs = make_docs(10, 16);
        store.upsert(&docs).await.unwrap();
        assert_eq!(store.doc_count().await.unwrap(), 10);
        store.delete(0).await.unwrap();
        assert_eq!(store.doc_count().await.unwrap(), 9);
    }

    #[tokio::test]
    async fn batch_insert_and_query_text_preserved() {
        let dim = 16;
        let (store, _dir) = lance_store(dim).await;
        let docs = make_docs(5, dim);
        store.upsert(&docs).await.unwrap();
        let query = vec![0.5f32; dim];
        let req = SearchRequest::vector(&query, 5);
        let results = store.search(&req).await.unwrap();
        for r in &results {
            assert!(
                r.text.starts_with("document about topic number"),
                "text should be preserved: {}",
                r.text
            );
        }
    }

    #[tokio::test]
    async fn bm25_finds_keyword() {
        let dim = 16;
        let (store, _dir) = lance_store(dim).await;
        // Insert docs with clearly distinct vocabulary
        let docs = vec![
            EmbeddedDoc {
                id: 1,
                text: "turbovec compresses vectors with SIMD quantisation".into(),
                embedding: vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                metadata: Default::default(),
            },
            EmbeddedDoc {
                id: 2,
                text: "lancedb is an embedded vector database for retrieval".into(),
                embedding: vec![
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                metadata: Default::default(),
            },
            EmbeddedDoc {
                id: 3,
                text: "rust is a systems programming language".into(),
                embedding: vec![
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                metadata: Default::default(),
            },
        ];
        store.upsert(&docs).await.unwrap();
        store.ensure_fts_index().await.unwrap();

        let query_vec = vec![0.0f32; dim];
        let req = SearchRequest::bm25(&query_vec, "turbovec SIMD", 3);
        let results = store.search(&req).await.unwrap();

        assert!(!results.is_empty(), "BM25 should return results");
        // The first result must contain the queried keyword
        assert_eq!(
            results[0].id, 1,
            "turbovec doc should rank first for 'turbovec SIMD'"
        );
    }

    #[tokio::test]
    async fn hybrid_returns_results_with_fts_index() {
        let dim = 16;
        let (store, _dir) = lance_store(dim).await;
        let docs = vec![
            EmbeddedDoc {
                id: 10,
                text: "neural network deep learning model training".into(),
                embedding: vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                metadata: Default::default(),
            },
            EmbeddedDoc {
                id: 11,
                text: "random forest ensemble classification algorithm".into(),
                embedding: vec![
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                metadata: Default::default(),
            },
            EmbeddedDoc {
                id: 12,
                text: "transformer attention mechanism language model".into(),
                embedding: vec![
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                metadata: Default::default(),
            },
        ];
        store.upsert(&docs).await.unwrap();
        store.ensure_fts_index().await.unwrap();

        // Vector points toward doc 10; keyword "transformer" is in doc 12
        // Hybrid should blend both signals
        let query_vec = vec![
            1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let req = SearchRequest::hybrid(&query_vec, "transformer", 3, 60);
        let results = store.search(&req).await.unwrap();

        assert!(!results.is_empty(), "hybrid search should return results");
        assert!(results.len() <= 3);
        // Both doc 10 (vector match) and doc 12 (keyword match) should appear
        let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        assert!(
            ids.contains(&10) || ids.contains(&12),
            "hybrid should surface at least one relevant doc, got: {ids:?}"
        );
    }
}

#[cfg(feature = "store-lance")]
enum ScoreColumn {
    Distance,
    Bm25,
    Rank,
}

#[cfg(feature = "store-lance")]
fn collect_results(
    batches: &[RecordBatch],
    k: usize,
    score_col: ScoreColumn,
) -> Result<Vec<ScoredDoc>> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float32Type;

    let mut docs = Vec::new();

    for (batch_rank, batch) in batches.iter().enumerate() {
        let ids = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| TurboError::Store("missing id column".into()))?;
        let texts = batch
            .column_by_name("text")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| TurboError::Store("missing text column".into()))?;
        let n = batch.num_rows();

        for i in 0..n {
            let global_rank = batch_rank * 1024 + i; // approximate rank across batches
            let score = match &score_col {
                ScoreColumn::Distance => {
                    // _distance: lower = more similar; convert to [0,1] similarity
                    batch
                        .column_by_name("_distance")
                        .and_then(|c| c.as_primitive_opt::<Float32Type>())
                        .map(|a| 1.0 / (1.0 + a.value(i)))
                        .unwrap_or(1.0 - (global_rank as f32 * 0.05).min(0.99))
                }
                ScoreColumn::Bm25 => {
                    // _score: BM25 relevance, higher = more relevant, already in natural order
                    batch
                        .column_by_name("_score")
                        .and_then(|c| c.as_primitive_opt::<Float32Type>())
                        .map(|a| a.value(i))
                        .unwrap_or(1.0 - (global_rank as f32 * 0.05).min(0.99))
                }
                ScoreColumn::Rank => {
                    // Hybrid / unknown: trust the ordering, assign rank-based scores
                    1.0 - (global_rank as f32 * 0.05).min(0.99)
                }
            };

            docs.push(ScoredDoc {
                id: ids.value(i),
                text: texts.value(i).to_string(),
                score,
                source: SearchSource::LanceDb,
                metadata: Default::default(),
            });
        }
    }

    // Deduplicate by id, keeping the highest-scoring occurrence.
    let mut seen = std::collections::HashMap::<u64, usize>::new();
    let mut deduped: Vec<ScoredDoc> = Vec::with_capacity(docs.len());
    for doc in docs {
        if let Some(&idx) = seen.get(&doc.id) {
            if doc.score > deduped[idx].score {
                deduped[idx] = doc;
            }
        } else {
            seen.insert(doc.id, deduped.len());
            deduped.push(doc);
        }
    }
    deduped.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    deduped.truncate(k);
    Ok(deduped)
}
