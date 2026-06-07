use async_trait::async_trait;
use blob::InMemoryBackend;
use bytes::Bytes;
use common::{BlobBackend, EmbeddedDoc, Result, ScoredDoc, SearchRequest, SearchSource,
             SearchType, TurboError, VectorStore};
use compressor::QuantizedIndex;
use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
use tokio::sync::RwLock;

/// In-memory ANN index backed by turbovec TurboQuant.
/// Persists to / loads from a BlobBackend for durability.
/// Stores a text map alongside the index so results can include document text.
pub struct TurboVecStore {
    index: Arc<RwLock<QuantizedIndex>>,
    /// id → document text, kept in memory for result enrichment.
    texts: Arc<RwLock<HashMap<u64, String>>>,
    blob: Arc<dyn BlobBackend>,
    table_name: String,
    warm: Arc<AtomicBool>,
    count: Arc<AtomicUsize>,
    dim: usize,
    bits: usize,
}

impl TurboVecStore {
    const INDEX_KEY_SUFFIX: &'static str = "index.tq";
    const META_KEY_SUFFIX: &'static str = "meta.json";

    pub fn new_in_memory(table_name: impl Into<String>, dim: usize, bits: usize) -> Self {
        Self::with_blob(table_name, Arc::new(InMemoryBackend::new()), dim, bits)
    }

    pub fn with_blob(
        table_name: impl Into<String>,
        blob: Arc<dyn BlobBackend>,
        dim: usize,
        bits: usize,
    ) -> Self {
        let name = table_name.into();
        Self {
            index: Arc::new(RwLock::new(QuantizedIndex::new(dim, bits))),
            texts: Arc::new(RwLock::new(HashMap::new())),
            blob,
            table_name: name,
            warm: Arc::new(AtomicBool::new(false)),
            count: Arc::new(AtomicUsize::new(0)),
            dim,
            bits,
        }
    }

    fn index_key(&self) -> String {
        format!("turbovec/{}/{}", self.table_name, Self::INDEX_KEY_SUFFIX)
    }

    fn meta_key(&self) -> String {
        format!("turbovec/{}/{}", self.table_name, Self::META_KEY_SUFFIX)
    }

    /// Persist index + metadata (count, dim, bits, doc texts) to blob storage.
    pub async fn flush(&self) -> Result<()> {
        let bytes = self.index.read().await.to_bytes()
            .map_err(|e| TurboError::Store(e.to_string()))?;
        self.blob.put(&self.index_key(), bytes).await?;

        let count = self.count.load(Ordering::Relaxed);
        // Serialize text map as {"id_str": "text", ...} alongside count/dim/bits.
        let texts_snapshot = self.texts.read().await;
        let texts_obj: serde_json::Map<String, serde_json::Value> = texts_snapshot
            .iter()
            .map(|(id, t)| (id.to_string(), serde_json::Value::String(t.clone())))
            .collect();
        let meta = serde_json::json!({
            "count": count, "dim": self.dim, "bits": self.bits,
            "texts": texts_obj
        });
        self.blob.put(&self.meta_key(), Bytes::from(meta.to_string())).await
    }

    /// Load index (count, text map) from blob storage.
    pub async fn load_from_blob(&self) -> Result<()> {
        let bytes = self.blob.get(&self.index_key()).await?;
        let loaded = QuantizedIndex::from_bytes(bytes, self.dim, self.bits)
            .map_err(|e| TurboError::Store(e.to_string()))?;
        *self.index.write().await = loaded;

        if let Ok(meta_bytes) = self.blob.get(&self.meta_key()).await {
            if let Ok(meta) = serde_json::from_slice::<serde_json::Value>(&meta_bytes) {
                if let Some(n) = meta["count"].as_u64() {
                    self.count.store(n as usize, Ordering::SeqCst);
                }
                if let Some(obj) = meta["texts"].as_object() {
                    let mut texts = self.texts.write().await;
                    for (k, v) in obj {
                        if let (Ok(id), Some(text)) = (k.parse::<u64>(), v.as_str()) {
                            texts.insert(id, text.to_string());
                        }
                    }
                }
            }
        }
        self.warm.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// True once the index has been loaded/built and is ready to serve queries.
    pub fn is_warm(&self) -> bool {
        self.warm.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl VectorStore for TurboVecStore {
    async fn upsert(&self, docs: &[EmbeddedDoc]) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let ids: Vec<u64> = docs.iter().map(|d| d.id).collect();
        let vecs: Vec<Vec<f32>> = docs.iter().map(|d| d.embedding.clone()).collect();
        self.index.write().await.add_batch(&ids, &vecs);
        // Populate text map for result enrichment.
        let mut texts = self.texts.write().await;
        for doc in docs {
            texts.insert(doc.id, doc.text.clone());
        }
        drop(texts);
        self.count.fetch_add(docs.len(), Ordering::Relaxed);
        self.warm.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn search(&self, request: &SearchRequest<'_>) -> Result<Vec<ScoredDoc>> {
        if !matches!(request.search_type, SearchType::Vector) {
            return Err(TurboError::UnsupportedSearchType(
                "TurboVecStore only supports SearchType::Vector".into(),
            ));
        }
        let hits = self.index.read().await.search(request.query_embedding, request.k);
        let texts = self.texts.read().await;
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
    }

    async fn delete(&self, id: u64) -> Result<()> {
        self.index.write().await.remove(id);
        self.texts.write().await.remove(&id);
        let _ = self.count.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            Some(n.saturating_sub(1))
        });
        Ok(())
    }

    async fn doc_count(&self) -> Result<usize> {
        Ok(self.count.load(Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::SearchRequest;

    fn make_vecs(n: usize, dim: usize) -> Vec<EmbeddedDoc> {
        (0..n as u64)
            .map(|id| EmbeddedDoc {
                id,
                text: format!("doc {id}"),
                embedding: vec![id as f32 / n as f32; dim],
                metadata: Default::default(),
            })
            .collect()
    }

    #[tokio::test]
    async fn upsert_and_search() {
        let store = TurboVecStore::new_in_memory("test", 64, 4);
        let docs = make_vecs(20, 64);
        store.upsert(&docs).await.unwrap();
        let req = SearchRequest::vector(&docs[0].embedding, 5);
        let results = store.search(&req).await.unwrap();
        assert_eq!(results.len(), 5);
    }

    #[tokio::test]
    async fn bm25_returns_unsupported_error() {
        let store = TurboVecStore::new_in_memory("test", 64, 4);
        let q = vec![0.0f32; 64];
        let req = SearchRequest::bm25(&q, "hello", 5);
        assert!(matches!(
            store.search(&req).await,
            Err(TurboError::UnsupportedSearchType(_))
        ));
    }

    #[tokio::test]
    async fn flush_and_reload() {
        let blob = Arc::new(InMemoryBackend::new());
        let store1 = TurboVecStore::with_blob("tbl", blob.clone(), 64, 4);
        let docs = make_vecs(10, 64);
        store1.upsert(&docs).await.unwrap();
        store1.flush().await.unwrap();

        let store2 = TurboVecStore::with_blob("tbl", blob, 64, 4);
        store2.load_from_blob().await.unwrap();
        assert!(store2.is_warm());
        let req = SearchRequest::vector(&docs[0].embedding, 3);
        let results = store2.search(&req).await.unwrap();
        assert_eq!(results.len(), 3);
    }
}
