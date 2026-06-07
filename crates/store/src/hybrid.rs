use async_trait::async_trait;
use common::{EmbeddedDoc, Result, ScoredDoc, SearchMode, SearchRequest, VectorStore};
use std::sync::Arc;

#[cfg(feature = "store-lance")]
use crate::lance_store::LanceDbStore;
use crate::turbovec_store::TurboVecStore;

/// How to initialise the turbovec hot index at startup.
#[derive(Debug, Clone, Default)]
pub enum LoadStrategy {
    /// Download serialized index from blob storage. Fast, requires prior flush.
    #[default]
    FromBlob,
    /// Rebuild turbovec in-memory from LanceDB (source of truth). Slower but always consistent.
    #[cfg(feature = "store-lance")]
    RebuildFromLance,
    /// Start serving from LanceDB immediately; build turbovec in background. Zero downtime.
    Lazy,
}

/// Composes TurboVecStore (hot, in-memory SIMD) + LanceDbStore (cold, persistent).
/// Write path always updates both stores concurrently.
/// Read path is governed by SearchMode.
pub struct HybridStore {
    pub hot: Arc<TurboVecStore>,
    #[cfg(feature = "store-lance")]
    pub cold: Arc<LanceDbStore>,
    pub mode: SearchMode,
}

impl HybridStore {
    #[cfg(feature = "store-lance")]
    pub fn new(hot: Arc<TurboVecStore>, cold: Arc<LanceDbStore>, mode: SearchMode) -> Self {
        Self { hot, cold, mode }
    }
}

#[async_trait]
impl VectorStore for HybridStore {
    async fn upsert(&self, docs: &[EmbeddedDoc]) -> Result<()> {
        #[cfg(feature = "store-lance")]
        {
            tokio::try_join!(self.hot.upsert(docs), self.cold.upsert(docs))?;
        }
        #[cfg(not(feature = "store-lance"))]
        {
            self.hot.upsert(docs).await?;
        }
        Ok(())
    }

    async fn search(&self, request: &SearchRequest<'_>) -> Result<Vec<ScoredDoc>> {
        #[cfg(feature = "store-lance")]
        {
            return match &self.mode {
                SearchMode::Hot => self.hot.search(request).await,
                SearchMode::Cold => self.cold.search(request).await,
                SearchMode::Auto => {
                    if self.hot.is_warm() {
                        self.hot.search(request).await
                    } else {
                        self.cold.search(request).await
                    }
                }
                SearchMode::Race => {
                    tokio::select! {
                        r = self.hot.search(request) => r,
                        r = self.cold.search(request) => r,
                    }
                }
                SearchMode::Federated(merge_k) => {
                    let (hot_r, cold_r) =
                        tokio::join!(self.hot.search(request), self.cold.search(request));
                    let mut all: Vec<ScoredDoc> = Vec::new();
                    if let Ok(mut r) = hot_r {
                        all.append(&mut r);
                    }
                    if let Ok(mut r) = cold_r {
                        all.append(&mut r);
                    }
                    all.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let mut seen = std::collections::HashSet::new();
                    all.retain(|doc| seen.insert(doc.id));
                    all.truncate(*merge_k);
                    Ok(all)
                }
            };
        }
        #[cfg(not(feature = "store-lance"))]
        match &self.mode {
            SearchMode::Hot => self.hot.search(request).await,
            _ => Err(TurboError::Store(
                "store-lance feature required for Cold/Auto/Race/Federated modes".into(),
            )),
        }
    }

    async fn delete(&self, id: u64) -> Result<()> {
        #[cfg(feature = "store-lance")]
        tokio::try_join!(self.hot.delete(id), self.cold.delete(id))?;
        #[cfg(not(feature = "store-lance"))]
        self.hot.delete(id).await?;
        Ok(())
    }

    async fn doc_count(&self) -> Result<usize> {
        #[cfg(feature = "store-lance")]
        return self.cold.doc_count().await;
        #[cfg(not(feature = "store-lance"))]
        return self.hot.doc_count().await;
    }
}

#[cfg(all(feature = "store-lance", test))]
mod tests {
    use super::*;
    use blob::InMemoryBackend;
    use common::{EmbeddedDoc, SearchRequest, SearchSource};

    async fn make_hybrid(dim: usize) -> (HybridStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap().to_string();
        let hot = Arc::new(TurboVecStore::with_blob(
            "h",
            Arc::new(InMemoryBackend::new()),
            dim,
            4,
        ));
        let cold = Arc::new(LanceDbStore::new(&uri, "docs", dim).await.unwrap());
        let store = HybridStore::new(hot, cold, SearchMode::Auto);
        (store, dir)
    }

    fn docs(n: usize, dim: usize) -> Vec<EmbeddedDoc> {
        (0..n as u64)
            .map(|id| EmbeddedDoc {
                id,
                text: format!("doc {id}"),
                embedding: vec![id as f32 / n as f32 + 0.001; dim],
                metadata: Default::default(),
            })
            .collect()
    }

    #[tokio::test]
    async fn dual_write_populates_both_stores() {
        let (store, _dir) = make_hybrid(32).await;
        store.upsert(&docs(50, 32)).await.unwrap();
        // hot count via Arc
        assert_eq!(
            store.hot.doc_count().await.unwrap(),
            50,
            "turbovec should have 50 docs"
        );
        // cold count via hybrid (delegates to LanceDB)
        assert_eq!(
            store.doc_count().await.unwrap(),
            50,
            "lancedb should have 50 docs"
        );
    }

    #[tokio::test]
    async fn auto_mode_serves_from_hot_when_warm() {
        let (store, _dir) = make_hybrid(32).await;
        store.upsert(&docs(20, 32)).await.unwrap();
        assert!(store.hot.is_warm(), "hot store should be warm after upsert");
        let query = vec![0.5f32; 32];
        let req = SearchRequest::vector(&query, 5);
        let results = store.search(&req).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(
            results[0].source,
            SearchSource::TurboVec,
            "auto mode should use turbovec when warm"
        );
    }

    #[tokio::test]
    async fn cold_mode_serves_from_lancedb() {
        let dim = 32;
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap().to_string();
        let hot = Arc::new(TurboVecStore::with_blob(
            "h",
            Arc::new(InMemoryBackend::new()),
            dim,
            4,
        ));
        let cold = Arc::new(LanceDbStore::new(&uri, "docs", dim).await.unwrap());
        let store = HybridStore::new(hot, cold, SearchMode::Cold);
        store.upsert(&docs(20, dim)).await.unwrap();
        let query = vec![0.5f32; dim];
        let req = SearchRequest::vector(&query, 3);
        let results = store.search(&req).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(
            results[0].source,
            SearchSource::LanceDb,
            "cold mode must use lancedb"
        );
    }

    #[tokio::test]
    async fn delete_removes_from_both_stores() {
        let (store, _dir) = make_hybrid(16).await;
        store.upsert(&docs(10, 16)).await.unwrap();
        assert_eq!(store.doc_count().await.unwrap(), 10);
        store.delete(0).await.unwrap();
        assert_eq!(store.doc_count().await.unwrap(), 9);
        assert_eq!(store.hot.doc_count().await.unwrap(), 9);
    }
}
