use async_trait::async_trait;
use bytes::Bytes;
use common::{BlobBackend, Result, TurboError};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ─── InMemoryBackend ──────────────────────────────────────────────────────────

/// In-memory blob store for unit tests. Not thread-safe across processes.
#[derive(Clone, Default)]
pub struct InMemoryBackend {
    data: Arc<RwLock<HashMap<String, Bytes>>>,
}

impl InMemoryBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl BlobBackend for InMemoryBackend {
    async fn put(&self, key: &str, data: Bytes) -> Result<()> {
        self.data.write().await.insert(key.to_string(), data);
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Bytes> {
        self.data
            .read()
            .await
            .get(key)
            .cloned()
            .ok_or_else(|| TurboError::NotFound(key.to_string()))
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.data.write().await.remove(key);
        Ok(())
    }

    async fn list_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        let keys = self
            .data
            .read()
            .await
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        Ok(keys)
    }
}

// ─── LocalFsBackend ───────────────────────────────────────────────────────────

/// Local filesystem blob backend. Keys map directly to file paths under `base_dir`.
pub struct LocalFsBackend {
    base_dir: std::path::PathBuf,
}

impl LocalFsBackend {
    pub fn new(base_dir: impl Into<std::path::PathBuf>) -> anyhow::Result<Self> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(&base_dir)?;
        Ok(Self { base_dir })
    }
}

#[async_trait]
impl BlobBackend for LocalFsBackend {
    async fn put(&self, key: &str, data: Bytes) -> Result<()> {
        let path = self.base_dir.join(key);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(TurboError::Io)?;
        }
        tokio::fs::write(&path, data).await.map_err(TurboError::Io)
    }

    async fn get(&self, key: &str) -> Result<Bytes> {
        let path = self.base_dir.join(key);
        match tokio::fs::read(&path).await {
            Ok(data) => Ok(Bytes::from(data)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(TurboError::NotFound(key.to_string()))
            }
            Err(e) => Err(TurboError::Io(e)),
        }
    }

    async fn delete(&self, key: &str) -> Result<()> {
        tokio::fs::remove_file(self.base_dir.join(key))
            .await
            .map_err(TurboError::Io)
    }

    async fn list_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        let mut keys = Vec::new();
        let mut rd = tokio::fs::read_dir(&self.base_dir)
            .await
            .map_err(TurboError::Io)?;
        while let Some(entry) = rd.next_entry().await.map_err(TurboError::Io)? {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(prefix) {
                keys.push(name);
            }
        }
        Ok(keys)
    }
}

// ─── S3Backend ────────────────────────────────────────────────────────────────

#[cfg(feature = "blob-s3")]
mod s3;
#[cfg(feature = "blob-s3")]
pub use s3::S3Backend;

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn inmemory_put_get_identity() {
        let store = InMemoryBackend::new();
        let data = Bytes::from_static(b"hello turbo-rag");
        store.put("test/key", data.clone()).await.unwrap();
        let got = store.get("test/key").await.unwrap();
        assert_eq!(got, data);
    }

    #[tokio::test]
    async fn inmemory_missing_key_returns_not_found() {
        let store = InMemoryBackend::new();
        let err = store.get("does/not/exist").await.unwrap_err();
        assert!(matches!(err, TurboError::NotFound(_)));
    }

    #[tokio::test]
    async fn inmemory_delete_removes_key() {
        let store = InMemoryBackend::new();
        store.put("k", Bytes::from_static(b"v")).await.unwrap();
        store.delete("k").await.unwrap();
        assert!(matches!(store.get("k").await, Err(TurboError::NotFound(_))));
    }

    #[tokio::test]
    async fn inmemory_list_prefix() {
        let store = InMemoryBackend::new();
        store.put("turbo/a", Bytes::new()).await.unwrap();
        store.put("turbo/b", Bytes::new()).await.unwrap();
        store.put("lance/c", Bytes::new()).await.unwrap();
        let keys = store.list_prefix("turbo/").await.unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"turbo/a".to_string()));
        assert!(keys.contains(&"turbo/b".to_string()));
    }

    #[tokio::test]
    async fn inmemory_exists() {
        let store = InMemoryBackend::new();
        assert!(!store.exists("x").await.unwrap());
        store.put("x", Bytes::from_static(b"1")).await.unwrap();
        assert!(store.exists("x").await.unwrap());
    }
}
