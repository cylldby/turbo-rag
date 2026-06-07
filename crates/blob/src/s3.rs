#[cfg(feature = "blob-s3")]
use async_trait::async_trait;
#[cfg(feature = "blob-s3")]
use bytes::Bytes;
#[cfg(feature = "blob-s3")]
use common::{BlobBackend, Result, TurboError};
#[cfg(feature = "blob-s3")]
use object_store::{aws::AmazonS3Builder, path::Path, ObjectStore};
#[cfg(feature = "blob-s3")]
use std::sync::Arc;

/// S3-compatible blob backend.
/// Set `AWS_ENDPOINT_URL=http://localhost:9000` for MinIO.
/// Omit for real AWS S3.
#[cfg(feature = "blob-s3")]
pub struct S3Backend {
    store: Arc<dyn ObjectStore>,
    #[allow(dead_code)]
    bucket: String,
}

#[cfg(feature = "blob-s3")]
impl S3Backend {
    pub fn from_env(bucket: impl Into<String>) -> anyhow::Result<Self> {
        let bucket = bucket.into();
        let mut builder = AmazonS3Builder::from_env().with_bucket_name(&bucket);
        // Allow path-style URLs required by MinIO.
        if std::env::var("AWS_ENDPOINT_URL").is_ok() {
            builder = builder.with_allow_http(true);
        }
        let store = Arc::new(builder.build()?);
        Ok(Self { store, bucket })
    }
}

#[cfg(feature = "blob-s3")]
#[async_trait]
impl BlobBackend for S3Backend {
    async fn put(&self, key: &str, data: Bytes) -> Result<()> {
        let path = Path::from(key);
        self.store
            .put(&path, data.into())
            .await
            .map_err(|e| TurboError::Blob(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Bytes> {
        let path = Path::from(key);
        match self.store.get(&path).await {
            Ok(result) => result
                .bytes()
                .await
                .map_err(|e| TurboError::Blob(e.to_string())),
            Err(object_store::Error::NotFound { .. }) => Err(TurboError::NotFound(key.to_string())),
            Err(e) => Err(TurboError::Blob(e.to_string())),
        }
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.store
            .delete(&Path::from(key))
            .await
            .map_err(|e| TurboError::Blob(e.to_string()))
    }

    async fn list_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        use futures::TryStreamExt;
        let prefix_path = Path::from(prefix);
        let items = self
            .store
            .list(Some(&prefix_path))
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| TurboError::Blob(e.to_string()))?;
        Ok(items.into_iter().map(|m| m.location.to_string()).collect())
    }
}
