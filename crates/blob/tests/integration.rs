//! Integration tests for blob backends.
//! Requires MinIO running at AWS_ENDPOINT_URL (default: http://localhost:9000).
//! Run with: cargo test -p blob --features integration

#[cfg(feature = "integration")]
mod blob_integration {
    use blob::S3Backend;
    use bytes::Bytes;
    use common::BlobBackend;
    use std::sync::Arc;

    fn s3() -> Arc<S3Backend> {
        // Picks up AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY from env.
        // .env.test sets these to point at the local MinIO instance.
        let bucket =
            std::env::var("BLOB_BUCKET").unwrap_or_else(|_| "turbo-rag-dev".to_string());
        Arc::new(S3Backend::from_env(&bucket).expect("S3Backend init failed — is MinIO running?"))
    }

    fn unique_key(test: &str) -> String {
        format!("integration-test/{}/{}", test, uuid())
    }

    fn uuid() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        format!("{:x}{:x}", t.as_secs(), t.subsec_nanos())
    }

    #[tokio::test]
    async fn blob_minio_put_get_byte_identical() {
        let store = s3();
        let key = unique_key("put_get");
        let data = Bytes::from(b"turbo-rag integration test payload".to_vec());
        store.put(&key, data.clone()).await.unwrap();
        let got = store.get(&key).await.unwrap();
        assert_eq!(got, data, "round-tripped bytes must be identical");
        store.delete(&key).await.ok();
    }

    #[tokio::test]
    async fn blob_minio_large_payload() {
        let store = s3();
        let key = unique_key("large");
        let data = Bytes::from(vec![0xABu8; 1_048_576]); // 1 MB
        store.put(&key, data.clone()).await.unwrap();
        let got = store.get(&key).await.unwrap();
        assert_eq!(got.len(), 1_048_576);
        assert_eq!(got[0], 0xAB);
        store.delete(&key).await.ok();
    }

    #[tokio::test]
    async fn blob_minio_missing_key_returns_not_found() {
        use common::TurboError;
        let store = s3();
        let err = store.get("does/not/exist/xyz123").await.unwrap_err();
        assert!(
            matches!(err, TurboError::NotFound(_)),
            "expected NotFound, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn blob_minio_delete_removes_object() {
        use common::TurboError;
        let store = s3();
        let key = unique_key("delete");
        store.put(&key, Bytes::from_static(b"tmp")).await.unwrap();
        store.delete(&key).await.unwrap();
        let err = store.get(&key).await.unwrap_err();
        assert!(matches!(err, TurboError::NotFound(_)));
    }

    #[tokio::test]
    async fn blob_minio_exists_tracks_object_lifecycle() {
        let store = s3();
        let key = unique_key("exists");
        assert!(!store.exists(&key).await.unwrap(), "should not exist before put");
        store.put(&key, Bytes::from_static(b"x")).await.unwrap();
        assert!(store.exists(&key).await.unwrap(), "should exist after put");
        store.delete(&key).await.unwrap();
        assert!(!store.exists(&key).await.unwrap(), "should not exist after delete");
    }

    #[tokio::test]
    async fn blob_minio_list_prefix() {
        let store = s3();
        let prefix = unique_key("list");
        store.put(&format!("{prefix}/a"), Bytes::from_static(b"a")).await.unwrap();
        store.put(&format!("{prefix}/b"), Bytes::from_static(b"b")).await.unwrap();
        store
            .put("integration-test/other/x", Bytes::from_static(b"x"))
            .await
            .unwrap();
        let keys = store.list_prefix(&prefix).await.unwrap();
        assert_eq!(keys.len(), 2, "expected 2 keys with prefix, got {keys:?}");
        store.delete(&format!("{prefix}/a")).await.ok();
        store.delete(&format!("{prefix}/b")).await.ok();
    }
}
