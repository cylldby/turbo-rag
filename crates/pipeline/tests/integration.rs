//! Integration tests for the ingestion pipeline.
//!
//! These tests require a live WireMock instance at http://localhost:8080
//! with a stub on POST /v1/embeddings returning a 768-dim embedding.
//!
//! Run with:
//!   cargo test -p pipeline --features integration -- --nocapture

#[cfg(feature = "integration")]
mod integration {
    use blob::InMemoryBackend;
    use common::{Document, SearchMode, VectorStore};
    use embedder::OpenAICompatBackend;
    use pipeline::IngestionPipeline;
    use std::collections::HashMap;
    use std::sync::Arc;
    use store::{HybridStore, LanceDbStore, TurboVecStore};

    fn make_docs(n: usize) -> Vec<Document> {
        (0..n as u64)
            .map(|i| {
                let mut metadata = HashMap::new();
                metadata.insert("category".to_string(), format!("cat_{}", i % 5));
                metadata.insert("source".to_string(), "integration_test".to_string());
                Document {
                    id: i,
                    text: format!(
                        "This is synthetic document number {} about topic {} with some additional text for context.",
                        i,
                        i % 10
                    ),
                    metadata,
                }
            })
            .collect()
    }

    #[tokio::test]
    async fn pipeline_500_docs() {
        let wiremock_base = "http://localhost:8080/v1";

        // OpenAI-compat backend pointing at WireMock, dim=768
        let embedder = Arc::new(OpenAICompatBackend::new(
            wiremock_base,
            None::<String>,
            "mock-embed",
            768,
        ));

        // Temp dir for LanceDB
        let dir = tempfile::tempdir().expect("failed to create tempdir");

        // Build HybridStore: turbovec hot + lancedb cold
        let hot = Arc::new(TurboVecStore::with_blob(
            "test",
            Arc::new(InMemoryBackend::new()),
            768,
            4,
        ));
        let cold = Arc::new(
            LanceDbStore::new(dir.path().to_str().unwrap(), "docs", 768)
                .await
                .expect("failed to create LanceDbStore"),
        );
        let store = Arc::new(HybridStore::new(
            hot.clone(),
            cold.clone(),
            SearchMode::Auto,
        ));

        // Use batch_size=1 so WireMock's single-embedding stub always satisfies the request
        let pipeline = IngestionPipeline::new(embedder, store.clone(), 1);

        let docs = make_docs(500);
        let stats = pipeline.run(docs).await.expect("pipeline failed");

        assert_eq!(stats.total_docs, 500, "expected 500 docs ingested");
        assert_eq!(
            hot.doc_count().await.unwrap(),
            500,
            "hot (turbovec) store should have 500 docs"
        );
        assert_eq!(
            cold.doc_count().await.unwrap(),
            500,
            "cold (lancedb) store should have 500 docs"
        );
        // 768-dim, 4-bit → 8x compression ratio; allow slight floating-point slack
        assert!(
            stats.compression_ratio >= 7.9,
            "expected compression_ratio >= 7.9, got {}",
            stats.compression_ratio
        );
    }
}
