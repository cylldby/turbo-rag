//! End-to-end recall tests for the full RAG pipeline.
//!
//! These tests load the fixture corpus (50 docs) and queries (20 queries),
//! ingest via OpenAICompatBackend (WireMock) + HybridStore, and verify
//! recall@10 >= 0.5.
//!
//! NOTE: WireMock returns a constant embedding for all docs, so cosine
//! similarity is identical for all results. The test mainly verifies that
//! the pipeline runs end-to-end and returns k results without errors.
//! Real recall testing requires a live embedding model (gated by OLLAMA_AVAILABLE=1).
//!
//! Run with:
//!   cargo test -p pipeline --features e2e -- --nocapture

#[cfg(feature = "e2e")]
mod e2e {
    use blob::InMemoryBackend;
    use common::{Document, EmbeddingBackend, SearchMode, SearchRequest, VectorStore};
    use embedder::OpenAICompatBackend;
    use pipeline::IngestionPipeline;
    use serde::Deserialize;
    use std::collections::HashMap;
    use std::sync::Arc;
    use store::{HybridStore, LanceDbStore, TurboVecStore};

    #[derive(Debug, Deserialize)]
    struct CorpusDoc {
        id: u64,
        text: String,
        #[serde(default)]
        metadata: HashMap<String, String>,
    }

    #[derive(Debug, Deserialize)]
    struct QueryRecord {
        #[allow(dead_code)]
        id: u64,
        text: String,
        relevant_ids: Vec<u64>,
        #[allow(dead_code)]
        category: String,
    }

    fn load_corpus(path: &str) -> Vec<Document> {
        let content = std::fs::read_to_string(path).expect("failed to read corpus.jsonl");
        content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|line| {
                let rec: CorpusDoc =
                    serde_json::from_str(line).expect("failed to parse corpus line");
                Document {
                    id: rec.id,
                    text: rec.text,
                    metadata: rec.metadata,
                }
            })
            .collect()
    }

    fn load_queries(path: &str) -> Vec<QueryRecord> {
        let content = std::fs::read_to_string(path).expect("failed to read queries.jsonl");
        content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("failed to parse query line"))
            .collect()
    }

    #[tokio::test]
    async fn recall_on_fixture_corpus() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        // Walk up from crates/pipeline to workspace root
        let workspace_root = std::path::Path::new(manifest_dir)
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let corpus_path = workspace_root.join("data/fixtures/corpus.jsonl");
        let queries_path = workspace_root.join("data/fixtures/queries.jsonl");

        let docs = load_corpus(corpus_path.to_str().unwrap());
        let queries = load_queries(queries_path.to_str().unwrap());

        assert_eq!(docs.len(), 50, "expected 50 corpus docs");
        assert_eq!(queries.len(), 20, "expected 20 queries");

        let wiremock_base = "http://localhost:8080/v1";
        let embedder = Arc::new(OpenAICompatBackend::new(
            wiremock_base,
            None::<String>,
            "mock-embed",
            768,
        ));

        let dir = tempfile::tempdir().expect("failed to create tempdir");
        let hot = Arc::new(TurboVecStore::with_blob(
            "e2e_test",
            Arc::new(InMemoryBackend::new()),
            768,
            4,
        ));
        let cold = Arc::new(
            LanceDbStore::new(dir.path().to_str().unwrap(), "e2e_docs", 768)
                .await
                .expect("failed to create LanceDbStore"),
        );
        let store = Arc::new(HybridStore::new(hot, cold, SearchMode::Auto));

        // batch_size=1 so WireMock's single-embedding stub always satisfies the request
        let pipeline = IngestionPipeline::new(embedder.clone(), store.clone(), 1);
        let stats = pipeline.run(docs).await.expect("ingestion pipeline failed");
        assert_eq!(stats.total_docs, 50, "expected 50 docs ingested");

        // Evaluate recall@10 for each query
        let k = 10usize;
        let mut hit_count = 0usize;

        for query in &queries {
            // Embed the query text (same WireMock stub returns constant vector)
            let query_vec = embedder
                .embed_one(&query.text)
                .await
                .expect("failed to embed query");

            let req = SearchRequest::vector(&query_vec, k);
            let results = store
                .search(&req)
                .await
                .expect("search failed");

            assert!(
                results.len() <= k,
                "search returned more than k={k} results: {}",
                results.len()
            );

            // Check whether any relevant_id appears in the top-k
            let result_ids: std::collections::HashSet<u64> =
                results.iter().map(|r| r.id).collect();
            let hit = query
                .relevant_ids
                .iter()
                .any(|rid| result_ids.contains(rid));
            if hit {
                hit_count += 1;
            }
        }

        let recall_at_10 = hit_count as f64 / queries.len() as f64;
        eprintln!(
            "recall@10 = {}/{} = {:.2}",
            hit_count,
            queries.len(),
            recall_at_10
        );

        // WireMock returns constant embeddings so all docs are equally similar.
        // The store returns the first k docs it encounters; with 50 docs and k=10
        // the hit rate depends on whether any relevant doc falls in the first 10.
        // We use a very lenient threshold here; real recall requires a live model.
        assert!(
            recall_at_10 >= 0.5,
            "recall@10 {:.2} < 0.5 — pipeline may not be returning results correctly",
            recall_at_10
        );
    }
}
