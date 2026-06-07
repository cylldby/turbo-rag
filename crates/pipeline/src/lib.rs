use common::{
    compression_ratio, Document, EmbeddedDoc, EmbeddingBackend, PipelineStats, Result,
    TurboError, VectorStore,
};
use futures::stream::{self, StreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

pub struct IngestionPipeline {
    embedder: Arc<dyn EmbeddingBackend>,
    store: Arc<dyn VectorStore>,
    batch_size: usize,
    /// Max concurrent embedding requests in flight at once.
    concurrency: usize,
    bits: usize,
}

impl IngestionPipeline {
    pub fn new(
        embedder: Arc<dyn EmbeddingBackend>,
        store: Arc<dyn VectorStore>,
        batch_size: usize,
    ) -> Self {
        Self { embedder, store, batch_size, concurrency: 4, bits: 4 }
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    pub fn with_bits(mut self, bits: usize) -> Self {
        self.bits = bits;
        self
    }

    pub async fn run(&self, docs: Vec<Document>) -> Result<PipelineStats> {
        let total = docs.len();
        if total == 0 {
            return Err(TurboError::Pipeline("no documents to ingest".into()));
        }

        let mp = MultiProgress::new();
        let embed_pb = mp.add(ProgressBar::new(total as u64));
        embed_pb.set_style(
            ProgressStyle::with_template(
                "embed  [{bar:35.cyan/blue}] {pos}/{len} docs  {per_sec}",
            )
            .unwrap()
            .progress_chars("█▓░"),
        );

        let start = Instant::now();
        let dim = self.embedder.dim();
        let original_bytes = total * dim * 4;

        // ── Phase 1: preprocess in parallel with rayon ──────────────────────
        // Normalize and trim text on all CPU cores before hitting the embedding API.
        let preprocessed: Vec<(u64, String, _)> = docs
            .par_iter()
            .map(|d| (d.id, d.text.trim().to_string(), d.metadata.clone()))
            .collect();

        // ── Phase 2: embed concurrently with buffered async stream ───────────
        // `buffered(concurrency)` keeps N embedding requests in flight at once.
        // For local fastembed, concurrency=1 is fine (ONNX uses all cores internally).
        // For Ollama / OpenAI, concurrency=4 multiplies throughput.
        let embed_pb_clone = embed_pb.clone();
        let chunks: Vec<Vec<(u64, String, _)>> = preprocessed
            .chunks(self.batch_size)
            .map(|c| c.to_vec())
            .collect();

        let embedded_batches: Vec<Result<Vec<EmbeddedDoc>>> = stream::iter(chunks)
            .map(|chunk| {
                let embedder = self.embedder.clone();
                let pb = embed_pb_clone.clone();
                async move {
                    let texts: Vec<String> = chunk.iter().map(|(_, t, _)| t.clone()).collect();
                    let embeddings = embedder.embed_batch(&texts).await?;
                    let embedded: Vec<EmbeddedDoc> = chunk
                        .into_iter()
                        .zip(embeddings)
                        .map(|((id, text, meta), emb)| EmbeddedDoc {
                            id,
                            text,
                            embedding: emb,
                            metadata: meta,
                        })
                        .collect();
                    pb.inc(embedded.len() as u64);
                    Ok(embedded)
                }
            })
            .buffered(self.concurrency)
            .collect()
            .await;

        embed_pb.finish_with_message("embedded");

        // ── Phase 3: insert into store ───────────────────────────────────────
        let store_pb = mp.add(ProgressBar::new(total as u64));
        store_pb.set_style(
            ProgressStyle::with_template(
                "store  [{bar:35.green/black}] {pos}/{len} docs",
            )
            .unwrap()
            .progress_chars("█▓░"),
        );

        for batch_result in embedded_batches {
            let batch =
                batch_result.map_err(|e| TurboError::Pipeline(e.to_string()))?;
            let n = batch.len();
            self.store
                .upsert(&batch)
                .await
                .map_err(|e| TurboError::Pipeline(e.to_string()))?;
            store_pb.inc(n as u64);
        }
        store_pb.finish_with_message("stored");

        let elapsed = start.elapsed();
        let docs_per_sec = total as f64 / elapsed.as_secs_f64();
        let ratio = compression_ratio(dim, self.bits);
        let compressed_bytes = (original_bytes as f32 / ratio) as usize;

        Ok(PipelineStats {
            total_docs: total,
            elapsed,
            docs_per_sec,
            compression_ratio: ratio,
            original_mb: original_bytes as f64 / 1_048_576.0,
            compressed_mb: compressed_bytes as f64 / 1_048_576.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob::InMemoryBackend;
    use embedder::MockEmbedder;
    use store::TurboVecStore;

    fn make_docs(n: usize) -> Vec<Document> {
        (0..n as u64)
            .map(|i| Document {
                id: i,
                text: format!("  document number {i}  "), // leading/trailing space to test trim
                metadata: Default::default(),
            })
            .collect()
    }

    #[tokio::test]
    async fn pipeline_processes_all_docs() {
        let dim = 64;
        let embedder = Arc::new(MockEmbedder::new(dim));
        let store = Arc::new(TurboVecStore::new_in_memory("test", dim, 4));
        let pipeline = IngestionPipeline::new(embedder, store.clone(), 8);
        let stats = pipeline.run(make_docs(50)).await.unwrap();
        assert_eq!(stats.total_docs, 50);
        assert!(stats.docs_per_sec > 0.0);
        assert_eq!(store.doc_count().await.unwrap(), 50);
    }

    #[tokio::test]
    async fn pipeline_batch_boundary_correctness() {
        let dim = 32;
        let embedder = Arc::new(MockEmbedder::new(dim));
        let store = Arc::new(TurboVecStore::new_in_memory("test", dim, 4));
        // batch_size=7 with 20 docs: 2 full batches of 7 + 1 of 6
        let pipeline = IngestionPipeline::new(embedder, store.clone(), 7);
        let stats = pipeline.run(make_docs(20)).await.unwrap();
        assert_eq!(stats.total_docs, 20);
        assert_eq!(store.doc_count().await.unwrap(), 20);
    }

    #[tokio::test]
    async fn pipeline_empty_input_returns_error() {
        let embedder = Arc::new(MockEmbedder::new(64));
        let store = Arc::new(TurboVecStore::new_in_memory("test", 64, 4));
        let pipeline = IngestionPipeline::new(embedder, store, 32);
        assert!(pipeline.run(vec![]).await.is_err());
    }

    #[tokio::test]
    async fn pipeline_concurrency_does_not_change_result() {
        let dim = 32;
        let embedder = Arc::new(MockEmbedder::new(dim));
        let store = Arc::new(TurboVecStore::new_in_memory("test", dim, 4));
        let pipeline =
            IngestionPipeline::new(embedder, store.clone(), 16).with_concurrency(2);
        let stats = pipeline.run(make_docs(100)).await.unwrap();
        assert_eq!(stats.total_docs, 100);
        assert_eq!(store.doc_count().await.unwrap(), 100);
    }

    #[tokio::test]
    async fn pipeline_stats_compression_ratio() {
        let dim = 768;
        let embedder = Arc::new(MockEmbedder::new(dim));
        let store = Arc::new(TurboVecStore::new_in_memory("test", dim, 4));
        let pipeline =
            IngestionPipeline::new(embedder, store, 10).with_bits(4);
        let stats = pipeline.run(make_docs(5)).await.unwrap();
        // 768-dim, 4-bit → 8x compression ratio
        assert!((stats.compression_ratio - 8.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn pipeline_text_trimmed() {
        let dim = 16;
        let embedder = Arc::new(MockEmbedder::new(dim));
        // Use a store that echoes text back on search
        use store::LanceDbStore;
        let dir = tempfile::tempdir().unwrap();
        let lance = Arc::new(
            LanceDbStore::new(dir.path().to_str().unwrap(), "docs", dim)
                .await
                .unwrap(),
        );
        let pipeline = IngestionPipeline::new(embedder, lance.clone(), 10);
        pipeline.run(make_docs(3)).await.unwrap();
        // Text was trimmed during preprocessing
        let query = vec![0.5f32; dim];
        let req = common::SearchRequest::vector(&query, 3);
        let results = lance.search(&req).await.unwrap();
        for r in &results {
            assert!(!r.text.starts_with(' '), "text should be trimmed: '{}'", r.text);
            assert!(!r.text.ends_with(' '), "text should be trimmed: '{}'", r.text);
        }
    }
}
