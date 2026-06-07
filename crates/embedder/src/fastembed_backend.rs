use async_trait::async_trait;
use common::{EmbeddingBackend, Result, TurboError};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::Mutex;

/// Local ONNX embedding via fastembed 5.
/// `TextEmbedding::embed` takes `&mut self`, hence the `Mutex`.
/// Model weights are downloaded on first construction (~100–670 MB, cached locally).
pub struct FastEmbedBackend {
    model: Mutex<TextEmbedding>,
    dim: usize,
    name: &'static str,
}

impl FastEmbedBackend {
    /// BGE Base EN v1.5 — 768-dim, good all-round choice for local dev.
    pub fn bge_base_en() -> anyhow::Result<Self> {
        Self::build(EmbeddingModel::BGEBaseENV15, 768, "BGEBaseENV15")
    }

    /// Mxbai Embed Large v1 — 1024-dim, zero-cost upgrade path (~670 MB download).
    pub fn mxbai_large() -> anyhow::Result<Self> {
        Self::build(EmbeddingModel::MxbaiEmbedLargeV1, 1024, "MxbaiEmbedLargeV1")
    }

    /// BGE Small EN v1.5 — 384-dim, fastest inference, smallest download.
    pub fn bge_small_en() -> anyhow::Result<Self> {
        Self::build(EmbeddingModel::BGESmallENV15, 384, "BGESmallENV15")
    }

    fn build(model_id: EmbeddingModel, dim: usize, name: &'static str) -> anyhow::Result<Self> {
        let model = TextEmbedding::try_new(
            InitOptions::new(model_id).with_show_download_progress(true),
        )?;
        Ok(Self { model: Mutex::new(model), dim, name })
    }
}

#[async_trait]
impl EmbeddingBackend for FastEmbedBackend {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // fastembed's embed() is sync and CPU-bound. For production, wrap in
        // spawn_blocking; for now, run inline to keep the implementation simple.
        let texts_owned = texts.to_vec();
        self.model
            .lock()
            .map_err(|_| TurboError::Embedding("mutex poisoned".into()))?
            .embed(texts_owned, Some(64))
            .map_err(|e| TurboError::Embedding(e.to_string()))
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::cosine_similarity;

    // FastEmbed downloads models at runtime — skip in unit tests, run in integration.
    // Use `cargo test -p embedder --features backend-fastembed -- --ignored` to run.

    #[tokio::test]
    #[ignore = "downloads ~100MB model on first run"]
    async fn bge_small_embed_and_dim() {
        let backend = FastEmbedBackend::bge_small_en().unwrap();
        assert_eq!(backend.dim(), 384);
        let vecs = backend.embed_batch(&["hello world".to_string()]).await.unwrap();
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].len(), 384);
    }

    #[tokio::test]
    #[ignore = "downloads ~100MB model on first run"]
    async fn bge_small_same_text_cosine_near_one() {
        let backend = FastEmbedBackend::bge_small_en().unwrap();
        let vecs = backend
            .embed_batch(&["semantic search".to_string(), "semantic search".to_string()])
            .await
            .unwrap();
        let sim = cosine_similarity(&vecs[0], &vecs[1]);
        assert!(sim > 0.999, "same text should have cosine~1, got {sim}");
    }

    #[tokio::test]
    #[ignore = "downloads ~100MB model on first run"]
    async fn bge_small_distinct_texts_lower_similarity() {
        let backend = FastEmbedBackend::bge_small_en().unwrap();
        let vecs = backend
            .embed_batch(&[
                "machine learning and neural networks".to_string(),
                "cooking pasta with olive oil".to_string(),
            ])
            .await
            .unwrap();
        let sim = cosine_similarity(&vecs[0], &vecs[1]);
        assert!(sim < 0.9, "unrelated texts should have cosine < 0.9, got {sim}");
    }
}
