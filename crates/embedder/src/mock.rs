use async_trait::async_trait;
use common::{EmbeddingBackend, Result};

/// Deterministic fixed-value embedder for unit tests. No network, no model.
pub struct MockEmbedder {
    dim: usize,
    fill_value: f32,
}

impl MockEmbedder {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            fill_value: 0.1,
        }
    }

    pub fn with_value(dim: usize, fill_value: f32) -> Self {
        Self { dim, fill_value }
    }
}

#[async_trait]
impl EmbeddingBackend for MockEmbedder {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|_| vec![self.fill_value; self.dim])
            .collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        "mock-embed"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::EmbeddingBackend;

    #[tokio::test]
    async fn correct_dim_single() {
        let m = MockEmbedder::new(768);
        let vecs = m.embed_batch(&["hello".to_string()]).await.unwrap();
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].len(), 768);
    }

    #[tokio::test]
    async fn batch_count_matches_input() {
        let m = MockEmbedder::new(128);
        let texts: Vec<String> = (0..47).map(|i| format!("doc {i}")).collect();
        let vecs = m.embed_batch(&texts).await.unwrap();
        assert_eq!(vecs.len(), 47);
    }

    #[tokio::test]
    async fn fill_value_respected() {
        let m = MockEmbedder::with_value(32, 0.42);
        let vecs = m.embed_batch(&["x".to_string()]).await.unwrap();
        assert!(vecs[0].iter().all(|&v| (v - 0.42).abs() < 1e-6));
    }

    #[tokio::test]
    async fn embed_one_shortcut() {
        let m = MockEmbedder::new(64);
        let v = m.embed_one("single text").await.unwrap();
        assert_eq!(v.len(), 64);
    }

    #[tokio::test]
    async fn batch_split_100_docs_at_batch32() {
        // Pipeline splits into 3 batches of 32 + 1 of 4; all embeddings must be present.
        let m = MockEmbedder::new(16);
        let texts: Vec<String> = (0..100).map(|i| format!("t{i}")).collect();
        // embed_batch processes the whole slice in one call (caller is responsible for splitting)
        let result = m.embed_batch(&texts).await.unwrap();
        assert_eq!(result.len(), 100);
        assert!(result.iter().all(|v| v.len() == 16));
    }

    #[test]
    fn dim_accessor() {
        assert_eq!(MockEmbedder::new(384).dim(), 384);
        assert_eq!(MockEmbedder::new(1536).dim(), 1536);
    }
}
