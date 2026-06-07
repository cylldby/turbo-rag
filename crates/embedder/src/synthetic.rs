use common::Result;

// random_unit_vec is always available (used by the bench CLI command).
// The EmbeddingBackend impl is only compiled when backend-synthetic is active.

#[cfg(feature = "backend-synthetic")]
use async_trait::async_trait;
#[cfg(feature = "backend-synthetic")]
use common::EmbeddingBackend;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[cfg(feature = "backend-synthetic")]
/// Returns random unit-norm f32 vectors at any dimension.
/// No model download, no API key. Used for turbovec speed benchmarks at d=1536.
pub struct SyntheticEmbedder {
    dim: usize,
}

#[cfg(feature = "backend-synthetic")]
impl SyntheticEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

#[cfg(feature = "backend-synthetic")]
#[async_trait]
impl EmbeddingBackend for SyntheticEmbedder {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| random_unit_vec(self.dim)).collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        "synthetic"
    }
}

pub fn random_unit_vec(dim: usize) -> Vec<f32> {
    let mut rng = SmallRng::from_entropy();
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "backend-synthetic")]
    #[tokio::test]
    async fn correct_dim() {
        let e = SyntheticEmbedder::new(1536);
        assert_eq!(e.dim(), 1536);
        let vecs = e.embed_batch(&["x".to_string()]).await.unwrap();
        assert_eq!(vecs[0].len(), 1536);
    }

    #[cfg(feature = "backend-synthetic")]
    #[tokio::test]
    async fn batch_size_matches() {
        let e = SyntheticEmbedder::new(128);
        let texts: Vec<String> = (0..17).map(|i| format!("t{i}")).collect();
        let vecs = e.embed_batch(&texts).await.unwrap();
        assert_eq!(vecs.len(), 17);
    }

    #[test]
    fn random_unit_vec_is_unit_norm() {
        let v = random_unit_vec(256);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
    }

    #[test]
    fn random_unit_vecs_differ() {
        let v1 = random_unit_vec(64);
        let v2 = random_unit_vec(64);
        // Probability of collision is astronomically small
        let dot: f32 = v1.iter().zip(&v2).map(|(a, b)| a * b).sum();
        assert!(
            dot.abs() < 0.99,
            "two random unit vectors should not be identical"
        );
    }
}
