use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

#[derive(Error, Debug)]
pub enum TurboError {
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("blob error: {0}")]
    Blob(String),
    #[error("store error: {0}")]
    Store(String),
    #[error("pipeline error: {0}")]
    Pipeline(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("unsupported search type for this backend: {0}")]
    UnsupportedSearchType(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, TurboError>;

// ─── Document types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: u64,
    pub text: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct EmbeddedDoc {
    pub id: u64,
    pub text: String,
    pub embedding: Vec<f32>,
    #[allow(dead_code)]
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredDoc {
    pub id: u64,
    pub text: String,
    pub score: f32,
    pub source: SearchSource,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SearchSource {
    TurboVec,
    LanceDb,
    Federated,
}

// ─── Search ───────────────────────────────────────────────────────────────────

/// What kind of search the store should perform.
#[derive(Debug, Clone, Default)]
pub enum SearchType {
    /// Approximate nearest-neighbor cosine similarity. All stores support this.
    #[default]
    Vector,
    /// BM25 full-text search. LanceDB only.
    Bm25,
    /// Reciprocal Rank Fusion of Vector + BM25. LanceDB only.
    Hybrid {
        /// Fusion constant; typically 60.
        rrf_k: usize,
    },
}

/// SQL-style equality filter pushed down to the store.
#[derive(Debug, Clone)]
pub struct MetadataFilter {
    pub column: String,
    pub value: String,
}

/// Unified search request across all VectorStore implementations.
#[derive(Debug)]
pub struct SearchRequest<'a> {
    /// Pre-computed query vector. Always required.
    pub query_embedding: &'a [f32],
    /// Raw query text. Required for `Bm25` and `Hybrid`; ignored by TurboVecStore.
    pub query_text: Option<&'a str>,
    /// Number of results to return.
    pub k: usize,
    pub search_type: SearchType,
    pub filter: Option<MetadataFilter>,
}

impl<'a> SearchRequest<'a> {
    pub fn vector(embedding: &'a [f32], k: usize) -> Self {
        Self {
            query_embedding: embedding,
            query_text: None,
            k,
            search_type: SearchType::Vector,
            filter: None,
        }
    }

    pub fn hybrid(embedding: &'a [f32], text: &'a str, k: usize, rrf_k: usize) -> Self {
        Self {
            query_embedding: embedding,
            query_text: Some(text),
            k,
            search_type: SearchType::Hybrid { rrf_k },
            filter: None,
        }
    }

    pub fn bm25(embedding: &'a [f32], text: &'a str, k: usize) -> Self {
        Self {
            query_embedding: embedding,
            query_text: Some(text),
            k,
            search_type: SearchType::Bm25,
            filter: None,
        }
    }
}

/// Which hot/cold path a HybridStore query should take.
#[derive(Debug, Clone, Default)]
pub enum SearchMode {
    /// turbovec only — pure SIMD, requires warm index.
    Hot,
    /// LanceDB only — full features (BM25, filtering, SQL).
    Cold,
    /// tokio::select! on both — first result wins. turbovec wins >99%. For benchmarking.
    Race,
    /// Merge top-k from both, deduplicate by id, re-rank by score.
    Federated(usize),
    /// turbovec when warm, LanceDB otherwise. Production default.
    #[default]
    Auto,
}

// ─── Pipeline stats ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub total_docs: usize,
    pub elapsed: std::time::Duration,
    pub docs_per_sec: f64,
    pub compression_ratio: f32,
    pub original_mb: f64,
    pub compressed_mb: f64,
}

impl PipelineStats {
    pub fn print_report(&self) {
        println!("\n─── Ingestion Stats ──────────────────────────────────");
        println!("  Documents  : {}", self.total_docs);
        println!("  Elapsed    : {:.2?}", self.elapsed);
        println!("  Throughput : {:.0} docs/sec", self.docs_per_sec);
        println!("  Original   : {:.1} MB", self.original_mb);
        println!("  Compressed : {:.1} MB", self.compressed_mb);
        println!("  Ratio      : {:.1}x", self.compression_ratio);
        println!("──────────────────────────────────────────────────────");
    }
}

// ─── Core traits ──────────────────────────────────────────────────────────────

/// Converts raw text into dense float32 vectors.
#[async_trait]
pub trait EmbeddingBackend: Send + Sync {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dim(&self) -> usize;
    fn model_name(&self) -> &str;

    async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut batch = self.embed_batch(&[text.to_string()]).await?;
        batch
            .pop()
            .ok_or_else(|| TurboError::Embedding("empty embedding response".into()))
    }
}

/// Opaque key-value object storage (local fs, S3-compatible, GCS).
#[async_trait]
pub trait BlobBackend: Send + Sync {
    async fn put(&self, key: &str, data: Bytes) -> Result<()>;
    async fn get(&self, key: &str) -> Result<Bytes>;
    async fn delete(&self, key: &str) -> Result<()>;
    async fn list_prefix(&self, prefix: &str) -> Result<Vec<String>>;
    async fn exists(&self, key: &str) -> Result<bool> {
        match self.get(key).await {
            Ok(_) => Ok(true),
            Err(TurboError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

/// Upsert + ANN search over embedded documents.
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn upsert(&self, docs: &[EmbeddedDoc]) -> Result<()>;
    async fn search(&self, request: &SearchRequest<'_>) -> Result<Vec<ScoredDoc>>;
    async fn delete(&self, id: u64) -> Result<()>;
    async fn doc_count(&self) -> Result<usize>;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Cosine similarity between two equal-length slices.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Compression ratio: original f32 bytes vs quantized bytes.
pub fn compression_ratio(dim: usize, bits: usize) -> f32 {
    let original = (dim * 4) as f32;
    let compressed = (dim * bits) as f32 / 8.0;
    original / compressed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_self_is_one() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "cosine(v,v) should be 1.0, got {sim}"
        );
    }

    #[test]
    fn compression_ratios() {
        assert_eq!(compression_ratio(768, 4) as u32, 8);
        assert_eq!(compression_ratio(1536, 2) as u32, 16);
    }
}
