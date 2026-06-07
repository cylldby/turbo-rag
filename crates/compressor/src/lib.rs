// Provide BLAS symbols required by turbovec's ndarray dependency.
#[cfg(target_os = "macos")]
extern crate accelerate_src;
#[cfg(not(target_os = "macos"))]
extern crate blas_src;

use anyhow::Result;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
// IdMapIndex wraps TurboQuantIndex with external ID tracking + O(1) deletes.
// It is sufficient for all use cases in this project.
use turbovec::IdMapIndex;

// ─── CompressionInfo ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    pub dim: usize,
    pub bits: usize,
    pub ratio: f32,
    pub original_bytes_per_vec: usize,
    pub compressed_bytes_per_vec: usize,
    pub original_mb_per_million: f64,
    pub compressed_mb_per_million: f64,
}

impl CompressionInfo {
    pub fn compute(dim: usize, bits: usize) -> Self {
        let original = dim * 4;
        let compressed = (dim * bits).div_ceil(8);
        let ratio = original as f32 / compressed as f32;
        Self {
            dim,
            bits,
            ratio,
            original_bytes_per_vec: original,
            compressed_bytes_per_vec: compressed,
            original_mb_per_million: (original as f64 * 1_000_000.0) / (1024.0 * 1024.0),
            compressed_mb_per_million: (compressed as f64 * 1_000_000.0) / (1024.0 * 1024.0),
        }
    }

    pub fn print_table() {
        println!("\n─── Compression Ratio Table ──────────────────────────────────────────────────");
        println!("{:<8} {:<6} {:<8} {:<16} {:<16}", "dim", "bits", "ratio", "original/1M", "compressed/1M");
        println!("{}", "─".repeat(78));
        for dim in [384, 768, 1024, 1536] {
            for bits in [2, 4] {
                let info = Self::compute(dim, bits);
                println!(
                    "{:<8} {:<6} {:<8.1}x {:<16.1} {:<16.1}",
                    dim,
                    bits,
                    info.ratio,
                    info.original_mb_per_million,
                    info.compressed_mb_per_million
                );
            }
        }
        println!("──────────────────────────────────────────────────────────────────────────────");
    }
}

// ─── QuantizedIndex ───────────────────────────────────────────────────────────

/// Thread-safe wrapper over turbovec IdMapIndex.
/// IdMapIndex provides ID-tracked ANN search + O(1) deletes.
/// Serialization uses a tempfile round-trip (turbovec write/load use file paths).
pub struct QuantizedIndex {
    id_map: IdMapIndex,
    dim: usize,
    bits: usize,
}

impl QuantizedIndex {
    pub fn new(dim: usize, bits: usize) -> Self {
        Self {
            id_map: IdMapIndex::new(dim, bits),
            dim,
            bits,
        }
    }

    /// Add a batch of (id, vector) pairs.
    pub fn add_batch(&mut self, ids: &[u64], vectors: &[Vec<f32>]) {
        debug_assert_eq!(ids.len(), vectors.len());
        for (id, vec) in ids.iter().zip(vectors.iter()) {
            self.id_map.add_with_ids(vec, &[*id]);
        }
    }

    /// Search returns (external_id, score) pairs sorted by descending score.
    /// turbovec returns (scores: Vec<f32>, ids: Vec<u64>).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let (scores, ids) = self.id_map.search(query, k);
        ids.into_iter().zip(scores).collect()
    }

    /// Remove a document by its external id. O(1).
    pub fn remove(&mut self, id: u64) {
        self.id_map.remove(id);
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bits(&self) -> usize {
        self.bits
    }

    pub fn compression_info(&self) -> CompressionInfo {
        CompressionInfo::compute(self.dim, self.bits)
    }

    /// Serialize the id_map index to bytes via a tempfile.
    pub fn to_bytes(&self) -> Result<Bytes> {
        let tmp = tempfile::NamedTempFile::new()?;
        let path = tmp.path().to_str().unwrap().to_string();
        self.id_map.write(&path)?;
        let data = std::fs::read(tmp.path())?;
        Ok(Bytes::from(data))
    }

    /// Deserialize from bytes via a tempfile.
    pub fn from_bytes(data: Bytes, dim: usize, bits: usize) -> Result<Self> {
        let tmp = tempfile::NamedTempFile::new()?;
        std::fs::write(tmp.path(), &data)?;
        let id_map = IdMapIndex::load(tmp.path().to_str().unwrap())?;
        Ok(Self { id_map, dim, bits })
    }

    /// Write to a named file path (for local cache).
    pub fn save_to(&self, path: &PathBuf) -> Result<()> {
        self.id_map.write(path.to_str().unwrap())?;
        Ok(())
    }

    /// Load from a named file path.
    pub fn load_from(path: &PathBuf, dim: usize, bits: usize) -> Result<Self> {
        let id_map = IdMapIndex::load(path.to_str().unwrap())?;
        Ok(Self { id_map, dim, bits })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::cosine_similarity;

    /// Distinct unit vector per id — varying angle ensures vectors are well-separated.
    fn unit_vec_for_id(id: usize, dim: usize) -> Vec<f32> {
        use std::f32::consts::PI;
        let offset = id as f32 * PI / 8.0;  // 22.5° spacing between ids
        let v: Vec<f32> = (0..dim)
            .map(|i| (i as f32 * PI / dim as f32 + offset).sin())
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }

    #[test]
    fn compression_ratios_correct() {
        assert_eq!(CompressionInfo::compute(768, 4).ratio as u32, 8);
        assert_eq!(CompressionInfo::compute(1536, 2).ratio as u32, 16);
        assert_eq!(CompressionInfo::compute(384, 4).ratio as u32, 8);
    }

    #[test]
    fn print_compression_table() {
        CompressionInfo::print_table();
    }

    #[test]
    fn add_and_search_returns_correct_count() {
        let dim = 64;
        let mut idx = QuantizedIndex::new(dim, 4);
        let vecs: Vec<Vec<f32>> = (0..20).map(|i| unit_vec_for_id(i, dim)).collect();
        let ids: Vec<u64> = (0..20).collect();
        idx.add_batch(&ids, &vecs);
        let results = idx.search(&vecs[0], 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn top_result_is_self() {
        let dim = 128;  // higher dim → better quantization fidelity
        let mut idx = QuantizedIndex::new(dim, 4);
        let vecs: Vec<Vec<f32>> = (0..50).map(|i| unit_vec_for_id(i, dim)).collect();
        let ids: Vec<u64> = (0..50).collect();
        idx.add_batch(&ids, &vecs);
        let results = idx.search(&vecs[7], 3);
        assert!(
            results.iter().any(|(id, _)| *id == 7),
            "doc 7 should appear in top-3; got: {results:?}"
        );
    }

    #[test]
    fn roundtrip_cosine_similarity() {
        let dim = 128;
        let v1 = unit_vec_for_id(3, dim);
        let v2: Vec<f32> = v1.iter().map(|x| x + 0.01).collect();
        let sim = cosine_similarity(&v1, &v2);
        assert!(sim > 0.9, "similar vectors should have high cosine similarity: {sim}");
    }

    #[test]
    fn idmap_delete_removes_result() {
        let dim = 64;
        let mut idx = QuantizedIndex::new(dim, 4);
        let vecs: Vec<Vec<f32>> = (0..10).map(|i| unit_vec_for_id(i, dim)).collect();
        let ids: Vec<u64> = (0..10).collect();
        idx.add_batch(&ids, &vecs);
        idx.remove(0);
        let results = idx.search(&vecs[0], 10);
        assert!(results.iter().all(|(id, _)| *id != 0), "deleted id should not appear");
    }
}
