/// Ratatui TUI module for live ingestion and benchmark visualisation.
/// Rendered when --tui flag is passed to `ingest` or `bench`.
/// Full implementation in M8.

#[derive(Debug, Clone)]
pub struct IngestFrame {
    pub total: u64,
    pub done: u64,
    pub docs_per_sec: f64,
    pub compression_ratio: f32,
    pub original_mb: f64,
    pub compressed_mb: f64,
    pub turbovec_warm: bool,
    pub lancedb_count: usize,
}

#[derive(Debug, Clone)]
pub struct BenchSample {
    pub turbovec_ms: f64,
    pub lancedb_ms: f64,
    pub corpus_size: usize,
    pub queries_done: usize,
    pub queries_total: usize,
    pub race_wins: usize,
}

// Full TUI rendering logic added in M8 using ratatui + crossterm.
// The bench command pipes BenchSample events via tokio::sync::mpsc to the renderer.
