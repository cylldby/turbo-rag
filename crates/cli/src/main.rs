mod backend;
mod config;
mod commands;
pub mod tui;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "turbo-rag", about = "RAG pipeline: turbovec compression + LanceDB retrieval")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest documents from a JSONL file into both stores.
    Ingest {
        #[arg(long, short)]
        input: std::path::PathBuf,
        #[arg(long, default_value = "64")]
        batch_size: usize,
        #[arg(long, default_value = "4", help = "Turbovec bit width (2 or 4)")]
        bits: usize,
        #[arg(long, help = "Enable ratatui live TUI")]
        tui: bool,
    },
    /// Query the vector store with a text prompt.
    Query {
        text: String,
        #[arg(long, default_value = "5")]
        top_k: usize,
        #[arg(long, default_value = "auto", help = "hot|cold|race|federated|auto")]
        mode: String,
        #[arg(long, default_value = "vector", help = "vector|bm25|hybrid")]
        search_type: String,
    },
    /// Run latency benchmarks and print comparison table. Optional TUI.
    Bench {
        #[arg(long)]
        corpus: Option<std::path::PathBuf>,
        #[arg(long)]
        queries: Option<std::path::PathBuf>,
        #[arg(long, default_value = "10000", help = "Comma-separated corpus sizes, e.g. 10000,100000")]
        scales: String,
        #[arg(long, help = "Enable ratatui live TUI")]
        tui: bool,
    },
    /// Check all configured backends are reachable.
    Doctor,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("turbo_rag=info".parse()?),
        )
        .init();

    // Prometheus metrics exporter on :9091/metrics
    // Grafana scrapes this via prometheus.yml. Silently skip if port is already bound.
    let metrics_port: u16 = std::env::var("METRICS_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9091);
    let builder = metrics_exporter_prometheus::PrometheusBuilder::new()
        .with_http_listener(([0, 0, 0, 0], metrics_port));
    match builder.install() {
        Ok(()) => tracing::info!("metrics exporter on :{metrics_port}/metrics"),
        Err(e) => tracing::debug!("metrics exporter not started (port busy?): {e}"),
    }

    let cli = Cli::parse();
    match cli.command {
        Commands::Ingest { input, batch_size, bits, tui } => {
            commands::ingest(input, batch_size, bits, tui).await
        }
        Commands::Query { text, top_k, mode, search_type } => {
            commands::query(text, top_k, &mode, &search_type).await
        }
        Commands::Bench { corpus, queries, scales, tui } => {
            commands::bench(corpus, queries, &scales, tui).await
        }
        Commands::Doctor => commands::doctor().await,
    }
}
