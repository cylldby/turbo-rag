use anyhow::Result;
use common::{Document, EmbeddingBackend, SearchRequest, VectorStore};
use metrics::{counter, gauge, histogram};
use pipeline::IngestionPipeline;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::{backend, config};

// ── ingest ────────────────────────────────────────────────────────────────────

pub async fn ingest(input: PathBuf, batch_size: usize, bits: usize, _use_tui: bool) -> Result<()> {
    let mut cfg = config::load()?;
    cfg.pipeline.batch_size = batch_size;
    cfg.pipeline.turbovec_bits = bits;
    // Ingest always starts fresh — don't load the old index before replacing it.
    cfg.pipeline.load_strategy = "lazy".into();

    println!("turbo-rag ingest  input={input:?}  batch={batch_size}  bits={bits}-bit");
    println!();

    // ── load documents ───────────────────────────────────────────────────────
    let docs = load_jsonl(&input)?;
    println!("  loaded {} documents", docs.len());

    // ── build backends ───────────────────────────────────────────────────────
    let embedder = backend::build_embedder(&cfg).await?;
    let store = backend::build_store(&cfg).await?;

    // ── run pipeline ─────────────────────────────────────────────────────────
    let pipeline =
        IngestionPipeline::new(embedder, store.clone() as Arc<dyn VectorStore>, batch_size)
            .with_bits(bits);
    let stats = pipeline.run(docs).await?;
    stats.print_report();

    // ── emit metrics ─────────────────────────────────────────────────────────
    counter!("turborag_docs_ingested_total").increment(stats.total_docs as u64);
    gauge!("turborag_docs_per_second").set(stats.docs_per_sec as f64);
    gauge!("turborag_compression_ratio").set(stats.compression_ratio as f64);
    gauge!("turborag_compressed_mb").set(stats.compressed_mb);
    gauge!("turborag_original_mb").set(stats.original_mb);

    // ── build FTS index for BM25 / hybrid search ────────────────────────────
    print!("  building full-text search index... ");
    store.cold.ensure_fts_index().await?;
    println!("ok");

    // ── flush turbovec to blob ───────────────────────────────────────────────
    print!("  flushing turbovec index to blob... ");
    store.hot.flush().await?;
    println!("ok");

    Ok(())
}

// ── query ─────────────────────────────────────────────────────────────────────

pub async fn query(text: String, top_k: usize, mode: &str, search_type: &str) -> Result<()> {
    let mut cfg = config::load()?;
    cfg.search.mode = mode.to_string();

    let embedder = backend::build_embedder(&cfg).await?;
    let store = backend::build_store(&cfg).await?;

    // ── embed query ──────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let embedding = embedder.embed_one(&text).await?;
    let embed_ms = t0.elapsed().as_millis();

    // ── build search request ─────────────────────────────────────────────────
    let search_req = match search_type {
        "bm25" => SearchRequest::bm25(&embedding, &text, top_k),
        "hybrid" => SearchRequest::hybrid(&embedding, &text, top_k, 60),
        _ => SearchRequest::vector(&embedding, top_k),
    };

    // ── search ───────────────────────────────────────────────────────────────
    let t1 = Instant::now();
    let results = (store.clone() as Arc<dyn VectorStore>)
        .search(&search_req)
        .await?;
    let search_ms = t1.elapsed().as_millis();

    // ── emit metrics ─────────────────────────────────────────────────────────
    counter!("turborag_queries_total", "mode" => mode.to_string(), "type" => search_type.to_string())
        .increment(1);
    histogram!("turborag_embed_latency_ms").record(embed_ms as f64);
    histogram!("turborag_search_latency_ms", "mode" => mode.to_string()).record(search_ms as f64);
    if let Some(r) = results.first() {
        gauge!("turborag_search_source",
            "source" => format!("{:?}", r.source))
        .set(1.0);
    }

    // ── print results ─────────────────────────────────────────────────────────
    println!("\nQuery : {text:?}");
    println!("Mode  : {mode}  |  Type: {search_type}  |  Top-{top_k}");
    println!(
        "Times : embed {embed_ms}ms  search {search_ms}ms  total {}ms",
        embed_ms + search_ms
    );
    println!();
    println!("{:<4} {:<10} {:<8} text", "#", "id", "score");
    println!("{}", "─".repeat(72));
    for (i, r) in results.iter().enumerate() {
        let snippet = if r.text.len() > 55 {
            &r.text[..55]
        } else {
            &r.text
        };
        println!("{:<4} {:<10} {:<8.4} {}", i + 1, r.id, r.score, snippet);
    }
    if results.is_empty() {
        println!("  (no results — run `turbo-rag ingest` first)");
    }
    println!();
    println!("Source: {:?}", results.first().map(|r| &r.source));

    Ok(())
}

// ── bench ─────────────────────────────────────────────────────────────────────

pub async fn bench(
    corpus: Option<PathBuf>,
    _queries: Option<PathBuf>,
    scales: &str,
    _use_tui: bool,
) -> Result<()> {
    use embedder::SyntheticEmbedder;
    use store::{LanceDbStore, TurboVecStore};

    let scales: Vec<usize> = scales
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let dim = 768usize;
    let bits = 4usize;

    println!("turbo-rag bench  dim={dim}  bits={bits}");
    println!("scales: {scales:?}");
    println!();

    // Print header
    println!(
        "{:<10} {:<16} {:<16} {:<12} {:<12}",
        "corpus", "turbovec p50", "turbovec p99", "lance p50", "lance p99"
    );
    println!("{}", "─".repeat(70));

    let synthetic = Arc::new(SyntheticEmbedder::new(dim));

    for &n_docs in &scales {
        // ── build TurboVec in-memory index ───────────────────────────────────
        let blob = Arc::new(blob::InMemoryBackend::new());
        let turbo = Arc::new(TurboVecStore::with_blob("bench", blob, dim, bits));

        // ── build local LanceDB index ────────────────────────────────────────
        let dir = tempfile::tempdir()?;
        let lance = Arc::new(LanceDbStore::new(dir.path().to_str().unwrap(), "bench", dim).await?);

        // ── ingest corpus (load from file or generate synthetic) ─────────────
        let docs = if let Some(ref path) = corpus {
            load_jsonl(path)?
                .into_iter()
                .take(n_docs)
                .collect::<Vec<_>>()
        } else {
            generate_synthetic_docs(n_docs, dim)
        };

        if docs.is_empty() {
            println!("{:<10} (no docs)", n_docs);
            continue;
        }

        // Embed + ingest
        let pipe_turbo = IngestionPipeline::new(
            synthetic.clone() as Arc<dyn EmbeddingBackend>,
            turbo.clone() as Arc<dyn VectorStore>,
            512,
        );
        pipe_turbo.run(docs.clone()).await?;

        let pipe_lance = IngestionPipeline::new(
            synthetic.clone() as Arc<dyn EmbeddingBackend>,
            lance.clone() as Arc<dyn VectorStore>,
            512,
        );
        pipe_lance.run(docs).await?;

        // ── measure search latencies ─────────────────────────────────────────
        const N_QUERIES: usize = 200;
        let mut turbo_ms = Vec::with_capacity(N_QUERIES);
        let mut lance_ms = Vec::with_capacity(N_QUERIES);

        for _ in 0..N_QUERIES {
            let q = embedder::synthetic::random_unit_vec(dim);
            let req = SearchRequest::vector(&q, 5);

            let t = Instant::now();
            let _ = (turbo.clone() as Arc<dyn VectorStore>).search(&req).await?;
            turbo_ms.push(t.elapsed().as_micros());

            let t = Instant::now();
            let _ = (lance.clone() as Arc<dyn VectorStore>).search(&req).await?;
            lance_ms.push(t.elapsed().as_micros());
        }

        let (tp50, tp99) = percentiles(&mut turbo_ms);
        let (lp50, lp99) = percentiles(&mut lance_ms);

        println!(
            "{:<10} {:<16} {:<16} {:<12} {:<12}",
            n_docs,
            format!("{:.2}ms", tp50 as f64 / 1000.0),
            format!("{:.2}ms", tp99 as f64 / 1000.0),
            format!("{:.2}ms", lp50 as f64 / 1000.0),
            format!("{:.2}ms", lp99 as f64 / 1000.0),
        );
    }

    println!();
    println!("turbovec = in-memory SIMD (TurboQuant {bits}-bit)   lance = LanceDB local HNSW");
    Ok(())
}

// ── doctor ────────────────────────────────────────────────────────────────────

pub async fn doctor() -> Result<()> {
    let cfg = config::load().unwrap_or_default();
    println!("\n─── turbo-rag doctor ──────────────────────────────────────────────────");

    let embed_status = match cfg.embedding.backend.as_str() {
        "openai-compat" => check_http(&format!("{}/models", cfg.embedding.base_url)).await,
        "fastembed" | "synthetic" | "mock" => Ok("local (no network)".to_string()),
        other => Err(format!("unknown backend: {other}")),
    };
    health_row(
        "embedding",
        &format!("{} / {}", cfg.embedding.backend, cfg.embedding.model),
        embed_status,
    );

    let blob_status = match cfg.blob.backend.as_str() {
        "local" => match std::fs::create_dir_all(&cfg.blob.local_path) {
            Ok(_) => Ok(format!("path: {}", cfg.blob.local_path)),
            Err(e) => Err(e.to_string()),
        },
        "s3" => {
            let ep = std::env::var("AWS_ENDPOINT_URL")
                .unwrap_or_else(|_| "https://s3.amazonaws.com".into());
            check_http(&ep).await
        }
        other => Err(format!("unknown backend: {other}")),
    };
    health_row("blob", &cfg.blob.backend, blob_status);

    let lance_status = if cfg.lance.uri.starts_with("s3://") {
        Ok("s3-native (verified with blob)".to_string())
    } else {
        match std::fs::create_dir_all(&cfg.lance.uri) {
            Ok(_) => Ok(format!("path: {}", cfg.lance.uri)),
            Err(e) => Err(e.to_string()),
        }
    };
    health_row("lancedb", &cfg.lance.uri, lance_status);

    println!("────────────────────────────────────────────────────────────────────────");
    Ok(())
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn load_jsonl(path: &PathBuf) -> Result<Vec<Document>> {
    // Raw struct that accepts both our fixture format and BEIR format:
    //   fixture: {"id": 1,      "text": "...", "metadata": {...}}
    //   BEIR:    {"_id": "123", "title": "...", "text": "...", "metadata": {}}
    #[derive(serde::Deserialize)]
    struct RawDoc {
        // Accept numeric id, string id, or BEIR _id
        #[serde(default)]
        id: Option<serde_json::Value>,
        #[serde(rename = "_id", default)]
        beir_id: Option<String>,
        title: Option<String>,
        text: String,
        #[serde(default)]
        metadata: std::collections::HashMap<String, String>,
    }

    let file =
        std::fs::File::open(path).map_err(|e| anyhow::anyhow!("cannot open {path:?}: {e}"))?;
    let reader = BufReader::new(file);
    let mut docs = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let raw: RawDoc =
            serde_json::from_str(&line).map_err(|e| anyhow::anyhow!("line {}: {e}", i + 1))?;

        // Resolve id: numeric id > string id > BEIR _id > line number fallback
        let id: u64 = match &raw.id {
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(i as u64),
            Some(serde_json::Value::String(s)) => s.parse().unwrap_or(i as u64),
            _ => raw
                .beir_id
                .as_deref()
                .and_then(|s| s.parse().ok())
                .unwrap_or(i as u64),
        };

        // Combine title + text if both present (common in BEIR datasets)
        let text = match raw.title.as_deref().filter(|t| !t.is_empty()) {
            Some(title) => format!("{title}. {}", raw.text),
            None => raw.text,
        };

        docs.push(Document {
            id,
            text,
            metadata: raw.metadata,
        });
    }
    Ok(docs)
}

fn generate_synthetic_docs(n: usize, _dim: usize) -> Vec<Document> {
    (0..n as u64)
        .map(|id| Document {
            id,
            text: format!("synthetic document {id} for benchmarking"),
            metadata: Default::default(),
        })
        .collect()
}

fn percentiles(samples: &mut [u128]) -> (u128, u128) {
    samples.sort_unstable();
    let p50 = samples[samples.len() / 2];
    let p99 = samples[samples.len() * 99 / 100];
    (p50, p99)
}

fn health_row(component: &str, detail: &str, result: std::result::Result<String, String>) {
    match result {
        Ok(msg) => println!("  ✓ {component:<12} {detail:<36} {msg}"),
        Err(err) => println!("  ✗ {component:<12} {detail:<36} ERROR: {err}"),
    }
}

async fn check_http(url: &str) -> std::result::Result<String, String> {
    match reqwest::get(url).await {
        Ok(resp) => Ok(format!("HTTP {}", resp.status())),
        Err(e) => Err(e.to_string()),
    }
}
