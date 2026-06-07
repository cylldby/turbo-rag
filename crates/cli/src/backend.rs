use crate::config::AppConfig;
use blob::{LocalFsBackend, S3Backend};
use common::{BlobBackend, EmbeddingBackend, SearchMode, VectorStore};
use embedder::{FastEmbedBackend, MockEmbedder, OpenAICompatBackend, SyntheticEmbedder};
use std::sync::Arc;
use store::{HybridStore, LanceDbStore, TurboVecStore};
use tracing::info;

pub async fn build_embedder(cfg: &AppConfig) -> anyhow::Result<Arc<dyn EmbeddingBackend>> {
    match cfg.embedding.backend.as_str() {
        "fastembed" => {
            info!("embedding backend: fastembed (local ONNX, dim={})", cfg.embedding.dim);
            let backend = match cfg.embedding.dim {
                1024 => FastEmbedBackend::mxbai_large()?,
                384 => FastEmbedBackend::bge_small_en()?,
                _ => FastEmbedBackend::bge_base_en()?,
            };
            Ok(Arc::new(backend))
        }
        "openai-compat" => {
            info!(
                "embedding backend: openai-compat  url={}  model={}  dim={}",
                cfg.embedding.base_url, cfg.embedding.model, cfg.embedding.dim
            );
            Ok(Arc::new(OpenAICompatBackend::new(
                &cfg.embedding.base_url,
                cfg.embedding.api_key.clone(),
                &cfg.embedding.model,
                cfg.embedding.dim,
            )))
        }
        "synthetic" => {
            info!("embedding backend: synthetic (random unit-norm, dim={})", cfg.embedding.dim);
            Ok(Arc::new(SyntheticEmbedder::new(cfg.embedding.dim)))
        }
        "mock" => Ok(Arc::new(MockEmbedder::new(cfg.embedding.dim))),
        other => anyhow::bail!("unknown embedding backend: '{other}' (valid: fastembed, openai-compat, synthetic)"),
    }
}

pub async fn build_blob(cfg: &AppConfig) -> anyhow::Result<Arc<dyn BlobBackend>> {
    match cfg.blob.backend.as_str() {
        "local" => {
            info!("blob backend: local  path={}", cfg.blob.local_path);
            Ok(Arc::new(LocalFsBackend::new(&cfg.blob.local_path)?))
        }
        "s3" => {
            info!("blob backend: s3  bucket={}", cfg.blob.bucket);
            Ok(Arc::new(S3Backend::from_env(&cfg.blob.bucket)?))
        }
        other => anyhow::bail!("unknown blob backend: '{other}' (valid: local, s3)"),
    }
}

pub async fn build_store(cfg: &AppConfig) -> anyhow::Result<Arc<HybridStore>> {
    let blob = build_blob(cfg).await?;
    let dim = cfg.embedding.dim;
    let bits = cfg.pipeline.turbovec_bits;

    info!(
        "vector stores: turbovec ({}d {}bit in-memory)  +  lancedb  uri={}",
        dim, bits, cfg.lance.uri
    );

    let hot = Arc::new(TurboVecStore::with_blob("main", blob, dim, bits));
    let cold = Arc::new(LanceDbStore::new(&cfg.lance.uri, "docs", dim).await?);

    let mode = match cfg.search.mode.as_str() {
        "hot" => SearchMode::Hot,
        "cold" => SearchMode::Cold,
        "race" => SearchMode::Race,
        "federated" => SearchMode::Federated(cfg.pipeline.batch_size),
        _ => SearchMode::Auto,
    };

    let store = Arc::new(HybridStore::new(hot.clone(), cold, mode));

    // Apply load strategy
    match cfg.pipeline.load_strategy.as_str() {
        "from-blob" => {
            info!("load strategy: from-blob — loading turbovec index");
            match hot.load_from_blob().await {
                Ok(_) => info!("turbovec index loaded ({} docs)", hot.doc_count().await.unwrap_or(0)),
                Err(e) => info!("turbovec index not found, starting fresh: {e}"),
            }
        }
        "lazy" => info!("load strategy: lazy — turbovec warms on first ingest"),
        other => info!("load strategy: {other}"),
    }

    Ok(store)
}
