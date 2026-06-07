use serde::{Deserialize, Serialize};

/// Layered configuration: config/default.toml < .env < env vars < CLI flags.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AppConfig {
    pub embedding: EmbeddingConfig,
    pub blob: BlobConfig,
    pub lance: LanceConfig,
    pub pipeline: PipelineConfig,
    pub search: SearchConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct EmbeddingConfig {
    pub backend: String,       // fastembed | openai-compat | synthetic
    pub base_url: String,      // for openai-compat
    pub api_key: Option<String>,
    pub model: String,
    pub dim: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BlobConfig {
    pub backend: String,       // local | s3
    pub local_path: String,
    pub bucket: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LanceConfig {
    pub uri: String,           // ./data/lance or s3://bucket/lance
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PipelineConfig {
    pub batch_size: usize,
    pub turbovec_bits: usize,
    pub load_strategy: String, // from-blob | rebuild-from-lance | lazy
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SearchConfig {
    pub mode: String,          // hot | cold | race | federated | auto
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            embedding: EmbeddingConfig {
                backend: "openai-compat".into(),
                base_url: "http://localhost:11434/v1".into(),
                api_key: None,
                model: "nomic-embed-text".into(),
                dim: 768,
            },
            blob: BlobConfig {
                backend: "local".into(),
                local_path: "./data/blob".into(),
                bucket: "turbo-rag-dev".into(),
            },
            lance: LanceConfig {
                uri: "./data/lance".into(),
            },
            pipeline: PipelineConfig {
                batch_size: 64,
                turbovec_bits: 4,
                load_strategy: "from-blob".into(),
            },
            search: SearchConfig {
                mode: "auto".into(),
            },
        }
    }
}

pub fn load() -> anyhow::Result<AppConfig> {
    let cfg = config::Config::builder()
        .add_source(config::File::with_name("config/default").required(false))
        .add_source(config::Environment::default().separator("__"))
        .build()?;
    Ok(cfg.try_deserialize().unwrap_or_default())
}
