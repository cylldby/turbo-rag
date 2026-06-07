use async_trait::async_trait;
use common::{EmbeddingBackend, Result, TurboError};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// OpenAI-compatible `/v1/embeddings` endpoint.
///
/// Works with:
/// - **Ollama** (local, no key): `base_url = "http://localhost:11434/v1"`
/// - **OpenAI** (paid):          `base_url = "https://api.openai.com/v1"`
/// - **Voyage AI** (free tier):  `base_url = "https://api.voyageai.com/v1"`
/// - **WireMock** (tests):       `base_url = "http://localhost:8080/v1"`
/// - Any Azure / custom endpoint with the same HTTP shape.
pub struct OpenAICompatBackend {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    dim: usize,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedItem>,
}

#[derive(Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
}

impl OpenAICompatBackend {
    /// Ollama running on localhost — no API key required.
    pub fn ollama(model: impl Into<String>, dim: usize) -> Self {
        Self::new("http://localhost:11434/v1", None::<String>, model, dim)
    }

    /// Construct with arbitrary base URL. API key is optional (omit for Ollama).
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<impl Into<String>>,
        model: impl Into<String>,
        dim: usize,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.map(|k| k.into()),
            model: model.into(),
            dim,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[async_trait]
impl EmbeddingBackend for OpenAICompatBackend {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);
        let mut req = self.client.post(&url).json(&EmbedRequest {
            model: &self.model,
            input: texts,
        });
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp: EmbedResponse = req
            .send()
            .await
            .map_err(|e| TurboError::Embedding(format!("HTTP request failed: {e}")))?
            .error_for_status()
            .map_err(|e| TurboError::Embedding(format!("HTTP {e}")))?
            .json()
            .await
            .map_err(|e| TurboError::Embedding(format!("JSON parse failed: {e}")))?;

        if resp.data.is_empty() {
            return Err(TurboError::Embedding("empty embeddings in response".into()));
        }
        Ok(resp.data.into_iter().map(|item| item.embedding).collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
