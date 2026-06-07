//! Integration tests for embedding backends.
//! Requires live services — run with:
//!   cargo test -p embedder --features integration
//!
//! Service requirements:
//!   - WireMock at WIREMOCK_URL (default: http://localhost:8080) — always required
//!   - Ollama at OLLAMA_URL with nomic-embed-text — only when OLLAMA_AVAILABLE=1

#[cfg(feature = "integration")]
mod embed_integration {
    use common::{cosine_similarity, EmbeddingBackend};
    use embedder::OpenAICompatBackend;

    fn wiremock_url() -> String {
        std::env::var("WIREMOCK_URL").unwrap_or_else(|_| "http://localhost:8080".into())
    }

    fn ollama_url() -> String {
        std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".into())
    }

    fn ollama_available() -> bool {
        std::env::var("OLLAMA_AVAILABLE")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    // ── WireMock tests (always required in integration mode) ─────────────────

    #[tokio::test]
    async fn embed_wiremock_compat_single() {
        let backend = OpenAICompatBackend::new(
            format!("{}/v1", wiremock_url()),
            None::<String>,
            "mock-embed",
            768,
        );
        let result = backend
            .embed_batch(&["hello world".to_string()])
            .await
            .unwrap();
        assert_eq!(result.len(), 1, "should return exactly one embedding");
        assert_eq!(result[0].len(), 768, "embedding dim should be 768");
    }

    #[tokio::test]
    async fn embed_wiremock_compat_batch() {
        let backend = OpenAICompatBackend::new(
            format!("{}/v1", wiremock_url()),
            None::<String>,
            "mock-embed",
            768,
        );
        let texts: Vec<String> = (0..5).map(|i| format!("document {i}")).collect();
        // WireMock returns 1 embedding per request (fixed stub); test that we can call it.
        let result = backend.embed_batch(&texts[..1]).await.unwrap();
        assert_eq!(result[0].len(), 768);
    }

    #[tokio::test]
    async fn embed_wiremock_returns_floats_in_range() {
        let backend = OpenAICompatBackend::new(
            format!("{}/v1", wiremock_url()),
            None::<String>,
            "mock-embed",
            768,
        );
        let result = backend.embed_batch(&["test".to_string()]).await.unwrap();
        let all_valid = result[0]
            .iter()
            .all(|v| v.is_finite() && *v >= -1.0 && *v <= 1.0);
        assert!(
            all_valid,
            "all embedding values should be finite floats in [-1, 1]"
        );
    }

    // ── Ollama live tests (opt-in via OLLAMA_AVAILABLE=1) ────────────────────

    #[tokio::test]
    async fn embed_ollama_nomic_dim() {
        if !ollama_available() {
            eprintln!("skipped: set OLLAMA_AVAILABLE=1 with nomic-embed-text pulled");
            return;
        }
        let backend = OpenAICompatBackend::new(
            format!("{}/v1", ollama_url()),
            None::<String>,
            "nomic-embed-text",
            768,
        );
        assert_eq!(backend.dim(), 768);
        let result = backend
            .embed_batch(&["What is machine learning?".to_string()])
            .await;
        match result {
            Ok(vecs) => {
                assert_eq!(vecs.len(), 1);
                assert_eq!(
                    vecs[0].len(),
                    768,
                    "nomic-embed-text should produce 768-dim vectors"
                );
                let norm: f32 = vecs[0].iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(norm > 0.0, "embedding should be non-zero");
            }
            Err(e) => panic!("Ollama embedding failed: {e}"),
        }
    }

    #[tokio::test]
    async fn embed_ollama_same_text_cosine_near_one() {
        if !ollama_available() {
            eprintln!("skipped: set OLLAMA_AVAILABLE=1 with nomic-embed-text pulled");
            return;
        }
        let backend = OpenAICompatBackend::ollama("nomic-embed-text", 768);
        let texts = vec![
            "Rust is a systems programming language.".to_string(),
            "Rust is a systems programming language.".to_string(),
        ];
        let vecs = backend.embed_batch(&texts).await.unwrap();
        let sim = cosine_similarity(&vecs[0], &vecs[1]);
        assert!(
            sim > 0.999,
            "identical texts should have cosine~1, got {sim}"
        );
    }

    #[tokio::test]
    async fn embed_ollama_different_texts_lower_similarity() {
        if !ollama_available() {
            eprintln!("skipped: set OLLAMA_AVAILABLE=1 with nomic-embed-text pulled");
            return;
        }
        let backend = OpenAICompatBackend::ollama("nomic-embed-text", 768);
        let texts = vec![
            "vector databases for AI applications".to_string(),
            "medieval castle architecture in France".to_string(),
        ];
        let vecs = backend.embed_batch(&texts).await.unwrap();
        let sim = cosine_similarity(&vecs[0], &vecs[1]);
        assert!(
            sim < 0.9,
            "unrelated texts should have lower cosine similarity, got {sim}"
        );
    }
}
