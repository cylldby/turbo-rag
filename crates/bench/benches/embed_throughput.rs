//! Embedding throughput: docs/sec at various batch sizes.
//! Uses SyntheticEmbedder (always available) so no API key or model download needed.
use bench_helpers::make_rng;
use common::EmbeddingBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use embedder::SyntheticEmbedder;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn make_texts(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("document number {i} for embedding benchmark")).collect()
}

fn bench_synthetic_embed(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    for &dim in &[384usize, 768, 1536] {
        let embedder: Arc<dyn EmbeddingBackend> = Arc::new(SyntheticEmbedder::new(dim));
        let mut group = c.benchmark_group(format!("embed_throughput/dim{dim}"));

        for &batch in &[8usize, 32, 128, 512] {
            let texts = make_texts(batch);
            group.throughput(Throughput::Elements(batch as u64));
            group.bench_with_input(BenchmarkId::new("synthetic", batch), &(), |b, _| {
                b.to_async(&rt).iter(|| async {
                    embedder.embed_batch(&texts).await.unwrap()
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_synthetic_embed);
criterion_main!(benches);
