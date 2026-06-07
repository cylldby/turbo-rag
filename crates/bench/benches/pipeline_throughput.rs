//! End-to-end pipeline throughput: docs/sec from text to stored+indexed.
//! SyntheticEmbedder + TurboVecStore (in-memory) = no I/O bottleneck,
//! isolates the rayon + async pipeline overhead.
use bench_helpers::{make_rng, random_unit_vec};
use blob::InMemoryBackend;
use common::{Document, EmbeddingBackend, VectorStore};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use embedder::SyntheticEmbedder;
use pipeline::IngestionPipeline;
use std::sync::Arc;
use store::TurboVecStore;
use tokio::runtime::Runtime;

fn make_docs(n: usize) -> Vec<Document> {
    (0..n as u64)
        .map(|id| Document {
            id,
            text: format!("pipeline benchmark document number {id}"),
            metadata: Default::default(),
        })
        .collect()
}

fn bench_pipeline(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let dim = 768;

    let mut group = c.benchmark_group("pipeline_throughput");

    for &(n, batch_size) in &[(500usize, 64usize), (500, 128), (1000, 64), (1000, 256)] {
        let docs = make_docs(n);
        let label = format!("{n}docs_b{batch_size}");
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("synthetic_turbovec", &label),
            &(),
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let embedder: Arc<dyn EmbeddingBackend> = Arc::new(SyntheticEmbedder::new(dim));
                    let blob = Arc::new(InMemoryBackend::new());
                    let store: Arc<dyn VectorStore> =
                        Arc::new(TurboVecStore::with_blob("bench", blob, dim, 4));
                    let pipeline = IngestionPipeline::new(embedder, store, batch_size);
                    pipeline.run(docs.clone()).await.unwrap()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_pipeline);
criterion_main!(benches);
