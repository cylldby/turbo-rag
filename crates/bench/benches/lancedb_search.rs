//! LanceDB ANN search latency — companion to turbovec_search for comparison.
use bench_helpers::{make_rng, random_unit_vec};
use common::{EmbeddedDoc, SearchRequest, VectorStore};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use store::LanceDbStore;
use tokio::runtime::Runtime;

fn build_lance_store(n: usize, dim: usize) -> (Arc<LanceDbStore>, tempfile::TempDir) {
    let rt = Runtime::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let store = rt
        .block_on(LanceDbStore::new(
            dir.path().to_str().unwrap(),
            "bench",
            dim,
        ))
        .unwrap();
    let store = Arc::new(store);

    let mut rng = make_rng(42);
    let docs: Vec<EmbeddedDoc> = (0..n as u64)
        .map(|id| EmbeddedDoc {
            id,
            text: format!("doc {id}"),
            embedding: random_unit_vec(dim, &mut rng),
            metadata: Default::default(),
        })
        .collect();
    rt.block_on(store.upsert(&docs)).unwrap();
    (store, dir)
}

fn bench_lance_search(c: &mut Criterion) {
    let dim = 768;
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("lancedb_search/scale");
    group.throughput(Throughput::Elements(1));

    for &n in &[1_000usize, 10_000] {
        let (store, _dir) = build_lance_store(n, dim);
        let mut rng = make_rng(99);
        let query = random_unit_vec(dim, &mut rng);
        let label = format!("{n}_docs");
        group.bench_with_input(BenchmarkId::new("k5", &label), &(), |b, _| {
            b.to_async(&rt).iter(|| async {
                let req = SearchRequest::vector(&query, 5);
                (store.clone() as Arc<dyn VectorStore>)
                    .search(&req)
                    .await
                    .unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_lance_search);
criterion_main!(benches);
