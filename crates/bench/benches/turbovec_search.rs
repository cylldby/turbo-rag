//! Headline benchmark: turbovec SIMD search latency vs corpus scale.
//! Uses QuantizedIndex directly (zero async overhead) for accurate numbers.
use bench_helpers::{make_rng, random_unit_vec};
use compressor::QuantizedIndex;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn build_index(n: usize, dim: usize, bits: usize) -> QuantizedIndex {
    let mut rng = make_rng(42);
    let mut idx = QuantizedIndex::new(dim, bits);
    let ids: Vec<u64> = (0..n as u64).collect();
    let vecs: Vec<Vec<f32>> = (0..n).map(|_| random_unit_vec(dim, &mut rng)).collect();
    idx.add_batch(&ids, &vecs);
    idx
}

fn bench_search_by_scale(c: &mut Criterion) {
    let dim = 768;

    let mut group = c.benchmark_group("turbovec_search/scale");
    group.throughput(Throughput::Elements(1));

    for &(n, bits) in &[(1_000, 4), (10_000, 4), (100_000, 4)] {
        let idx = build_index(n, dim, bits);
        let mut rng = make_rng(99);
        let query = random_unit_vec(dim, &mut rng);
        let label = format!("{n}_docs_{bits}bit");
        group.bench_with_input(BenchmarkId::new("k5", &label), &(), |b, _| {
            b.iter(|| idx.search(&query, 5));
        });
    }
    group.finish();
}

fn bench_search_by_bits(c: &mut Criterion) {
    let dim = 768;
    let n = 10_000;

    let mut group = c.benchmark_group("turbovec_search/bits");
    group.throughput(Throughput::Elements(1));

    for &bits in &[2usize, 4] {
        let idx = build_index(n, dim, bits);
        let mut rng = make_rng(99);
        let query = random_unit_vec(dim, &mut rng);
        group.bench_with_input(BenchmarkId::new("k5", bits), &(), |b, _| {
            b.iter(|| idx.search(&query, 5));
        });
    }
    group.finish();
}

fn bench_search_by_k(c: &mut Criterion) {
    let dim = 768;
    let n = 10_000;
    let bits = 4;
    let idx = build_index(n, dim, bits);
    let mut rng = make_rng(99);
    let query = random_unit_vec(dim, &mut rng);

    let mut group = c.benchmark_group("turbovec_search/k");
    group.throughput(Throughput::Elements(1));

    for &k in &[1usize, 5, 10, 50] {
        group.bench_with_input(BenchmarkId::new("10k_docs", k), &k, |b, &k| {
            b.iter(|| idx.search(&query, k));
        });
    }
    group.finish();
}

fn bench_search_by_dim(c: &mut Criterion) {
    let n = 10_000;
    let bits = 4;

    let mut group = c.benchmark_group("turbovec_search/dim");
    group.throughput(Throughput::Elements(1));

    for &dim in &[384usize, 768, 1536] {
        let idx = build_index(n, dim, bits);
        let mut rng = make_rng(99);
        let query = random_unit_vec(dim, &mut rng);
        group.bench_with_input(BenchmarkId::new("k5_10k", dim), &(), |b, _| {
            b.iter(|| idx.search(&query, 5));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_search_by_scale,
    bench_search_by_bits,
    bench_search_by_k,
    bench_search_by_dim
);
criterion_main!(benches);
