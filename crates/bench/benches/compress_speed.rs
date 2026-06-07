//! Compression throughput: ns/vector for add_batch at various dim × bits.
use bench_helpers::{make_rng, random_unit_vec};
use compressor::QuantizedIndex;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

const BATCH: usize = 512;

fn bench_compress_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress/add_batch");

    for &dim in &[384usize, 768, 1536] {
        for &bits in &[2usize, 4] {
            let mut rng = make_rng(42);
            let ids: Vec<u64> = (0..BATCH as u64).collect();
            let vecs: Vec<Vec<f32>> = (0..BATCH).map(|_| random_unit_vec(dim, &mut rng)).collect();

            group.throughput(Throughput::Elements(BATCH as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("dim{dim}_{bits}bit"), BATCH),
                &(),
                |b, _| {
                    b.iter(|| {
                        let mut idx = QuantizedIndex::new(dim, bits);
                        idx.add_batch(&ids, &vecs);
                        idx
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_compress_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress/serialize");

    for &(dim, bits) in &[(768usize, 4usize), (1536, 4), (768, 2)] {
        let mut rng = make_rng(42);
        let n = 10_000;
        let ids: Vec<u64> = (0..n as u64).collect();
        let vecs: Vec<Vec<f32>> = (0..n).map(|_| random_unit_vec(dim, &mut rng)).collect();
        let mut idx = QuantizedIndex::new(dim, bits);
        idx.add_batch(&ids, &vecs);

        group.bench_with_input(
            BenchmarkId::new(format!("to_bytes_dim{dim}_{bits}bit"), n),
            &(),
            |b, _| {
                b.iter(|| idx.to_bytes().unwrap());
            },
        );
    }
    group.finish();
}

fn bench_compression_ratio(c: &mut Criterion) {
    // This "benchmark" just runs once and prints the ratio table — useful as a quick sanity check.
    let mut group = c.benchmark_group("compress/ratio");

    for &dim in &[384usize, 768, 1536] {
        for &bits in &[2usize, 4] {
            let mut rng = make_rng(42);
            let ids: Vec<u64> = (0..BATCH as u64).collect();
            let vecs: Vec<Vec<f32>> = (0..BATCH).map(|_| random_unit_vec(dim, &mut rng)).collect();
            let mut idx = QuantizedIndex::new(dim, bits);
            idx.add_batch(&ids, &vecs);
            let info = idx.compression_info();

            group.bench_with_input(
                BenchmarkId::new(format!("dim{dim}_{bits}bit"), BATCH),
                &(),
                |b, _| {
                    b.iter(|| {
                        // ratio reported in throughput label
                        info.ratio
                    });
                },
            );

            println!(
                "dim={dim:>4}  bits={bits}  ratio={:.1}x  {:.0}B/vec → {:.0}B/vec",
                info.ratio, info.original_bytes_per_vec, info.compressed_bytes_per_vec
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_compress_add,
    bench_compress_roundtrip,
    bench_compression_ratio
);
criterion_main!(benches);
