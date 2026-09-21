use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rivet_blas_sys::cblas::prelude::*;

fn bench_sdot(c: &mut Criterion) {
    let mut group = c.benchmark_group("level1/sdot");

    for size in [1024_i64, 16_384] {
        let x: Vec<f32> = (0..size).map(|i| (i % 17) as f32).collect();
        let y: Vec<f32> = (0..size).map(|i| (i % 11) as f32).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| unsafe {
                black_box(cblas_sdot(size as CBlasInt, x.as_ptr(), 1, y.as_ptr(), 1))
            });
        });
    }

    group.finish();
}

fn bench_dgemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("level3/dgemm");

    for size in [32_i64, 64, 128] {
        let elements = (size * size) as usize;
        let a = vec![1.0_f64; elements];
        let b = vec![2.0_f64; elements];
        let mut result = vec![0.0_f64; elements];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, &size| {
            bench.iter(|| unsafe {
                cblas_dgemm(
                    CBlasLayout::CBlasRowMajor,
                    CBlasTranspose::CBlasNoTrans,
                    CBlasTranspose::CBlasNoTrans,
                    size as CBlasInt,
                    size as CBlasInt,
                    size as CBlasInt,
                    1.0,
                    a.as_ptr(),
                    size as CBlasInt,
                    b.as_ptr(),
                    size as CBlasInt,
                    0.0,
                    result.as_mut_ptr(),
                    size as CBlasInt,
                );
                black_box(result[0]);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_sdot, bench_dgemm);

criterion_main!(benches);
