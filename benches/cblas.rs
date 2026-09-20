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

#[cfg(all(
    any(feature = "openblas", feature = "flexiblas"),
    not(feature = "intel-mkl")
))]
fn bench_openblas_saxpby(c: &mut Criterion) {
    use rivet_blas_sys::cblas::cblas_level_one_openblas::cblas_saxpby;

    let mut group = c.benchmark_group("openblas/saxpby");
    let size = 16_384_i64;
    let x: Vec<f32> = (0..size).map(|i| (i % 17) as f32).collect();
    let initial_y: Vec<f32> = (0..size).map(|i| (i % 11) as f32).collect();
    let mut y = initial_y.clone();

    group.bench_function(BenchmarkId::from_parameter(size), |bench| {
        bench.iter(|| unsafe {
            y.copy_from_slice(&initial_y);
            cblas_saxpby(size as CBlasInt, 2.0, x.as_ptr(), 1, 0.5, y.as_mut_ptr(), 1);
            black_box(y[0]);
        });
    });
    group.finish();
}

#[cfg(feature = "intel-mkl")]
fn bench_mkl_dgemm_64(c: &mut Criterion) {
    use rivet_blas_sys::cblas::cblas_level_three_mkl_64::cblas_dgemm_64;
    use rivet_blas_sys::cblas::cblas_types::MklCBlasInt64;

    let mut group = c.benchmark_group("intel-mkl/dgemm_64");
    let size = 128_i64;
    let elements = (size * size) as usize;
    let a = vec![1.0_f64; elements];
    let b = vec![2.0_f64; elements];
    let mut result = vec![0.0_f64; elements];

    group.bench_function(BenchmarkId::from_parameter(size), |bench| {
        bench.iter(|| unsafe {
            cblas_dgemm_64(
                CBlasLayout::CBlasRowMajor,
                CBlasTranspose::CBlasNoTrans,
                CBlasTranspose::CBlasNoTrans,
                size as MklCBlasInt64,
                size as MklCBlasInt64,
                size as MklCBlasInt64,
                1.0,
                a.as_ptr(),
                size as MklCBlasInt64,
                b.as_ptr(),
                size as MklCBlasInt64,
                0.0,
                result.as_mut_ptr(),
                size as MklCBlasInt64,
            );
            black_box(result[0]);
        });
    });
    group.finish();
}

#[cfg(all(
    any(feature = "openblas", feature = "flexiblas"),
    not(feature = "intel-mkl")
))]
criterion_group!(benches, bench_sdot, bench_dgemm, bench_openblas_saxpby);

#[cfg(feature = "intel-mkl")]
criterion_group!(benches, bench_sdot, bench_dgemm, bench_mkl_dgemm_64);

#[cfg(not(any(feature = "openblas", feature = "flexiblas", feature = "intel-mkl")))]
criterion_group!(benches, bench_sdot, bench_dgemm);

criterion_main!(benches);
