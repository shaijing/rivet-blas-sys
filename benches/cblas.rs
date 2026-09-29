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

#[cfg(rivet_blas_mkl)]
fn bench_mkl_sparse_mv(c: &mut Criterion) {
    use rivet_blas_sys::mkl::prelude::*;

    let mut group = c.benchmark_group("mkl/sparse/dmv");
    for rows in [1024_usize, 16_384] {
        let mut offsets = Vec::<MklInt>::with_capacity(rows + 1);
        let mut indices = Vec::<MklInt>::with_capacity(rows * 3);
        let mut values = Vec::<f64>::with_capacity(rows * 3);
        offsets.push(0);
        for row in 0..rows {
            for col in row.saturating_sub(1)..=(row + 1).min(rows - 1) {
                indices.push(col as MklInt);
                values.push(if col == row { 2.0 } else { -1.0 });
            }
            offsets.push(indices.len() as MklInt);
        }
        let x = vec![1.0_f64; rows];
        let mut y = vec![0.0_f64; rows];
        let mut a = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                mkl_sparse_d_create_csr(
                    &mut a,
                    SparseIndexBase::Zero,
                    rows as MklInt,
                    rows as MklInt,
                    offsets.as_mut_ptr(),
                    offsets.as_mut_ptr().add(1),
                    indices.as_mut_ptr(),
                    values.as_mut_ptr()
                ),
                SparseStatus::SUCCESS
            );
            assert_eq!(
                mkl_sparse_set_mv_hint(
                    a,
                    SparseOperation::NonTranspose,
                    MatrixDescr::default(),
                    1000
                ),
                SparseStatus::SUCCESS
            );
            assert_eq!(mkl_sparse_optimize(a), SparseStatus::SUCCESS);
            assert_eq!(
                mkl_sparse_d_mv(
                    SparseOperation::NonTranspose,
                    1.0,
                    a,
                    MatrixDescr::default(),
                    x.as_ptr(),
                    0.0,
                    y.as_mut_ptr()
                ),
                SparseStatus::SUCCESS
            );
        }
        assert_eq!(y[0], 1.0);
        assert!(y[1..rows - 1].iter().all(|&v| v == 0.0));
        assert_eq!(y[rows - 1], 1.0);
        group.bench_with_input(BenchmarkId::from_parameter(rows), &rows, |bench, _| {
            bench.iter(|| unsafe {
                black_box(mkl_sparse_d_mv(
                    SparseOperation::NonTranspose,
                    1.0,
                    a,
                    MatrixDescr::default(),
                    x.as_ptr(),
                    0.0,
                    y.as_mut_ptr(),
                ));
                black_box(y[0]);
            });
        });
        unsafe {
            assert_eq!(mkl_sparse_destroy(a), SparseStatus::SUCCESS);
        }
    }
    group.finish();
}

#[cfg(rivet_blas_mkl)]
criterion_group!(benches, bench_sdot, bench_dgemm, bench_mkl_sparse_mv);
#[cfg(not(rivet_blas_mkl))]
criterion_group!(benches, bench_sdot, bench_dgemm);

criterion_main!(benches);
