//! Ordinary and explicit 64-bit Intel oneMKL Sparse BLAS.

#[cfg(rivet_blas_mkl)]
fn main() {
    use rivet_blas_sys::mkl::{prelude::*, sparse_64};

    // A = [2 0; 1 3]. Keep all three arrays alive until the handle is destroyed.
    let mut offsets: [MklInt; 3] = [0, 1, 3];
    let mut indices: [MklInt; 3] = [0, 0, 1];
    let mut values = [2.0_f64, 1.0, 3.0];
    let x = [1.0_f64, 2.0];
    let mut y = [0.0_f64; 2];
    let mut a = std::ptr::null_mut();
    let mut product = std::ptr::null_mut();

    unsafe {
        assert_eq!(
            mkl_sparse_d_create_csr(
                &mut a,
                SparseIndexBase::Zero,
                2,
                2,
                offsets.as_mut_ptr(),
                offsets.as_mut_ptr().add(1),
                indices.as_mut_ptr(),
                values.as_mut_ptr(),
            ),
            SparseStatus::SUCCESS,
        );
        assert_eq!(
            mkl_sparse_set_mv_hint(a, SparseOperation::NonTranspose, MatrixDescr::default(), 10),
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
        assert_eq!(y, [2.0, 7.0]);
        println!("Sparse A * x = {y:?}");

        assert_eq!(
            mkl_sparse_sp2m(
                SparseOperation::NonTranspose,
                MatrixDescr::default(),
                a,
                SparseOperation::NonTranspose,
                MatrixDescr::default(),
                a,
                SparseRequest::FullMult,
                &mut product
            ),
            SparseStatus::SUCCESS
        );
        assert_eq!(
            mkl_sparse_d_mv(
                SparseOperation::NonTranspose,
                1.0,
                product,
                MatrixDescr::default(),
                x.as_ptr(),
                0.0,
                y.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
        assert_eq!(y, [4.0, 23.0]);
        println!("Sparse A² * x = {y:?}");
        assert_eq!(mkl_sparse_destroy(product), SparseStatus::SUCCESS);
        assert_eq!(mkl_sparse_destroy(a), SparseStatus::SUCCESS);
    }

    // Explicit *_64 calls always take 64-bit index arrays, even in an LP64 build.
    let mut offsets64: [MklInt64; 3] = [0, 1, 3];
    let mut indices64: [MklInt64; 3] = [0, 0, 1];
    let mut a64 = std::ptr::null_mut();
    unsafe {
        assert_eq!(
            sparse_64::mkl_sparse_d_create_csr_64(
                &mut a64,
                SparseIndexBase::Zero,
                2,
                2,
                offsets64.as_mut_ptr(),
                offsets64.as_mut_ptr().add(1),
                indices64.as_mut_ptr(),
                values.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
        assert_eq!(
            sparse_64::mkl_sparse_d_mv_64(
                SparseOperation::NonTranspose,
                1.0,
                a64,
                MatrixDescr::default(),
                x.as_ptr(),
                0.0,
                y.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
        assert_eq!(sparse_64::mkl_sparse_destroy_64(a64), SparseStatus::SUCCESS);
    }
    assert_eq!(y, [2.0, 7.0]);
    println!("Sparse A * x with 64-bit indices = {y:?}");
}

#[cfg(not(rivet_blas_mkl))]
fn main() {
    eprintln!("Run with --no-default-features -F mkl-dynamic-lp64-seq --example mkl_sparse");
}
