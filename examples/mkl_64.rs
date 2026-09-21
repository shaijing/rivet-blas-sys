//! Example for Intel MKL's explicit 64-bit CBLAS symbols.

#[cfg(rivet_blas_mkl)]
fn main() {
    use rivet_blas_sys::cblas::cblas_level_one_mkl_64::cblas_ddot_64;
    use rivet_blas_sys::cblas::cblas_level_three_mkl_64::cblas_dgemm_64;
    use rivet_blas_sys::cblas::cblas_types::{CBlasLayout, CBlasTranspose, MklCBlasInt64};

    let x = [1.0_f64, 2.0, 3.0];
    let y = [4.0_f64, 5.0, 6.0];
    let dot = unsafe { cblas_ddot_64(3, x.as_ptr(), 1, y.as_ptr(), 1) };
    assert_eq!(dot, 32.0);
    println!("MKL ddot_64: {dot}");

    let a = [1.0_f64, 2.0, 3.0, 4.0];
    let b = [5.0_f64, 6.0, 7.0, 8.0];
    let mut c = [0.0_f64; 4];
    unsafe {
        cblas_dgemm_64(
            CBlasLayout::CBlasRowMajor,
            CBlasTranspose::CBlasNoTrans,
            CBlasTranspose::CBlasNoTrans,
            2 as MklCBlasInt64,
            2 as MklCBlasInt64,
            2 as MklCBlasInt64,
            1.0,
            a.as_ptr(),
            2 as MklCBlasInt64,
            b.as_ptr(),
            2 as MklCBlasInt64,
            0.0,
            c.as_mut_ptr(),
            2 as MklCBlasInt64,
        );
    }
    assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    println!("MKL dgemm_64: {c:?}");
}

#[cfg(not(rivet_blas_mkl))]
fn main() {
    eprintln!("This example requires an `mkl-*` feature.");
}
