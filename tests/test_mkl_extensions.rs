#![cfg(any(
    feature = "mkl-dynamic-ilp64-gomp",
    feature = "mkl-dynamic-ilp64-iomp",
    feature = "mkl-dynamic-ilp64-seq",
    feature = "mkl-dynamic-ilp64-tbb",
    feature = "mkl-dynamic-lp64-gomp",
    feature = "mkl-dynamic-lp64-iomp",
    feature = "mkl-dynamic-lp64-seq",
    feature = "mkl-dynamic-lp64-tbb",
    feature = "mkl-static-ilp64-gomp",
    feature = "mkl-static-ilp64-iomp",
    feature = "mkl-static-ilp64-seq",
    feature = "mkl-static-ilp64-tbb",
    feature = "mkl-static-lp64-gomp",
    feature = "mkl-static-lp64-iomp",
    feature = "mkl-static-lp64-seq",
    feature = "mkl-static-lp64-tbb",
    feature = "mkl-sdl",
))]

use rivet_blas_sys::cblas::cblas_level_one_mkl::cblas_isamin;
use rivet_blas_sys::cblas::cblas_level_three_mkl::cblas_strmm_oop;
use rivet_blas_sys::cblas::cblas_types::{
    CBlasDiag, CBlasLayout, CBlasSide, CBlasTranspose, CBlasUplo,
};

#[test]
fn mkl_ordinary_extension_supports_iamin() {
    let x = [3.0_f32, -1.0, 2.0];
    let index = unsafe { cblas_isamin(3, x.as_ptr(), 1) };

    assert_eq!(index, 1);
}

#[test]
fn mkl_trmm_oop_writes_to_a_separate_matrix() {
    let a = [2.0_f32, 0.0, 1.0, 3.0];
    let b = [1.0_f32, 2.0, 3.0, 4.0];
    let mut c = [0.0_f32; 4];

    unsafe {
        cblas_strmm_oop(
            CBlasLayout::CBlasRowMajor,
            CBlasSide::CblasLeft,
            CBlasUplo::CblasLower,
            CBlasTranspose::CBlasNoTrans,
            CBlasDiag::CblasNonUnit,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
    }

    assert_eq!(c, [2.0, 4.0, 10.0, 14.0]);
}
