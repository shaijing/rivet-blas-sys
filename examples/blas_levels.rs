//! Small end-to-end example covering the common CBLAS Level 1, 2, and 3 APIs.

use rivet_blas_sys::cblas::prelude::*;

fn main() {
    level_one_axpy();
    level_two_gemv();
    level_three_gemm();
}

fn level_one_axpy() {
    let x = [1.0_f32, 2.0, 3.0, 4.0];
    let mut y = [5.0_f32, 6.0, 7.0, 8.0];

    // y := 2.0 * x + y
    unsafe {
        cblas_saxpy(x.len() as CBlasInt, 2.0, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }

    assert_eq!(y, [7.0, 10.0, 13.0, 16.0]);
    println!("Level 1 saxpy: {y:?}");
}

fn level_two_gemv() {
    // A = [[1, 2, 3], [4, 5, 6]], x = [1, 2, 1]
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = [1.0_f32, 2.0, 1.0];
    let mut y = [0.0_f32, 0.0];

    // y := A * x
    unsafe {
        cblas_sgemv(
            CBlasLayout::CBlasRowMajor,
            CBlasTranspose::CBlasNoTrans,
            2,
            3,
            1.0,
            a.as_ptr(),
            3,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }

    assert_eq!(y, [8.0, 20.0]);
    println!("Level 2 sgemv: {y:?}");
}

fn level_three_gemm() {
    // A is 2x3, B is 3x2, and C is 2x2 in row-major storage.
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut c = [0.0_f32; 4];

    // C := A * B
    unsafe {
        cblas_sgemm(
            CBlasLayout::CBlasRowMajor,
            CBlasTranspose::CBlasNoTrans,
            CBlasTranspose::CBlasNoTrans,
            2,
            2,
            3,
            1.0,
            a.as_ptr(),
            3,
            b.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
    }

    assert_eq!(c, [22.0, 28.0, 49.0, 64.0]);
    println!("Level 3 sgemm: {c:?}");
}
