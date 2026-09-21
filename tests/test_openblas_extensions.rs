#![cfg(any(rivet_blas_openblas, rivet_blas_flexiblas))]

use rivet_blas_sys::cblas::cblas_level_one_openblas::cblas_saxpby;
use rivet_blas_sys::cblas::cblas_level_three_openblas::{cblas_sgeadd, cblas_somatcopy};
use rivet_blas_sys::cblas::cblas_types::{CBlasLayout, CBlasTranspose};

#[test]
fn openblas_saxpby_updates_y_with_two_scalars() {
    let x = [1.0_f32, 2.0, 3.0];
    let mut y = [10.0_f32, 20.0, 30.0];

    unsafe {
        cblas_saxpby(3, 2.0, x.as_ptr(), 1, 0.5, y.as_mut_ptr(), 1);
    }

    assert_eq!(y, [7.0, 14.0, 21.0]);
}

#[test]
fn openblas_somatcopy_copies_a_matrix() {
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let mut b = [0.0_f32; 4];

    unsafe {
        cblas_somatcopy(
            CBlasLayout::CBlasRowMajor,
            CBlasTranspose::CBlasNoTrans,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            b.as_mut_ptr(),
            2,
        );
    }

    assert_eq!(b, a);
}

#[test]
fn openblas_sgeadd_combines_two_matrices() {
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let mut c = [5.0_f32, 6.0, 7.0, 8.0];

    unsafe {
        cblas_sgeadd(
            CBlasLayout::CBlasRowMajor,
            CBlasTranspose::CBlasNoTrans,
            CBlasTranspose::CBlasNoTrans,
            2,
            2,
            2.0,
            a.as_ptr(),
            2,
            3.0,
            c.as_mut_ptr(),
            2,
        );
    }

    assert_eq!(c, [17.0, 22.0, 27.0, 32.0]);
}
