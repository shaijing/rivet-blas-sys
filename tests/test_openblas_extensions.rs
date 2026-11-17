#![cfg(any(
    feature = "openblas-dynamic-ilp64",
    feature = "openblas-dynamic-lp64",
    feature = "openblas-static-ilp64",
    feature = "openblas-static-lp64",
    feature = "flexiblas-dynamic-ilp64",
    feature = "flexiblas-dynamic-lp64",
))]

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
        #[cfg(rivet_blas_geadd_transpose)]
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

        #[cfg(not(rivet_blas_geadd_transpose))]
        cblas_sgeadd(
            CBlasLayout::CBlasRowMajor,
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

// Exercise all four declarations with rectangular, padded matrices in both
// layouts. Extra storage stays unchanged, catching incorrect leading dimensions.
macro_rules! geadd_case {
    ($test:ident, $function:ident, $scalar:ty, $complex:tt) => {
        #[test]
        fn $test() {
            use rivet_blas_sys::cblas::cblas_level_three_openblas::$function;
            for layout in [CBlasLayout::CBlasRowMajor, CBlasLayout::CBlasColMajor] {
                let width = if $complex { 2 } else { 1 };
                let mut a = vec![99 as $scalar; 12 * width];
                let mut c = vec![77 as $scalar; 12 * width];
                let mut expected = c.clone();
                for row in 0..2 {
                    for col in 0..3 {
                        let index = if matches!(layout, CBlasLayout::CBlasRowMajor) {
                            row * 4 + col
                        } else {
                            col * 4 + row
                        } * width;
                        a[index] = (row * 3 + col + 1) as $scalar;
                        c[index] = (row * 3 + col + 5) as $scalar;
                        expected[index] = 2 as $scalar * a[index] + 3 as $scalar * c[index];
                        if $complex {
                            a[index + 1] = 1 as $scalar;
                            c[index + 1] = 2 as $scalar;
                            expected[index + 1] = 8 as $scalar;
                        }
                    }
                }
                // Complex coefficients use interleaved real/imaginary storage.
                let alpha = [2 as $scalar, 0 as $scalar];
                let beta = [3 as $scalar, 0 as $scalar];
                unsafe {
                    geadd_case!(@call $function, layout, alpha, beta, a, c, $complex);
                }
                assert_eq!(c, expected);
            }
        }
    };
    (@call $function:ident, $layout:ident, $alpha:ident, $beta:ident, $a:ident, $c:ident, false) => {
        #[cfg(rivet_blas_geadd_transpose)]
        $function($layout, CBlasTranspose::CBlasNoTrans, CBlasTranspose::CBlasNoTrans,
            2, 3, $alpha[0], $a.as_ptr(), 4, $beta[0], $c.as_mut_ptr(), 4);
        #[cfg(not(rivet_blas_geadd_transpose))]
        $function($layout, 2, 3, $alpha[0], $a.as_ptr(), 4, $beta[0], $c.as_mut_ptr(), 4);
    };
    (@call $function:ident, $layout:ident, $alpha:ident, $beta:ident, $a:ident, $c:ident, true) => {
        #[cfg(rivet_blas_geadd_transpose)]
        $function($layout, CBlasTranspose::CBlasNoTrans, CBlasTranspose::CBlasNoTrans,
            2, 3, $alpha.as_ptr().cast(), $a.as_ptr().cast(), 4,
            $beta.as_ptr().cast(), $c.as_mut_ptr().cast(), 4);
        #[cfg(not(rivet_blas_geadd_transpose))]
        $function($layout, 2, 3, $alpha.as_ptr().cast(), $a.as_ptr().cast(), 4,
            $beta.as_ptr().cast(), $c.as_mut_ptr().cast(), 4);
    };
}

geadd_case!(sgeadd_padded_layouts, cblas_sgeadd, f32, false);
geadd_case!(dgeadd_padded_layouts, cblas_dgeadd, f64, false);
geadd_case!(cgeadd_padded_layouts, cblas_cgeadd, f32, true);
geadd_case!(zgeadd_padded_layouts, cblas_zgeadd, f64, true);
