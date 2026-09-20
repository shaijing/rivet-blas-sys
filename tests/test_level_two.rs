//! Integration tests for BLAS Level 2 (matrix-vector) operations.
//! Requires a BLAS backend: `cargo test -F intel-mkl` or `cargo test -F openblas`

#[cfg(any(
    feature = "intel-mkl",
    feature = "openblas",
    feature = "netlib",
    feature = "accelerate"
))]
use rivet_blas_sys::cblas::prelude::*;

#[cfg(any(
    feature = "intel-mkl",
    feature = "openblas",
    feature = "netlib",
    feature = "accelerate"
))]
mod level_two {
    use super::*;

    #[test]
    fn test_sgemv() {
        // A = [[1, 2], [3, 4]] (row-major), x = [1, 2], y = [0, 0]
        // y := 1.0 * A * x + 0.0 * y = [5, 11]
        let layout = CBlasLayout::CBlasRowMajor;
        let trans = CBlasTranspose::CBlasNoTrans;
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let x = [1.0f32, 2.0];
        let mut y = [0.0f32; 2];
        unsafe {
            cblas_sgemv(
                layout,
                trans,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                0.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert!(
            (y[0] - 5.0f32).abs() < 1e-4,
            "sgemv[0]: expected 5.0, got {}",
            y[0]
        );
        assert!(
            (y[1] - 11.0f32).abs() < 1e-4,
            "sgemv[1]: expected 11.0, got {}",
            y[1]
        );
    }

    #[test]
    fn test_dgemv() {
        let layout = CBlasLayout::CBlasRowMajor;
        let trans = CBlasTranspose::CBlasNoTrans;
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let x = [1.0f64, 2.0];
        let mut y = [0.0f64; 2];
        unsafe {
            cblas_dgemv(
                layout,
                trans,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                0.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert!(
            (y[0] - 5.0f64).abs() < 1e-10,
            "dgemv[0]: expected 5.0, got {}",
            y[0]
        );
        assert!(
            (y[1] - 11.0f64).abs() < 1e-10,
            "dgemv[1]: expected 11.0, got {}",
            y[1]
        );
    }

    #[test]
    fn test_sgemv_transpose() {
        // A = [[1, 3], [2, 4]] (row-major), so A^T * x with A stored row-major
        // Test: y := A^T * x where A = [[1,2],[3,4]]
        // A^T * [1,2] = [1*1+3*2, 2*1+4*2] = [7, 10]
        let layout = CBlasLayout::CBlasRowMajor;
        let trans = CBlasTranspose::CBlasTrans;
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let x = [1.0f32, 2.0];
        let mut y = [0.0f32; 2];
        unsafe {
            cblas_sgemv(
                layout,
                trans,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                0.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert!(
            (y[0] - 7.0f32).abs() < 1e-4,
            "sgemv_trans[0]: expected 7.0, got {}",
            y[0]
        );
        assert!(
            (y[1] - 10.0f32).abs() < 1e-4,
            "sgemv_trans[1]: expected 10.0, got {}",
            y[1]
        );
    }

    #[test]
    fn test_sger() {
        // A := 1.0 * x * y^T + A, with x=[1,2], y=[3,4], A=[[0,0],[0,0]]
        // Result: [[3,4],[6,8]]
        let layout = CBlasLayout::CBlasRowMajor;
        let x = [1.0f32, 2.0];
        let y = [3.0f32, 4.0];
        let mut a = [0.0f32; 4];
        unsafe {
            cblas_sger(
                layout,
                2,
                2,
                1.0,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                a.as_mut_ptr(),
                2,
            );
        }
        let expected = [3.0f32, 4.0, 6.0, 8.0];
        for i in 0..4 {
            assert!(
                (a[i] - expected[i]).abs() < 1e-4,
                "sger[{i}]: expected {}, got {}",
                expected[i],
                a[i]
            );
        }
    }

    #[test]
    fn test_dger() {
        let layout = CBlasLayout::CBlasRowMajor;
        let x = [1.0f64, 2.0];
        let y = [3.0f64, 4.0];
        let mut a = [0.0f64; 4];
        unsafe {
            cblas_dger(
                layout,
                2,
                2,
                1.0,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                a.as_mut_ptr(),
                2,
            );
        }
        let expected = [3.0f64, 4.0, 6.0, 8.0];
        for i in 0..4 {
            assert!(
                (a[i] - expected[i]).abs() < 1e-10,
                "dger[{i}]: expected {}, got {}",
                expected[i],
                a[i]
            );
        }
    }

    #[test]
    fn test_strmv() {
        // Upper triangular A = [[1,2],[0,3]], x = [1,1]
        // A * x = [1*1+2*1, 0*1+3*1] = [3, 3]
        let layout = CBlasLayout::CBlasRowMajor;
        let uplo = CBlasUplo::CblasUpper;
        let trans = CBlasTranspose::CBlasNoTrans;
        let diag = CBlasDiag::CblasNonUnit;
        let a = [1.0f32, 2.0, 0.0, 3.0];
        let mut x = [1.0f32, 1.0];
        unsafe {
            cblas_strmv(
                layout,
                uplo,
                trans,
                diag,
                2,
                a.as_ptr(),
                2,
                x.as_mut_ptr(),
                1,
            );
        }
        assert!(
            (x[0] - 3.0f32).abs() < 1e-4,
            "strmv[0]: expected 3.0, got {}",
            x[0]
        );
        assert!(
            (x[1] - 3.0f32).abs() < 1e-4,
            "strmv[1]: expected 3.0, got {}",
            x[1]
        );
    }

    #[test]
    fn test_strsv() {
        // Upper triangular A = [[2,1],[0,3]], solve A*x = b where b = [5,6]
        // Back substitution: x[1] = 6/3 = 2, x[0] = (5 - 1*2)/2 = 1.5
        let layout = CBlasLayout::CBlasRowMajor;
        let uplo = CBlasUplo::CblasUpper;
        let trans = CBlasTranspose::CBlasNoTrans;
        let diag = CBlasDiag::CblasNonUnit;
        let a = [2.0f32, 1.0, 0.0, 3.0];
        let mut x = [5.0f32, 6.0];
        unsafe {
            cblas_strsv(
                layout,
                uplo,
                trans,
                diag,
                2,
                a.as_ptr(),
                2,
                x.as_mut_ptr(),
                1,
            );
        }
        assert!(
            (x[0] - 1.5f32).abs() < 1e-4,
            "strsv[0]: expected 1.5, got {}",
            x[0]
        );
        assert!(
            (x[1] - 2.0f32).abs() < 1e-4,
            "strsv[1]: expected 2.0, got {}",
            x[1]
        );
    }

    #[test]
    fn test_ssymv() {
        // Symmetric A = [[2,3],[3,5]], x = [1,2], y = [0,0]
        // y := 1.0 * A * x + 0.0 * y = [2+6, 3+10] = [8, 13]
        let layout = CBlasLayout::CBlasRowMajor;
        let uplo = CBlasUplo::CblasUpper;
        let a = [2.0f32, 3.0, 3.0, 5.0];
        let x = [1.0f32, 2.0];
        let mut y = [0.0f32; 2];
        unsafe {
            cblas_ssymv(
                layout,
                uplo,
                2,
                1.0,
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                0.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert!(
            (y[0] - 8.0f32).abs() < 1e-4,
            "ssymv[0]: expected 8.0, got {}",
            y[0]
        );
        assert!(
            (y[1] - 13.0f32).abs() < 1e-4,
            "ssymv[1]: expected 13.0, got {}",
            y[1]
        );
    }

    #[test]
    fn test_dgemv_transpose() {
        // A = [[1,2],[3,4]], so A^T * [1,2] = [7,10].
        let layout = CBlasLayout::CBlasRowMajor;
        let trans = CBlasTranspose::CBlasTrans;
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let x = [1.0f64, 2.0];
        let mut y = [0.0f64; 2];
        unsafe {
            cblas_dgemv(
                layout,
                trans,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                0.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert!(
            (y[0] - 7.0).abs() < 1e-10,
            "dgemv_trans[0]: expected 7.0, got {}",
            y[0]
        );
        assert!(
            (y[1] - 10.0).abs() < 1e-10,
            "dgemv_trans[1]: expected 10.0, got {}",
            y[1]
        );
    }

    #[test]
    fn test_dsymv() {
        // A = [[2,3],[3,5]], x = [1,2], so A*x = [8,13].
        let layout = CBlasLayout::CBlasRowMajor;
        let uplo = CBlasUplo::CblasUpper;
        let a = [2.0f64, 3.0, 0.0, 5.0];
        let x = [1.0f64, 2.0];
        let mut y = [0.0f64; 2];
        unsafe {
            cblas_dsymv(
                layout,
                uplo,
                2,
                1.0,
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                0.0,
                y.as_mut_ptr(),
                1,
            );
        }
        assert!(
            (y[0] - 8.0).abs() < 1e-10,
            "dsymv[0]: expected 8.0, got {}",
            y[0]
        );
        assert!(
            (y[1] - 13.0).abs() < 1e-10,
            "dsymv[1]: expected 13.0, got {}",
            y[1]
        );
    }

    #[test]
    fn test_dsyr_and_dsyr2() {
        let layout = CBlasLayout::CBlasRowMajor;
        let uplo = CBlasUplo::CblasUpper;
        let x = [1.0f64, 2.0];
        let y = [3.0f64, 4.0];

        let mut rank_one = [0.0f64; 4];
        unsafe {
            cblas_dsyr(
                layout,
                uplo,
                2,
                2.0,
                x.as_ptr(),
                1,
                rank_one.as_mut_ptr(),
                2,
            )
        };
        // Only the selected upper triangle is defined by the BLAS routine.
        assert!((rank_one[0] - 2.0).abs() < 1e-10);
        assert!((rank_one[1] - 4.0).abs() < 1e-10);
        assert!((rank_one[3] - 8.0).abs() < 1e-10);

        let mut rank_two = [0.0f64; 4];
        unsafe {
            cblas_dsyr2(
                layout,
                uplo,
                2,
                1.0,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                rank_two.as_mut_ptr(),
                2,
            );
        }
        assert!((rank_two[0] - 6.0).abs() < 1e-10);
        assert!((rank_two[1] - 10.0).abs() < 1e-10);
        assert!((rank_two[3] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_dtrmv_and_dtrsv() {
        let layout = CBlasLayout::CBlasRowMajor;
        let uplo = CBlasUplo::CblasUpper;
        let trans = CBlasTranspose::CBlasNoTrans;
        let diag = CBlasDiag::CblasNonUnit;

        let a = [1.0f64, 2.0, 0.0, 3.0];
        let mut product = [1.0f64, 1.0];
        unsafe {
            cblas_dtrmv(
                layout,
                uplo,
                trans,
                diag,
                2,
                a.as_ptr(),
                2,
                product.as_mut_ptr(),
                1,
            );
        }
        assert!((product[0] - 3.0).abs() < 1e-10);
        assert!((product[1] - 3.0).abs() < 1e-10);

        let a = [2.0f64, 1.0, 0.0, 3.0];
        let mut solution = [5.0f64, 6.0];
        unsafe {
            cblas_dtrsv(
                layout,
                uplo,
                trans,
                diag,
                2,
                a.as_ptr(),
                2,
                solution.as_mut_ptr(),
                1,
            );
        }
        assert!((solution[0] - 1.5).abs() < 1e-10);
        assert!((solution[1] - 2.0).abs() < 1e-10);
    }
}
