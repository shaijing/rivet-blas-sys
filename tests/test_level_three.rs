//! Integration tests for BLAS Level 3 (matrix-matrix) operations.
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
mod level_three {
    use super::*;

    #[test]
    fn test_sgemm() {
        // C := 1.0 * A * B + 0.0 * C
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]], C = [[0,0],[0,0]]
        // A*B = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        let layout = CBlasLayout::CBlasRowMajor;
        let trans = CBlasTranspose::CBlasNoTrans;
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        unsafe {
            cblas_sgemm(
                layout,
                trans,
                trans,
                2,
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
        let expected = [19.0f32, 22.0, 43.0, 50.0];
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-4,
                "sgemm[{i}]: expected {}, got {}",
                expected[i],
                c[i]
            );
        }
    }

    #[test]
    fn test_dgemm() {
        let layout = CBlasLayout::CBlasRowMajor;
        let trans = CBlasTranspose::CBlasNoTrans;
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [5.0f64, 6.0, 7.0, 8.0];
        let mut c = [0.0f64; 4];
        unsafe {
            cblas_dgemm(
                layout,
                trans,
                trans,
                2,
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
        let expected = [19.0f64, 22.0, 43.0, 50.0];
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-10,
                "dgemm[{i}]: expected {}, got {}",
                expected[i],
                c[i]
            );
        }
    }

    #[test]
    fn test_dgemm_with_beta() {
        // C := 1.0 * A * B + 2.0 * C where C starts as [[1,1],[1,1]]
        // A*B = [[19,22],[43,50]], 2*C_old = [[2,2],[2,2]]
        // result = [[21,24],[45,52]]
        let layout = CBlasLayout::CBlasRowMajor;
        let trans = CBlasTranspose::CBlasNoTrans;
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [5.0f64, 6.0, 7.0, 8.0];
        let mut c = [1.0f64; 4];
        unsafe {
            cblas_dgemm(
                layout,
                trans,
                trans,
                2,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                2.0,
                c.as_mut_ptr(),
                2,
            );
        }
        let expected = [21.0f64, 24.0, 45.0, 52.0];
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-10,
                "dgemm_beta[{i}]: expected {}, got {}",
                expected[i],
                c[i]
            );
        }
    }

    #[test]
    fn test_dgemm_transpose() {
        // C := 1.0 * A^T * B + 0.0 * C
        // A = [[1,3],[2,4]] (row-major), A^T = [[1,2],[3,4]]
        // A^T * B where B = [[5,6],[7,8]]
        // = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        let layout = CBlasLayout::CBlasRowMajor;
        let transa = CBlasTranspose::CBlasTrans;
        let transb = CBlasTranspose::CBlasNoTrans;
        let a = [1.0f64, 3.0, 2.0, 4.0];
        let b = [5.0f64, 6.0, 7.0, 8.0];
        let mut c = [0.0f64; 4];
        unsafe {
            cblas_dgemm(
                layout,
                transa,
                transb,
                2,
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
        let expected = [19.0f64, 22.0, 43.0, 50.0];
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-10,
                "dgemm_trans[{i}]: expected {}, got {}",
                expected[i],
                c[i]
            );
        }
    }

    #[test]
    fn test_ssymm() {
        // C := 1.0 * A * B + 0.0 * C, A symmetric upper
        // A = [[2,3],[0,5]] (upper triangle), B = [[1,0],[0,1]] (identity)
        // A*B = [[2,3],[3,5]] (full symmetric result)
        let layout = CBlasLayout::CBlasRowMajor;
        let side = CBlasSide::CblasLeft;
        let uplo = CBlasUplo::CblasUpper;
        let a = [2.0f32, 3.0, 0.0, 5.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];
        unsafe {
            cblas_ssymm(
                layout,
                side,
                uplo,
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
        assert!(
            (c[0] - 2.0f32).abs() < 1e-4,
            "ssymm[0]: expected 2.0, got {}",
            c[0]
        );
        assert!(
            (c[3] - 5.0f32).abs() < 1e-4,
            "ssymm[3]: expected 5.0, got {}",
            c[3]
        );
    }

    #[test]
    fn test_ssyrk() {
        // C := 1.0 * A * A^T + 0.0 * C, A = [[1,2],[3,4]]
        // A*A^T = [[1+4,3+8],[3+8,9+16]] = [[5,11],[11,25]]
        let layout = CBlasLayout::CBlasRowMajor;
        let uplo = CBlasUplo::CblasUpper;
        let trans = CBlasTranspose::CBlasNoTrans;
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let mut c = [0.0f32; 4];
        unsafe {
            cblas_ssyrk(
                layout,
                uplo,
                trans,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                0.0,
                c.as_mut_ptr(),
                2,
            );
        }
        assert!(
            (c[0] - 5.0f32).abs() < 1e-4,
            "ssyrk[0]: expected 5.0, got {}",
            c[0]
        );
        assert!(
            (c[1] - 11.0f32).abs() < 1e-4,
            "ssyrk[1]: expected 11.0, got {}",
            c[1]
        );
    }

    #[test]
    fn test_strmm() {
        // B := 1.0 * A * B, A = [[2,1],[0,3]] (upper triangular), B = [[1,2],[3,4]]
        // A * B = [[2+3, 4+3], [0+9, 0+12]]... wait let me recalculate
        // A*B = [[2*1+1*3, 2*2+1*4],[0*1+3*3, 0*2+3*4]] = [[5,8],[9,12]]
        let layout = CBlasLayout::CBlasRowMajor;
        let side = CBlasSide::CblasLeft;
        let uplo = CBlasUplo::CblasUpper;
        let trans = CBlasTranspose::CBlasNoTrans;
        let diag = CBlasDiag::CblasNonUnit;
        let a = [2.0f32, 1.0, 0.0, 3.0];
        let mut b = [1.0f32, 2.0, 3.0, 4.0];
        unsafe {
            cblas_strmm(
                layout,
                side,
                uplo,
                trans,
                diag,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                b.as_mut_ptr(),
                2,
            );
        }
        assert!(
            (b[0] - 5.0f32).abs() < 1e-4,
            "strmm[0]: expected 5.0, got {}",
            b[0]
        );
        assert!(
            (b[1] - 8.0f32).abs() < 1e-4,
            "strmm[1]: expected 8.0, got {}",
            b[1]
        );
        assert!(
            (b[2] - 9.0f32).abs() < 1e-4,
            "strmm[2]: expected 9.0, got {}",
            b[2]
        );
        assert!(
            (b[3] - 12.0f32).abs() < 1e-4,
            "strmm[3]: expected 12.0, got {}",
            b[3]
        );
    }

    #[test]
    fn test_dsymm() {
        // A = [[2,3],[3,5]] and B is the identity, so A*B = A.
        let layout = CBlasLayout::CBlasRowMajor;
        let side = CBlasSide::CblasLeft;
        let uplo = CBlasUplo::CblasUpper;
        let a = [2.0f64, 3.0, 0.0, 5.0];
        let b = [1.0f64, 0.0, 0.0, 1.0];
        let mut c = [0.0f64; 4];
        unsafe {
            cblas_dsymm(
                layout,
                side,
                uplo,
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
        let expected = [2.0, 3.0, 3.0, 5.0];
        for (index, (&actual, &expected)) in c.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-10,
                "dsymm[{index}]: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_dsyrk() {
        // A = [[1,2],[3,4]], so A*A^T = [[5,11],[11,25]].
        let layout = CBlasLayout::CBlasRowMajor;
        let uplo = CBlasUplo::CblasUpper;
        let trans = CBlasTranspose::CBlasNoTrans;
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let mut c = [0.0f64; 4];
        unsafe {
            cblas_dsyrk(
                layout,
                uplo,
                trans,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                0.0,
                c.as_mut_ptr(),
                2,
            );
        }
        assert!(
            (c[0] - 5.0).abs() < 1e-10,
            "dsyrk[0]: expected 5.0, got {}",
            c[0]
        );
        assert!(
            (c[1] - 11.0).abs() < 1e-10,
            "dsyrk[1]: expected 11.0, got {}",
            c[1]
        );
        assert!(
            (c[3] - 25.0).abs() < 1e-10,
            "dsyrk[3]: expected 25.0, got {}",
            c[3]
        );
    }

    #[test]
    fn test_dtrmm_and_dtrsm() {
        let layout = CBlasLayout::CBlasRowMajor;
        let side = CBlasSide::CblasLeft;
        let uplo = CBlasUplo::CblasUpper;
        let trans = CBlasTranspose::CBlasNoTrans;
        let diag = CBlasDiag::CblasNonUnit;
        let a = [2.0f64, 1.0, 0.0, 3.0];

        let mut product = [1.0f64, 2.0, 3.0, 4.0];
        unsafe {
            cblas_dtrmm(
                layout,
                side,
                uplo,
                trans,
                diag,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                product.as_mut_ptr(),
                2,
            );
        }
        assert_eq!(product, [5.0, 8.0, 9.0, 12.0]);

        // Solve A*X = B for B = A*[[1,2],[3,4]].
        let mut solution = [5.0f64, 8.0, 9.0, 12.0];
        unsafe {
            cblas_dtrsm(
                layout,
                side,
                uplo,
                trans,
                diag,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                solution.as_mut_ptr(),
                2,
            );
        }
        assert!((solution[0] - 1.0).abs() < 1e-10);
        assert!((solution[1] - 2.0).abs() < 1e-10);
        assert!((solution[2] - 3.0).abs() < 1e-10);
        assert!((solution[3] - 4.0).abs() < 1e-10);
    }
}
