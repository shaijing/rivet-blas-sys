//! Integration tests for BLAS Level 1 (vector-vector) operations.
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
mod level_one {
    use super::*;

    #[test]
    fn test_sdot() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let y = [5.0f32, 6.0, 7.0, 8.0];
        let result = unsafe { cblas_sdot(4, x.as_ptr(), 1, y.as_ptr(), 1) };
        let expected: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (result - expected).abs() < 1e-5,
            "sdot: expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_ddot() {
        let x = [1.0f64, 2.0, 3.0, 4.0];
        let y = [5.0f64, 6.0, 7.0, 8.0];
        let result = unsafe { cblas_ddot(4, x.as_ptr(), 1, y.as_ptr(), 1) };
        let expected: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (result - expected).abs() < 1e-10,
            "ddot: expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_sdot_with_stride() {
        let x = [1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0];
        let y = [5.0f32, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0];
        let result = unsafe { cblas_sdot(4, x.as_ptr(), 2, y.as_ptr(), 2) };
        let expected: f32 = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;
        assert!(
            (result - expected).abs() < 1e-5,
            "sdot stride: expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_sasum() {
        let x = [3.0f32, -4.0, 5.0];
        let result = unsafe { cblas_sasum(3, x.as_ptr(), 1) };
        assert!(
            (result - 12.0f32).abs() < 1e-5,
            "sasum: expected 12.0, got {result}"
        );
    }

    #[test]
    fn test_dasum() {
        let x = [3.0f64, -4.0, 5.0];
        let result = unsafe { cblas_dasum(3, x.as_ptr(), 1) };
        assert!(
            (result - 12.0f64).abs() < 1e-10,
            "dasum: expected 12.0, got {result}"
        );
    }

    #[test]
    fn test_snrm2() {
        let x = [3.0f32, 4.0];
        let result = unsafe { cblas_snrm2(2, x.as_ptr(), 1) };
        let expected = 5.0f32; // sqrt(9 + 16) = 5
        assert!(
            (result - expected).abs() < 1e-5,
            "snrm2: expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_dnrm2() {
        let x = [3.0f64, 4.0];
        let result = unsafe { cblas_dnrm2(2, x.as_ptr(), 1) };
        let expected = 5.0f64;
        assert!(
            (result - expected).abs() < 1e-10,
            "dnrm2: expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_saxpy() {
        let x = [1.0f32, 2.0, 3.0];
        let mut y = [4.0f32, 5.0, 6.0];
        unsafe {
            cblas_saxpy(3, 2.0, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        }
        // y := 2.0 * x + y = [2+4, 4+5, 6+6] = [6, 9, 12]
        assert!(
            (y[0] - 6.0f32).abs() < 1e-5,
            "saxpy[0]: expected 6.0, got {}",
            y[0]
        );
        assert!(
            (y[1] - 9.0f32).abs() < 1e-5,
            "saxpy[1]: expected 9.0, got {}",
            y[1]
        );
        assert!(
            (y[2] - 12.0f32).abs() < 1e-5,
            "saxpy[2]: expected 12.0, got {}",
            y[2]
        );
    }

    #[test]
    fn test_daxpy() {
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [4.0f64, 5.0, 6.0];
        unsafe {
            cblas_daxpy(3, 2.0, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        }
        assert!(
            (y[0] - 6.0f64).abs() < 1e-10,
            "daxpy[0]: expected 6.0, got {}",
            y[0]
        );
        assert!(
            (y[1] - 9.0f64).abs() < 1e-10,
            "daxpy[1]: expected 9.0, got {}",
            y[1]
        );
        assert!(
            (y[2] - 12.0f64).abs() < 1e-10,
            "daxpy[2]: expected 12.0, got {}",
            y[2]
        );
    }

    #[test]
    fn test_sscal() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        unsafe {
            cblas_sscal(4, 3.0, x.as_mut_ptr(), 1);
        }
        // x := 3.0 * x = [3, 6, 9, 12]
        assert!(
            (x[0] - 3.0f32).abs() < 1e-5,
            "sscal[0]: expected 3.0, got {}",
            x[0]
        );
        assert!(
            (x[1] - 6.0f32).abs() < 1e-5,
            "sscal[1]: expected 6.0, got {}",
            x[1]
        );
        assert!(
            (x[2] - 9.0f32).abs() < 1e-5,
            "sscal[2]: expected 9.0, got {}",
            x[2]
        );
        assert!(
            (x[3] - 12.0f32).abs() < 1e-5,
            "sscal[3]: expected 12.0, got {}",
            x[3]
        );
    }

    #[test]
    fn test_dscal() {
        let mut x = [1.0f64, 2.0, 3.0, 4.0];
        unsafe {
            cblas_dscal(4, 3.0, x.as_mut_ptr(), 1);
        }
        assert!(
            (x[0] - 3.0f64).abs() < 1e-10,
            "dscal[0]: expected 3.0, got {}",
            x[0]
        );
        assert!(
            (x[3] - 12.0f64).abs() < 1e-10,
            "dscal[3]: expected 12.0, got {}",
            x[3]
        );
    }

    #[test]
    fn test_scopy() {
        let x = [1.0f32, 2.0, 3.0];
        let mut y = [0.0f32; 3];
        unsafe {
            cblas_scopy(3, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        }
        assert_eq!(y, [1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn test_dcopy() {
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [0.0f64; 3];
        unsafe {
            cblas_dcopy(3, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        }
        assert_eq!(y, [1.0f64, 2.0, 3.0]);
    }

    #[test]
    fn test_sswap() {
        let mut x = [1.0f32, 2.0, 3.0];
        let mut y = [4.0f32, 5.0, 6.0];
        unsafe {
            cblas_sswap(3, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1);
        }
        assert_eq!(x, [4.0f32, 5.0, 6.0]);
        assert_eq!(y, [1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn test_isamax() {
        let x = [1.0f32, -5.0, 3.0, -2.0];
        let idx = unsafe { cblas_isamax(4, x.as_ptr(), 1) };
        assert_eq!(idx, 1, "isamax: expected index 1 (|-5| is max), got {idx}");
    }

    #[test]
    fn test_idamax() {
        let x = [1.0f64, -5.0, 3.0, -2.0];
        let idx = unsafe { cblas_idamax(4, x.as_ptr(), 1) };
        assert_eq!(idx, 1, "idamax: expected index 1, got {idx}");
    }

    #[test]
    fn test_sdsdot_and_dsdot() {
        let x = [1.0f32, 2.0, 3.0];
        let y = [4.0f32, 5.0, 6.0];

        let sdsdot = unsafe { cblas_sdsdot(3, 1.5, x.as_ptr(), 1, y.as_ptr(), 1) };
        assert!(
            (sdsdot - 33.5).abs() < 1e-5,
            "sdsdot: expected 33.5, got {sdsdot}"
        );

        let dsdot = unsafe { cblas_dsdot(3, x.as_ptr(), 1, y.as_ptr(), 1) };
        assert!(
            (dsdot - 32.0).abs() < 1e-10,
            "dsdot: expected 32.0, got {dsdot}"
        );
    }

    #[test]
    fn test_srot_and_drot() {
        let mut sx = [1.0f32, 2.0];
        let mut sy = [3.0f32, 4.0];
        unsafe { cblas_srot(2, sx.as_mut_ptr(), 1, sy.as_mut_ptr(), 1, 0.0, 1.0) };
        assert_eq!(sx, [3.0, 4.0]);
        assert_eq!(sy, [-1.0, -2.0]);

        let mut dx = [1.0f64, 2.0];
        let mut dy = [3.0f64, 4.0];
        unsafe { cblas_drot(2, dx.as_mut_ptr(), 1, dy.as_mut_ptr(), 1, 0.0, 1.0) };
        assert_eq!(dx, [3.0, 4.0]);
        assert_eq!(dy, [-1.0, -2.0]);
    }
}
