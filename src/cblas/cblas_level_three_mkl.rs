//! Intel MKL-only Level 3 CBLAS entry points.
//!
//! Unsuffixed symbols use the selected ordinary LP64 or ILP64 `CBlasInt` ABI.
//! Their explicit MKL `*_64` counterparts are in
//! [`super::cblas_level_three_mkl_64`].
//!
//! The `_oop` routines are MKL's out-of-place triangular operations: `a` and
//! `b` remain unchanged and the result is written to `c`.
//!
//! # Example
//!
//! ```no_run
//! use rivet_blas_sys::cblas::cblas_level_three_mkl::cblas_dtrmm_oop;
//! use rivet_blas_sys::cblas::cblas_types::{
//!     CBlasDiag, CBlasInt, CBlasLayout, CBlasSide, CBlasTranspose, CBlasUplo,
//! };
//!
//! let a = [1.0_f64, 0.0, 2.0, 3.0];
//! let b = [4.0_f64, 5.0, 6.0, 7.0];
//! let mut c = [0.0_f64; 4];
//! unsafe {
//!     cblas_dtrmm_oop(
//!         CBlasLayout::CBlasRowMajor,
//!         CBlasSide::CblasLeft,
//!         CBlasUplo::CblasUpper,
//!         CBlasTranspose::CBlasNoTrans,
//!         CBlasDiag::CblasNonUnit,
//!         2 as CBlasInt,
//!         2 as CBlasInt,
//!         1.0,
//!         a.as_ptr(),
//!         2 as CBlasInt,
//!         b.as_ptr(),
//!         2 as CBlasInt,
//!         0.0,
//!         c.as_mut_ptr(),
//!         2 as CBlasInt,
//!     );
//! }
//! ```

use crate::cblas::cblas_types::*;

unsafe extern "C" {
    /// Computes a half-precision general matrix product:
    /// `C := alpha * op(A) * op(B) + beta * C`.
    ///
    /// This MKL extension uses `CBlasF16` values for the matrices and scalar
    /// coefficients.
    ///
    /// # Safety
    /// All matrix pointers must reference valid storage consistent with the
    /// dimensions, transpose flags, and leading dimensions.
    pub fn cblas_hgemm(
        layout: CBlasLayout,
        transa: CBlasTranspose,
        transb: CBlasTranspose,
        m: CBlasInt,
        n: CBlasInt,
        k: CBlasInt,
        alpha: CBlasF16,
        a: *const CBlasF16,
        lda: CBlasInt,
        b: *const CBlasF16,
        ldb: CBlasInt,
        beta: CBlasF16,
        c: *mut CBlasF16,
        ldc: CBlasInt,
    );

    /// Computes an out-of-place single-precision triangular matrix product.
    /// The result is written as `C := alpha * op(A) * B + beta * C` or its
    /// right-sided equivalent according to `side`.
    ///
    /// # Safety
    /// `a`, `b`, and `c` must reference valid matrix storage. The leading
    /// dimensions and triangular flags must describe those matrices.
    pub fn cblas_strmm_oop(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: CBlasInt,
        n: CBlasInt,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: CBlasInt,
        b: *const CBlasFloat,
        ldb: CBlasInt,
        beta: CBlasFloat,
        c: *mut CBlasFloat,
        ldc: CBlasInt,
    );

    /// Computes an out-of-place double-precision triangular matrix product.
    /// The input matrices are preserved and the result is written to `c`.
    ///
    /// # Safety
    /// `a`, `b`, and `c` must reference valid matrix storage. The leading
    /// dimensions and triangular flags must describe those matrices.
    pub fn cblas_dtrmm_oop(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: CBlasInt,
        n: CBlasInt,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: CBlasInt,
        b: *const CBlasDouble,
        ldb: CBlasInt,
        beta: CBlasDouble,
        c: *mut CBlasDouble,
        ldc: CBlasInt,
    );

    /// Computes an out-of-place complex single-precision triangular matrix
    /// product. The input matrices are preserved and the result is written to
    /// `c`.
    ///
    /// # Safety
    /// `alpha`, `a`, `b`, `beta`, and `c` must reference valid complex storage
    /// consistent with the dimensions and leading dimensions.
    pub fn cblas_ctrmm_oop(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: CBlasInt,
        n: CBlasInt,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: CBlasInt,
        b: *const CBlasVoid,
        ldb: CBlasInt,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: CBlasInt,
    );

    /// Computes an out-of-place complex double-precision triangular matrix
    /// product. The input matrices are preserved and the result is written to
    /// `c`.
    ///
    /// # Safety
    /// `alpha`, `a`, `b`, `beta`, and `c` must reference valid complex storage
    /// consistent with the dimensions and leading dimensions.
    pub fn cblas_ztrmm_oop(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: CBlasInt,
        n: CBlasInt,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: CBlasInt,
        b: *const CBlasVoid,
        ldb: CBlasInt,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: CBlasInt,
    );

    /// Computes an out-of-place single-precision triangular solve and writes
    /// the result to `c`; `a` and `b` are preserved.
    ///
    /// # Safety
    /// `a`, `b`, and `c` must reference valid matrix storage. `a` must contain
    /// a non-singular triangular matrix described by the flags.
    pub fn cblas_strsm_oop(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: CBlasInt,
        n: CBlasInt,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: CBlasInt,
        b: *const CBlasFloat,
        ldb: CBlasInt,
        beta: CBlasFloat,
        c: *mut CBlasFloat,
        ldc: CBlasInt,
    );

    /// Computes an out-of-place double-precision triangular solve and writes
    /// the result to `c`; `a` and `b` are preserved.
    ///
    /// # Safety
    /// `a`, `b`, and `c` must reference valid matrix storage. `a` must contain
    /// a non-singular triangular matrix described by the flags.
    pub fn cblas_dtrsm_oop(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: CBlasInt,
        n: CBlasInt,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: CBlasInt,
        b: *const CBlasDouble,
        ldb: CBlasInt,
        beta: CBlasDouble,
        c: *mut CBlasDouble,
        ldc: CBlasInt,
    );

    /// Computes an out-of-place complex single-precision triangular solve and
    /// writes the result to `c`; `a` and `b` are preserved.
    ///
    /// # Safety
    /// Complex scalar and matrix pointers must be valid and `a` must contain a
    /// non-singular triangular matrix described by the flags.
    pub fn cblas_ctrsm_oop(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: CBlasInt,
        n: CBlasInt,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: CBlasInt,
        b: *const CBlasVoid,
        ldb: CBlasInt,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: CBlasInt,
    );

    /// Computes an out-of-place complex double-precision triangular solve and
    /// writes the result to `c`; `a` and `b` are preserved.
    ///
    /// # Safety
    /// Complex scalar and matrix pointers must be valid and `a` must contain a
    /// non-singular triangular matrix described by the flags.
    pub fn cblas_ztrsm_oop(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: CBlasInt,
        n: CBlasInt,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: CBlasInt,
        b: *const CBlasVoid,
        ldb: CBlasInt,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: CBlasInt,
    );
}
