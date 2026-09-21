//! OpenBLAS/FlexiBLAS matrix extension entry points.
//!
//! These declarations are available with an `openblas-*` or `flexiblas-*`
//! configuration and use that configuration's ordinary `lp64`/`ilp64` ABI.
//!
//! # Example
//!
//! ```no_run
//! use rivet_blas_sys::cblas::cblas_level_three_openblas::cblas_somatcopy;
//! use rivet_blas_sys::cblas::cblas_types::{CBlasInt, CBlasLayout, CBlasTranspose};
//!
//! let a = [1.0_f32, 2.0, 3.0, 4.0];
//! let mut b = [0.0_f32; 4];
//! unsafe {
//!     cblas_somatcopy(
//!         CBlasLayout::CBlasRowMajor,
//!         CBlasTranspose::CBlasNoTrans,
//!         2 as CBlasInt,
//!         2 as CBlasInt,
//!         1.0,
//!         a.as_ptr(),
//!         2 as CBlasInt,
//!         b.as_mut_ptr(),
//!         2 as CBlasInt,
//!     );
//! }
//! assert_eq!(b, a);
//! ```

use crate::cblas::cblas_types::*;

unsafe extern "C" {
    /// Copies a single-precision matrix to `b`, applying `trans` and scaling
    /// the result by `alpha`: `B := alpha * op(A)`.
    ///
    /// # Safety
    /// `a` and `b` must reference valid, non-overlapping matrix storage with
    /// leading dimensions compatible with `layout`, `rows`, and `cols`.
    pub fn cblas_somatcopy(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        rows: CBlasInt,
        cols: CBlasInt,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: CBlasInt,
        b: *mut CBlasFloat,
        ldb: CBlasInt,
    );

    /// Copies a double-precision matrix to `b`, applying `trans` and scaling
    /// the result by `alpha`: `B := alpha * op(A)`.
    ///
    /// # Safety
    /// `a` and `b` must reference valid, non-overlapping matrix storage with
    /// compatible leading dimensions.
    pub fn cblas_domatcopy(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        rows: CBlasInt,
        cols: CBlasInt,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: CBlasInt,
        b: *mut CBlasDouble,
        ldb: CBlasInt,
    );

    /// Copies a complex single-precision matrix to `b`, applying `trans` and
    /// scaling the result by `alpha`: `B := alpha * op(A)`.
    ///
    /// # Safety
    /// Complex scalar and matrix pointers must be valid, and `a` and `b` must
    /// reference non-overlapping storage with compatible leading dimensions.
    pub fn cblas_comatcopy(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        rows: CBlasInt,
        cols: CBlasInt,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: CBlasInt,
        b: *mut CBlasVoid,
        ldb: CBlasInt,
    );

    /// Copies a complex double-precision matrix to `b`, applying `trans` and
    /// scaling the result by `alpha`: `B := alpha * op(A)`.
    ///
    /// # Safety
    /// Complex scalar and matrix pointers must be valid, and `a` and `b` must
    /// reference non-overlapping storage with compatible leading dimensions.
    pub fn cblas_zomatcopy(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        rows: CBlasInt,
        cols: CBlasInt,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: CBlasInt,
        b: *mut CBlasVoid,
        ldb: CBlasInt,
    );

    /// Scales and optionally transposes a single-precision matrix in place:
    /// `A := alpha * op(A)`.
    ///
    /// # Safety
    /// `a` must reference writable matrix storage large enough for the
    /// operation, and its leading dimensions must be valid.
    pub fn cblas_simatcopy(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        rows: CBlasInt,
        cols: CBlasInt,
        alpha: CBlasFloat,
        a: *mut CBlasFloat,
        lda: CBlasInt,
        ldb: CBlasInt,
    );

    /// Scales and optionally transposes a double-precision matrix in place:
    /// `A := alpha * op(A)`.
    ///
    /// # Safety
    /// `a` must reference writable matrix storage large enough for the
    /// operation, and its leading dimensions must be valid.
    pub fn cblas_dimatcopy(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        rows: CBlasInt,
        cols: CBlasInt,
        alpha: CBlasDouble,
        a: *mut CBlasDouble,
        lda: CBlasInt,
        ldb: CBlasInt,
    );

    /// Scales and optionally transposes a complex single-precision matrix in
    /// place: `A := alpha * op(A)`.
    ///
    /// # Safety
    /// `alpha` and `a` must reference valid complex storage large enough for
    /// the operation, and the leading dimensions must be valid.
    pub fn cblas_cimatcopy(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        rows: CBlasInt,
        cols: CBlasInt,
        alpha: *const CBlasVoid,
        a: *mut CBlasVoid,
        lda: CBlasInt,
        ldb: CBlasInt,
    );

    /// Scales and optionally transposes a complex double-precision matrix in
    /// place: `A := alpha * op(A)`.
    ///
    /// # Safety
    /// `alpha` and `a` must reference valid complex storage large enough for
    /// the operation, and the leading dimensions must be valid.
    pub fn cblas_zimatcopy(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        rows: CBlasInt,
        cols: CBlasInt,
        alpha: *const CBlasVoid,
        a: *mut CBlasVoid,
        lda: CBlasInt,
        ldb: CBlasInt,
    );

    /// Computes `C := alpha * A + beta * C` for single-precision matrices.
    ///
    /// # Safety
    /// `a` and `c` must reference valid matrix storage with compatible leading
    /// dimensions; `c` must be writable.
    pub fn cblas_sgeadd(
        layout: CBlasLayout,
        rows: CBlasInt,
        cols: CBlasInt,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: CBlasInt,
        beta: CBlasFloat,
        c: *mut CBlasFloat,
        ldc: CBlasInt,
    );

    /// Computes `C := alpha * A + beta * C` for double-precision matrices.
    ///
    /// # Safety
    /// `a` and `c` must reference valid matrix storage with compatible leading
    /// dimensions; `c` must be writable.
    pub fn cblas_dgeadd(
        layout: CBlasLayout,
        rows: CBlasInt,
        cols: CBlasInt,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: CBlasInt,
        beta: CBlasDouble,
        c: *mut CBlasDouble,
        ldc: CBlasInt,
    );

    /// Computes `C := alpha * A + beta * C` for complex single-precision
    /// matrices.
    ///
    /// # Safety
    /// Complex scalar and matrix pointers must be valid, and `c` must be
    /// writable with a compatible leading dimension.
    pub fn cblas_cgeadd(
        layout: CBlasLayout,
        rows: CBlasInt,
        cols: CBlasInt,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: CBlasInt,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: CBlasInt,
    );

    /// Computes `C := alpha * A + beta * C` for complex double-precision
    /// matrices.
    ///
    /// # Safety
    /// Complex scalar and matrix pointers must be valid, and `c` must be
    /// writable with a compatible leading dimension.
    pub fn cblas_zgeadd(
        layout: CBlasLayout,
        rows: CBlasInt,
        cols: CBlasInt,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: CBlasInt,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: CBlasInt,
    );
}
