//! OpenBLAS/FlexiBLAS Level 1 extensions.
//!
//! These declarations are available with the `openblas` or `flexiblas`
//! feature. They use the selected ordinary `lp64`/`ilp64` ABI.
//!
//! # Example
//!
//! ```no_run
//! use rivet_blas_sys::cblas::cblas_level_one_openblas::cblas_saxpby;
//! use rivet_blas_sys::cblas::cblas_types::CBlasInt;
//!
//! let x = [1.0_f32, 2.0, 3.0];
//! let mut y = [10.0_f32, 20.0, 30.0];
//! unsafe {
//!     cblas_saxpby(3 as CBlasInt, 2.0, x.as_ptr(), 1, 0.5, y.as_mut_ptr(), 1);
//! }
//! assert_eq!(y, [7.0, 14.0, 21.0]);
//! ```

use crate::cblas::cblas_types::*;

unsafe extern "C" {
    /// Returns the zero-based index of the smallest-magnitude element in a
    /// single-precision real vector.
    ///
    /// # Safety
    /// `x` must point to at least `n` logically strided elements and `incx`
    /// must be non-zero.
    pub fn cblas_isamin(n: CBlasInt, x: *const CBlasFloat, incx: CBlasInt) -> CBlasIndex;

    /// Returns the zero-based index of the smallest-magnitude element in a
    /// double-precision real vector.
    ///
    /// # Safety
    /// `x` must point to at least `n` logically strided elements and `incx`
    /// must be non-zero.
    pub fn cblas_idamin(n: CBlasInt, x: *const CBlasDouble, incx: CBlasInt) -> CBlasIndex;

    /// Returns the zero-based index of the smallest-magnitude element in a
    /// complex single-precision vector.
    ///
    /// # Safety
    /// `x` must point to valid complex storage for `n` logically strided
    /// elements and `incx` must be non-zero.
    pub fn cblas_icamin(n: CBlasInt, x: *const CBlasVoid, incx: CBlasInt) -> CBlasIndex;

    /// Returns the zero-based index of the smallest-magnitude element in a
    /// complex double-precision vector.
    ///
    /// # Safety
    /// `x` must point to valid complex storage for `n` logically strided
    /// elements and `incx` must be non-zero.
    pub fn cblas_izamin(n: CBlasInt, x: *const CBlasVoid, incx: CBlasInt) -> CBlasIndex;

    /// Computes `y := alpha * x + beta * y` for single-precision vectors.
    ///
    /// # Safety
    /// `x` and `y` must point to valid, sufficiently sized strided vectors;
    /// both increments must be non-zero.
    pub fn cblas_saxpby(
        n: CBlasInt,
        alpha: CBlasFloat,
        x: *const CBlasFloat,
        incx: CBlasInt,
        beta: CBlasFloat,
        y: *mut CBlasFloat,
        incy: CBlasInt,
    );

    /// Computes `y := alpha * x + beta * y` for double-precision vectors.
    ///
    /// # Safety
    /// `x` and `y` must point to valid, sufficiently sized strided vectors;
    /// both increments must be non-zero.
    pub fn cblas_daxpby(
        n: CBlasInt,
        alpha: CBlasDouble,
        x: *const CBlasDouble,
        incx: CBlasInt,
        beta: CBlasDouble,
        y: *mut CBlasDouble,
        incy: CBlasInt,
    );

    /// Computes `y := alpha * x + beta * y` for complex single-precision
    /// vectors.
    ///
    /// Complex values use the backend's CBLAS representation through
    /// `CBlasVoid` pointers.
    ///
    /// # Safety
    /// `alpha`, `x`, `beta`, and `y` must point to valid complex storage;
    /// both increments must be non-zero.
    pub fn cblas_caxpby(
        n: CBlasInt,
        alpha: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: CBlasInt,
        beta: *const CBlasVoid,
        y: *mut CBlasVoid,
        incy: CBlasInt,
    );

    /// Computes `y := alpha * x + beta * y` for complex double-precision
    /// vectors.
    ///
    /// Complex values use the backend's CBLAS representation through
    /// `CBlasVoid` pointers.
    ///
    /// # Safety
    /// `alpha`, `x`, `beta`, and `y` must point to valid complex storage;
    /// both increments must be non-zero.
    pub fn cblas_zaxpby(
        n: CBlasInt,
        alpha: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: CBlasInt,
        beta: *const CBlasVoid,
        y: *mut CBlasVoid,
        incy: CBlasInt,
    );
}
