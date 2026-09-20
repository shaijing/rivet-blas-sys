//! Intel MKL-only Level 1 CBLAS entry points.
//!
//! These unsuffixed symbols use `CBlasInt`, which selects MKL's ordinary
//! `MKL_INT` ABI through the `lp64` or `ilp64` feature. The explicit MKL
//! `*_64` symbols are declared in [`super::cblas_level_one_mkl_64`].
//!
//! # Example
//!
//! ```no_run
//! use rivet_blas_sys::cblas::cblas_level_one_mkl::cblas_isamin;
//! use rivet_blas_sys::cblas::cblas_types::CBlasInt;
//!
//! let values = [3.0_f32, -1.0, 2.0];
//! let index = unsafe { cblas_isamin(3 as CBlasInt, values.as_ptr(), 1) };
//! assert_eq!(index, 1);
//! ```

use crate::cblas::cblas_types::*;

unsafe extern "C" {
    /// Applies a complex single-precision plane rotation to two vectors.
    ///
    /// The operation is performed in place. `c` is the real cosine scalar;
    /// `s` points to the complex sine scalar used by MKL.
    ///
    /// # Safety
    /// `x`, `y`, and `s` must point to valid, sufficiently sized storage, and
    /// both increments must be non-zero.
    pub fn cblas_crot(
        n: CBlasInt,
        x: *mut CBlasVoid,
        incx: CBlasInt,
        y: *mut CBlasVoid,
        incy: CBlasInt,
        c: CBlasFloat,
        s: *const CBlasVoid,
    );

    /// Applies a complex double-precision plane rotation to two vectors.
    ///
    /// The operation is performed in place. `c` is the real cosine scalar;
    /// `s` points to the complex sine scalar used by MKL.
    ///
    /// # Safety
    /// `x`, `y`, and `s` must point to valid, sufficiently sized storage, and
    /// both increments must be non-zero.
    pub fn cblas_zrot(
        n: CBlasInt,
        x: *mut CBlasVoid,
        incx: CBlasInt,
        y: *mut CBlasVoid,
        incy: CBlasInt,
        c: CBlasDouble,
        s: *const CBlasVoid,
    );

    /// Returns the zero-based index of the element with the smallest
    /// absolute value in a single-precision real vector.
    ///
    /// # Safety
    /// `x` must point to at least `n` logically strided elements and `incx`
    /// must be non-zero.
    pub fn cblas_isamin(n: CBlasInt, x: *const CBlasFloat, incx: CBlasInt) -> CBlasIndex;

    /// Returns the zero-based index of the element with the smallest
    /// absolute value in a double-precision real vector.
    ///
    /// # Safety
    /// `x` must point to at least `n` logically strided elements and `incx`
    /// must be non-zero.
    pub fn cblas_idamin(n: CBlasInt, x: *const CBlasDouble, incx: CBlasInt) -> CBlasIndex;

    /// Returns the zero-based index of the element with the smallest
    /// magnitude in a complex single-precision vector.
    ///
    /// # Safety
    /// `x` must point to valid complex storage for `n` logically strided
    /// elements and `incx` must be non-zero.
    pub fn cblas_icamin(n: CBlasInt, x: *const CBlasVoid, incx: CBlasInt) -> CBlasIndex;

    /// Returns the zero-based index of the element with the smallest
    /// magnitude in a complex double-precision vector.
    ///
    /// # Safety
    /// `x` must point to valid complex storage for `n` logically strided
    /// elements and `incx` must be non-zero.
    pub fn cblas_izamin(n: CBlasInt, x: *const CBlasVoid, incx: CBlasInt) -> CBlasIndex;
}
