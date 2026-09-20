//! Intel MKL-only Level 1 CBLAS entry points.
//!
//! These unsuffixed symbols use `CBlasInt`, which selects MKL's ordinary
//! `MKL_INT` ABI through the `lp64` or `ilp64` feature. The explicit MKL
//! `*_64` symbols are declared in [`super::cblas_level_one_mkl_64`].

use crate::cblas::cblas_types::*;

unsafe extern "C" {
    pub fn cblas_crot(
        n: CBlasInt,
        x: *mut CBlasVoid,
        incx: CBlasInt,
        y: *mut CBlasVoid,
        incy: CBlasInt,
        c: CBlasFloat,
        s: *const CBlasVoid,
    );

    pub fn cblas_zrot(
        n: CBlasInt,
        x: *mut CBlasVoid,
        incx: CBlasInt,
        y: *mut CBlasVoid,
        incy: CBlasInt,
        c: CBlasDouble,
        s: *const CBlasVoid,
    );

    pub fn cblas_isamin(n: CBlasInt, x: *const CBlasFloat, incx: CBlasInt) -> CBlasIndex;
    pub fn cblas_idamin(n: CBlasInt, x: *const CBlasDouble, incx: CBlasInt) -> CBlasIndex;
    pub fn cblas_icamin(n: CBlasInt, x: *const CBlasVoid, incx: CBlasInt) -> CBlasIndex;
    pub fn cblas_izamin(n: CBlasInt, x: *const CBlasVoid, incx: CBlasInt) -> CBlasIndex;
}
