//! OpenBLAS/FlexiBLAS Level 1 extensions.

use crate::cblas::cblas_types::*;

unsafe extern "C" {
    pub fn cblas_isamin(n: CBlasInt, x: *const CBlasFloat, incx: CBlasInt) -> CBlasIndex;
    pub fn cblas_idamin(n: CBlasInt, x: *const CBlasDouble, incx: CBlasInt) -> CBlasIndex;
    pub fn cblas_icamin(n: CBlasInt, x: *const CBlasVoid, incx: CBlasInt) -> CBlasIndex;
    pub fn cblas_izamin(n: CBlasInt, x: *const CBlasVoid, incx: CBlasInt) -> CBlasIndex;

    pub fn cblas_saxpby(
        n: CBlasInt,
        alpha: CBlasFloat,
        x: *const CBlasFloat,
        incx: CBlasInt,
        beta: CBlasFloat,
        y: *mut CBlasFloat,
        incy: CBlasInt,
    );
    pub fn cblas_daxpby(
        n: CBlasInt,
        alpha: CBlasDouble,
        x: *const CBlasDouble,
        incx: CBlasInt,
        beta: CBlasDouble,
        y: *mut CBlasDouble,
        incy: CBlasInt,
    );
    pub fn cblas_caxpby(
        n: CBlasInt,
        alpha: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: CBlasInt,
        beta: *const CBlasVoid,
        y: *mut CBlasVoid,
        incy: CBlasInt,
    );
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
