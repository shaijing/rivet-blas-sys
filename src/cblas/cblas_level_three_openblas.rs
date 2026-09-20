//! OpenBLAS/FlexiBLAS matrix extension entry points.

use crate::cblas::cblas_types::*;

unsafe extern "C" {
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
