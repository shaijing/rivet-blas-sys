//! Explicit Intel MKL `*_64` Level 2 CBLAS entry points.
//!
//! Every declaration in this module binds an actual suffixed MKL symbol and
//! uses `MKL_INT64`, independent of the ordinary ABI feature.

use crate::cblas::cblas_types::*;

unsafe extern "C" {
    /// The ?gbmv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^T \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^H \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are scalars, $x$ and $y$ are vectors,
    /// and $A$ is an $m \\times n$ band matrix with $k_l$ sub-diagonals and $k_u$ super-diagonals.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `kl` - Specifies the number of sub-diagonals of the matrix $A$.
    /// * `ku` - Specifies the number of super-diagonals of the matrix $A$.
    /// * `alpha` - Specifies the scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k_l + k_u + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$ when trans = CblasNoTrans, otherwise $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (m-1) \cdot |\\text{incy}|)$ when trans = CblasNoTrans, otherwise $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_sgbmv_64(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        kl: MklCBlasInt64,
        ku: MklCBlasInt64,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
        beta: CBlasFloat,
        y: *mut CBlasFloat,
        incy: MklCBlasInt64,
    );

    /// The ?gbmv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^T \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^H \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are scalars, $x$ and $y$ are vectors,
    /// and $A$ is an $m \\times n$ band matrix with $k_l$ sub-diagonals and $k_u$ super-diagonals.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `kl` - Specifies the number of sub-diagonals of the matrix $A$.
    /// * `ku` - Specifies the number of super-diagonals of the matrix $A$.
    /// * `alpha` - Specifies the scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k_l + k_u + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$ when trans = CblasNoTrans, otherwise $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (m-1) \cdot |\\text{incy}|)$ when trans = CblasNoTrans, otherwise $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_dgbmv_64(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        kl: MklCBlasInt64,
        ku: MklCBlasInt64,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
        beta: CBlasDouble,
        y: *mut CBlasDouble,
        incy: MklCBlasInt64,
    );

    /// The ?gbmv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^T \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^H \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are complex scalars, $x$ and $y$ are complex vectors,
    /// and $A$ is an $m \\times n$ complex band matrix with $k_l$ sub-diagonals and $k_u$ super-diagonals.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `kl` - Specifies the number of sub-diagonals of the matrix $A$.
    /// * `ku` - Specifies the number of super-diagonals of the matrix $A$.
    /// * `alpha` - Specifies the complex scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the complex band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k_l + k_u + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$ when trans = CblasNoTrans, otherwise $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the complex scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (m-1) \cdot |\\text{incy}|)$ when trans = CblasNoTrans, otherwise $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_cgbmv_64(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        kl: MklCBlasInt64,
        ku: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        beta: *const CBlasVoid,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?gbmv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^T \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^H \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are complex double-precision scalars, $x$ and $y$ are complex double-precision vectors,
    /// and $A$ is an $m \\times n$ complex double-precision band matrix with $k_l$ sub-diagonals and $k_u$ super-diagonals.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `kl` - Specifies the number of sub-diagonals of the matrix $A$.
    /// * `ku` - Specifies the number of super-diagonals of the matrix $A$.
    /// * `alpha` - Specifies the complex double-precision scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the complex double-precision band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k_l + k_u + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$ when trans = CblasNoTrans, otherwise $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the complex double-precision scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (m-1) \cdot |\\text{incy}|)$ when trans = CblasNoTrans, otherwise $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_zgbmv_64(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        kl: MklCBlasInt64,
        ku: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        beta: *const CBlasVoid,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?gemv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^T \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are scalars, $x$ and $y$ are vectors,
    /// and $A$ is an $m \\times n$ general matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `alpha` - Specifies the scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the $m \\times n$ matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, m)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$ when trans = CblasNoTrans, otherwise $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (m-1) \cdot |\\text{incy}|)$ when trans = CblasNoTrans, otherwise $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_sgemv_64(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
        beta: CBlasFloat,
        y: *mut CBlasFloat,
        incy: MklCBlasInt64,
    );

    /// The ?gemv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^T \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are double-precision scalars, $x$ and $y$ are double-precision vectors,
    /// and $A$ is an $m \\times n$ general double-precision matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `alpha` - Specifies the double-precision scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the $m \\times n$ double-precision matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, m)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$ when trans = CblasNoTrans, otherwise $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the double-precision scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (m-1) \cdot |\\text{incy}|)$ when trans = CblasNoTrans, otherwise $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_dgemv_64(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
        beta: CBlasDouble,
        y: *mut CBlasDouble,
        incy: MklCBlasInt64,
    );

    /// The ?gemv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^T \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^H \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are complex scalars, $x$ and $y$ are complex vectors,
    /// and $A$ is an $m \\times n$ complex general matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `alpha` - Specifies the complex scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the $m \\times n$ complex matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, m)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$ when trans = CblasNoTrans, otherwise $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the complex scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (m-1) \cdot |\\text{incy}|)$ when trans = CblasNoTrans, otherwise $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_cgemv_64(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        beta: *const CBlasVoid,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?gemv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^T \\cdot x + \\beta \\cdot y$$
    ///
    /// or
    ///
    /// $$y := \\alpha \\cdot A^H \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are complex double-precision scalars, $x$ and $y$ are complex double-precision vectors,
    /// and $A$ is an $m \\times n$ complex double-precision general matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `alpha` - Specifies the complex double-precision scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the $m \\times n$ complex double-precision matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, m)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$ when trans = CblasNoTrans, otherwise $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the complex double-precision scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (m-1) \cdot |\\text{incy}|)$ when trans = CblasNoTrans, otherwise $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_zgemv_64(
        layout: CBlasLayout,
        trans: CBlasTranspose,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        beta: *const CBlasVoid,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?ger routine performs a rank-1 update of a general matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^T + A$$
    ///
    /// where $\\alpha$ is a scalar, $x$ is an $m$-element vector, $y$ is an $n$-element vector,
    /// and $A$ is an $m \\times n$ general matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `alpha` - Specifies the scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `a` - Array, size `lda * n`. On entry, the $m \\times n$ matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, m)$.
    pub fn cblas_sger_64(
        layout: CBlasLayout,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
        y: *const CBlasFloat,
        incy: MklCBlasInt64,
        a: *mut CBlasFloat,
        lda: MklCBlasInt64,
    );

    /// The ?ger routine performs a rank-1 update of a general matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^T + A$$
    ///
    /// where $\\alpha$ is a double-precision scalar, $x$ is an $m$-element double-precision vector,
    /// $y$ is an $n$-element double-precision vector, and $A$ is an $m \\times n$ double-precision general matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `alpha` - Specifies the double-precision scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `a` - Array, size `lda * n`. On entry, the $m \\times n$ double-precision matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, m)$.
    pub fn cblas_dger_64(
        layout: CBlasLayout,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
        y: *const CBlasDouble,
        incy: MklCBlasInt64,
        a: *mut CBlasDouble,
        lda: MklCBlasInt64,
    );

    /// The ?gerc routine performs a conjugated rank-1 update of a complex general matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^H + A$$
    ///
    /// where $\\alpha$ is a complex scalar, $x$ is an $m$-element complex vector,
    /// $y$ is an $n$-element complex vector, and $A$ is an $m \\times n$ complex general matrix.
    /// The conjugate of vector $y$ is used in the computation.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `alpha` - Specifies the complex scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `a` - Array, size `lda * n`. On entry, the $m \\times n$ complex matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, m)$.
    pub fn cblas_cgerc_64(
        layout: CBlasLayout,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *const CBlasVoid,
        incy: MklCBlasInt64,
        a: *mut CBlasVoid,
        lda: MklCBlasInt64,
    );

    /// The ?gerc routine performs a conjugated rank-1 update of a complex double-precision general matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^H + A$$
    ///
    /// where $\\alpha$ is a complex double-precision scalar, $x$ is an $m$-element complex double-precision vector,
    /// $y$ is an $n$-element complex double-precision vector, and $A$ is an $m \\times n$ complex double-precision general matrix.
    /// The conjugate of vector $y$ is used in the computation.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `alpha` - Specifies the complex double-precision scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `a` - Array, size `lda * n`. On entry, the $m \\times n$ complex double-precision matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, m)$.
    pub fn cblas_zgerc_64(
        layout: CBlasLayout,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *const CBlasVoid,
        incy: MklCBlasInt64,
        a: *mut CBlasVoid,
        lda: MklCBlasInt64,
    );

    /// The ?geru routine performs an unconjugated rank-1 update of a complex general matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^T + A$$
    ///
    /// where $\\alpha$ is a complex scalar, $x$ is an $m$-element complex vector,
    /// $y$ is an $n$-element complex vector, and $A$ is an $m \\times n$ complex general matrix.
    /// The vector $y$ is used without conjugation in the computation.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `alpha` - Specifies the complex scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `a` - Array, size `lda * n`. On entry, the $m \\times n$ complex matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, m)$.
    pub fn cblas_cgeru_64(
        layout: CBlasLayout,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *const CBlasVoid,
        incy: MklCBlasInt64,
        a: *mut CBlasVoid,
        lda: MklCBlasInt64,
    );

    /// The ?geru routine performs an unconjugated rank-1 update of a complex double-precision general matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^T + A$$
    ///
    /// where $\\alpha$ is a complex double-precision scalar, $x$ is an $m$-element complex double-precision vector,
    /// $y$ is an $n$-element complex double-precision vector, and $A$ is an $m \\times n$ complex double-precision general matrix.
    /// The vector $y$ is used without conjugation in the computation.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `m` - Specifies the number of rows of the matrix $A$.
    /// * `n` - Specifies the number of columns of the matrix $A$.
    /// * `alpha` - Specifies the complex double-precision scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (m-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `a` - Array, size `lda * n`. On entry, the $m \\times n$ complex double-precision matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, m)$.
    pub fn cblas_zgeru_64(
        layout: CBlasLayout,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *const CBlasVoid,
        incy: MklCBlasInt64,
        a: *mut CBlasVoid,
        lda: MklCBlasInt64,
    );

    /// The ?hbmv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are complex scalars, $x$ and $y$ are complex vectors,
    /// and $A$ is an $n \\times n$ complex Hermitian band matrix with $k$ super-diagonals.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `k` - Specifies the number of super-diagonals of the matrix $A$.
    /// * `alpha` - Specifies the complex scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the Hermitian band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the complex scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_chbmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        beta: *const CBlasVoid,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?hbmv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are complex double-precision scalars, $x$ and $y$ are complex double-precision vectors,
    /// and $A$ is an $n \\times n$ complex double-precision Hermitian band matrix with $k$ super-diagonals.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `k` - Specifies the number of super-diagonals of the matrix $A$.
    /// * `alpha` - Specifies the complex double-precision scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the complex double-precision Hermitian band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the complex double-precision scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_zhbmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        beta: *const CBlasVoid,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?hemv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are complex scalars, $x$ and $y$ are complex vectors,
    /// and $A$ is an $n \\times n$ complex Hermitian matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the complex scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the complex Hermitian matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the complex scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_chemv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        beta: *const CBlasVoid,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?hemv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are complex double-precision scalars, $x$ and $y$ are complex double-precision vectors,
    /// and $A$ is an $n \\times n$ complex double-precision Hermitian matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the complex double-precision scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the complex double-precision Hermitian matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the complex double-precision scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_zhemv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        beta: *const CBlasVoid,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?her routine performs a rank-1 update of a Hermitian matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot x^H + A$$
    ///
    /// where $\\alpha$ is a real scalar, $x$ is a complex vector,
    /// and $A$ is an $n \\times n$ complex Hermitian matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the real scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `a` - Array, size `lda * n`. On entry, the complex Hermitian matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    pub fn cblas_cher_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        a: *mut CBlasVoid,
        lda: MklCBlasInt64,
    );

    /// The ?her routine performs a rank-1 update of a Hermitian matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot x^H + A$$
    ///
    /// where $\\alpha$ is a real double-precision scalar, $x$ is a complex double-precision vector,
    /// and $A$ is an $n \\times n$ complex double-precision Hermitian matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the real double-precision scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `a` - Array, size `lda * n`. On entry, the complex double-precision Hermitian matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    pub fn cblas_zher_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        a: *mut CBlasVoid,
        lda: MklCBlasInt64,
    );

    /// The ?her2 routine performs a rank-2 update of a Hermitian matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^H + \\overline{\\alpha} \\cdot y \\cdot x^H + A$$
    ///
    /// where $\\alpha$ is a complex scalar, $x$ and $y$ are complex vectors,
    /// and $A$ is an $n \\times n$ complex Hermitian matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the complex scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `a` - Array, size `lda * n`. On entry, the complex Hermitian matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    pub fn cblas_cher2_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *const CBlasVoid,
        incy: MklCBlasInt64,
        a: *mut CBlasVoid,
        lda: MklCBlasInt64,
    );

    /// The ?her2 routine performs a rank-2 update of a Hermitian matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^H + \\overline{\\alpha} \\cdot y \\cdot x^H + A$$
    ///
    /// where $\\alpha$ is a complex double-precision scalar, $x$ and $y$ are complex double-precision vectors,
    /// and $A$ is an $n \\times n$ complex double-precision Hermitian matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the complex double-precision scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `a` - Array, size `lda * n`. On entry, the complex double-precision Hermitian matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    pub fn cblas_zher2_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *const CBlasVoid,
        incy: MklCBlasInt64,
        a: *mut CBlasVoid,
        lda: MklCBlasInt64,
    );

    /// The ?hpmv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are complex scalars, $x$ and $y$ are complex vectors,
    /// and $A$ is an $n \\times n$ complex Hermitian matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the complex scalar $\\alpha$.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the Hermitian matrix $A$ in packed storage.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the complex scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_chpmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        ap: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        beta: *const CBlasVoid,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?hpmv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are complex double-precision scalars, $x$ and $y$ are complex double-precision vectors,
    /// and $A$ is an $n \\times n$ complex double-precision Hermitian matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the complex double-precision scalar $\\alpha$.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the complex double-precision Hermitian matrix $A$ in packed storage.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the complex double-precision scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_zhpmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        ap: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        beta: *const CBlasVoid,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?hpr routine performs a rank-1 update of a Hermitian packed matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot x^H + A$$
    ///
    /// where $\\alpha$ is a real scalar, $x$ is a complex vector,
    /// and $A$ is an $n \\times n$ complex Hermitian matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the real scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the complex Hermitian matrix $A$ in packed storage. On exit, overwritten by the updated matrix.
    pub fn cblas_chpr_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        ap: *mut CBlasVoid,
    );

    /// The ?hpr routine performs a rank-1 update of a Hermitian packed matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot x^H + A$$
    ///
    /// where $\\alpha$ is a real double-precision scalar, $x$ is a complex double-precision vector,
    /// and $A$ is an $n \\times n$ complex double-precision Hermitian matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of matrix $A$.
    /// * `alpha` - Specifies the real double-precision scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the complex double-precision Hermitian matrix $A$ in packed storage. On exit, overwritten by the updated matrix.
    pub fn cblas_zhpr_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        ap: *mut CBlasVoid,
    );

    /// The ?hpr2 routine performs a rank-2 update of a Hermitian packed matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^H + \\overline{\\alpha} \\cdot y \\cdot x^H + A$$
    ///
    /// where $\\alpha$ is a complex scalar, $x$ and $y$ are complex vectors,
    /// and $A$ is an $n \\times n$ complex Hermitian matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the complex scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the complex Hermitian matrix $A$ in packed storage. On exit, overwritten by the updated matrix.
    pub fn cblas_chpr2_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *const CBlasVoid,
        incy: MklCBlasInt64,
        ap: *mut CBlasVoid,
    );

    /// The ?hpr2 routine performs a rank-2 update of a Hermitian packed matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^H + \\overline{\\alpha} \\cdot y \\cdot x^H + A$$
    ///
    /// where $\\alpha$ is a complex double-precision scalar, $x$ and $y$ are complex double-precision vectors,
    /// and $A$ is an $n \\times n$ complex double-precision Hermitian matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the complex double-precision scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the complex double-precision Hermitian matrix $A$ in packed storage. On exit, overwritten by the updated matrix.
    pub fn cblas_zhpr2_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *const CBlasVoid,
        incy: MklCBlasInt64,
        ap: *mut CBlasVoid,
    );

    /// The ?sbmv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are scalars, $x$ and $y$ are vectors,
    /// and $A$ is an $n \\times n$ symmetric band matrix with $k$ super-diagonals.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `k` - Specifies the number of super-diagonals of the matrix $A$.
    /// * `alpha` - Specifies the scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the symmetric band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_ssbmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
        beta: CBlasFloat,
        y: *mut CBlasFloat,
        incy: MklCBlasInt64,
    );

    /// The ?sbmv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are double-precision scalars, $x$ and $y$ are double-precision vectors,
    /// and $A$ is an $n \\times n$ double-precision symmetric band matrix with $k$ super-diagonals.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `k` - Specifies the number of super-diagonals of the matrix $A$.
    /// * `alpha` - Specifies the double-precision scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the double-precision symmetric band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the double-precision scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_dsbmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
        beta: CBlasDouble,
        y: *mut CBlasDouble,
        incy: MklCBlasInt64,
    );

    /// The ?spmv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are scalars, $x$ and $y$ are vectors,
    /// and $A$ is an $n \\times n$ symmetric matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the scalar $\\alpha$.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the symmetric matrix $A$ in packed storage.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_sspmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        ap: *const CBlasFloat,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
        beta: CBlasFloat,
        y: *mut CBlasFloat,
        incy: MklCBlasInt64,
    );

    /// The ?spmv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are double-precision scalars, $x$ and $y$ are double-precision vectors,
    /// and $A$ is an $n \\times n$ double-precision symmetric matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the double-precision scalar $\\alpha$.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the double-precision symmetric matrix $A$ in packed storage.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the double-precision scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_dspmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        ap: *const CBlasDouble,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
        beta: CBlasDouble,
        y: *mut CBlasDouble,
        incy: MklCBlasInt64,
    );

    /// The ?spr routine performs a rank-1 update of a symmetric packed matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot x^T + A$$
    ///
    /// where $\\alpha$ is a scalar, $x$ is a vector,
    /// and $A$ is an $n \\times n$ symmetric matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the symmetric matrix $A$ in packed storage. On exit, overwritten by the updated matrix.
    pub fn cblas_sspr_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
        ap: *mut CBlasFloat,
    );

    /// The ?spr routine performs a rank-1 update of a symmetric packed matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot x^T + A$$
    ///
    /// where $\\alpha$ is a double-precision scalar, $x$ is a double-precision vector,
    /// and $A$ is an $n \\times n$ double-precision symmetric matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the double-precision scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the double-precision symmetric matrix $A$ in packed storage. On exit, overwritten by the updated matrix.
    pub fn cblas_dspr_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
        ap: *mut CBlasDouble,
    );

    /// The ?spr2 routine performs a rank-2 update of a symmetric packed matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^T + \\alpha \\cdot y \\cdot x^T + A$$
    ///
    /// where $\\alpha$ is a scalar, $x$ and $y$ are vectors,
    /// and $A$ is an $n \\times n$ symmetric matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the symmetric matrix $A$ in packed storage. On exit, overwritten by the updated matrix.
    pub fn cblas_sspr2_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
        y: *const CBlasFloat,
        incy: MklCBlasInt64,
        ap: *mut CBlasFloat,
    );

    /// The ?spr2 routine performs a rank-2 update of a symmetric packed matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^T + \\alpha \\cdot y \\cdot x^T + A$$
    ///
    /// where $\\alpha$ is a double-precision scalar, $x$ and $y$ are double-precision vectors,
    /// and $A$ is an $n \\times n$ double-precision symmetric matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the double-precision scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the double-precision symmetric matrix $A$ in packed storage. On exit, overwritten by the updated matrix.
    pub fn cblas_dspr2_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
        y: *const CBlasDouble,
        incy: MklCBlasInt64,
        ap: *mut CBlasDouble,
    );

    /// The ?symv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are scalars, $x$ and $y$ are vectors,
    /// and $A$ is an $n \\times n$ symmetric matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the symmetric matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_ssymv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
        beta: CBlasFloat,
        y: *mut CBlasFloat,
        incy: MklCBlasInt64,
    );

    /// The ?symv routine performs a matrix-vector operation defined as
    ///
    /// $$y := \\alpha \\cdot A \\cdot x + \\beta \\cdot y$$
    ///
    /// where $\\alpha$ and $\\beta$ are double-precision scalars, $x$ and $y$ are double-precision vectors,
    /// and $A$ is an $n \\times n$ double-precision symmetric matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the double-precision scalar $\\alpha$.
    /// * `a` - Array, size `lda * n`. On entry, the double-precision symmetric matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `beta` - Specifies the double-precision scalar $\\beta$.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    pub fn cblas_dsymv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
        beta: CBlasDouble,
        y: *mut CBlasDouble,
        incy: MklCBlasInt64,
    );

    /// The ?syr routine performs a rank-1 update of a symmetric matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot x^T + A$$
    ///
    /// where $\\alpha$ is a scalar, $x$ is a vector,
    /// and $A$ is an $n \\times n$ symmetric matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `a` - Array, size `lda * n`. On entry, the symmetric matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    pub fn cblas_ssyr_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
        a: *mut CBlasFloat,
        lda: MklCBlasInt64,
    );

    /// The ?syr routine performs a rank-1 update of a symmetric matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot x^T + A$$
    ///
    /// where $\\alpha$ is a double-precision scalar, $x$ is a double-precision vector,
    /// and $A$ is an $n \\times n$ double-precision symmetric matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the double-precision scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `a` - Array, size `lda * n`. On entry, the double-precision symmetric matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    pub fn cblas_dsyr_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
        a: *mut CBlasDouble,
        lda: MklCBlasInt64,
    );

    /// The ?syr2 routine performs a rank-2 update of a symmetric matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^T + \\alpha \\cdot y \\cdot x^T + A$$
    ///
    /// where $\\alpha$ is a scalar, $x$ and $y$ are vectors,
    /// and $A$ is an $n \\times n$ symmetric matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `a` - Array, size `lda * n`. On entry, the symmetric matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    pub fn cblas_ssyr2_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
        y: *const CBlasFloat,
        incy: MklCBlasInt64,
        a: *mut CBlasFloat,
        lda: MklCBlasInt64,
    );

    /// The ?syr2 routine performs a rank-2 update of a symmetric matrix defined as
    ///
    /// $$A := \\alpha \\cdot x \\cdot y^T + \\alpha \\cdot y \\cdot x^T + A$$
    ///
    /// where $\\alpha$ is a double-precision scalar, $x$ and $y$ are double-precision vectors,
    /// and $A$ is an $n \\times n$ double-precision symmetric matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used: CblasUpper or CblasLower.
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `alpha` - Specifies the double-precision scalar $\\alpha$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    /// * `y` - Array, size at least $(1 + (n-1) \cdot |\\text{incy}|)$.
    /// * `incy` - Specifies the increment for indexing vector `y`.
    /// * `a` - Array, size `lda * n`. On entry, the double-precision symmetric matrix $A$. On exit, overwritten by the updated matrix.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    pub fn cblas_dsyr2_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
        y: *const CBlasDouble,
        incy: MklCBlasInt64,
        a: *mut CBlasDouble,
        lda: MklCBlasInt64,
    );

    /// The ?tbmv routine performs a matrix-vector operation defined as
    ///
    /// $$x := A \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^T \\cdot x$$
    ///
    /// where $x$ is a vector and $A$ is an $n \\times n$ triangular band matrix with $k$ super-diagonals or sub-diagonals.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `k` - Specifies the number of super-diagonals (if uplo = CblasUpper) or sub-diagonals (if uplo = CblasLower) of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the triangular band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the vector $x$. On exit, overwritten by the transformed vector.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_stbmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        x: *mut CBlasFloat,
        incx: MklCBlasInt64,
    );

    /// The ?tbmv routine performs a matrix-vector operation defined as
    ///
    /// $$x := A \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^T \\cdot x$$
    ///
    /// where $x$ is a double-precision vector and $A$ is an $n \\times n$ double-precision triangular band matrix with $k$ super-diagonals or sub-diagonals.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `k` - Specifies the number of super-diagonals (if uplo = CblasUpper) or sub-diagonals (if uplo = CblasLower) of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the double-precision triangular band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the double-precision vector $x$. On exit, overwritten by the transformed vector.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_dtbmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        x: *mut CBlasDouble,
        incx: MklCBlasInt64,
    );

    /// The ?tbmv routine performs a matrix-vector operation defined as
    ///
    /// $$x := A \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^T \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^H \\cdot x$$
    ///
    /// where $x$ is a complex vector and $A$ is an $n \\times n$ complex triangular band matrix with $k$ super-diagonals or sub-diagonals.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `k` - Specifies the number of super-diagonals (if uplo = CblasUpper) or sub-diagonals (if uplo = CblasLower) of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the complex triangular band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the complex vector $x$. On exit, overwritten by the transformed vector.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_ctbmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The ?tbmv routine performs a matrix-vector operation defined as
    ///
    /// $$x := A \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^T \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^H \\cdot x$$
    ///
    /// where $x$ is a complex double-precision vector and $A$ is an $n \\times n$ complex double-precision triangular band matrix with $k$ super-diagonals or sub-diagonals.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `k` - Specifies the number of super-diagonals (if uplo = CblasUpper) or sub-diagonals (if uplo = CblasLower) of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the complex double-precision triangular band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the complex double-precision vector $x$. On exit, overwritten by the transformed vector.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_ztbmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The ?tbsv routine solves a system of linear equations defined as
    ///
    /// $$A \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^T \\cdot x = b$$
    ///
    /// where $x$ and $b$ are vectors and $A$ is an $n \\times n$ triangular band matrix with $k$ super-diagonals or sub-diagonals.
    /// The solution $x$ overwrites the input vector $b$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `k` - Specifies the number of super-diagonals (if uplo = CblasUpper) or sub-diagonals (if uplo = CblasLower) of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the triangular band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the right-hand side vector $b$. On exit, overwritten by the solution vector $x$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_stbsv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        x: *mut CBlasFloat,
        incx: MklCBlasInt64,
    );

    /// The ?tbsv routine solves a system of linear equations defined as
    ///
    /// $$A \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^T \\cdot x = b$$
    ///
    /// where $x$ and $b$ are double-precision vectors and $A$ is an $n \\times n$ double-precision triangular band matrix with $k$ super-diagonals or sub-diagonals.
    /// The solution $x$ overwrites the input vector $b$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `k` - Specifies the number of super-diagonals (if uplo = CblasUpper) or sub-diagonals (if uplo = CblasLower) of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the double-precision triangular band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the right-hand side double-precision vector $b$. On exit, overwritten by the solution vector $x$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_dtbsv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        x: *mut CBlasDouble,
        incx: MklCBlasInt64,
    );

    /// The ?tbsv routine solves a system of linear equations defined as
    ///
    /// $$A \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^T \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^H \\cdot x = b$$
    ///
    /// where $x$ and $b$ are complex vectors and $A$ is an $n \\times n$ complex triangular band matrix with $k$ super-diagonals or sub-diagonals.
    /// The solution $x$ overwrites the input vector $b$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `k` - Specifies the number of super-diagonals (if uplo = CblasUpper) or sub-diagonals (if uplo = CblasLower) of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the complex triangular band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the right-hand side complex vector $b$. On exit, overwritten by the solution vector $x$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_ctbsv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The ?tbsv routine solves a system of linear equations defined as
    ///
    /// $$A \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^T \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^H \\cdot x = b$$
    ///
    /// where $x$ and $b$ are complex double-precision vectors and $A$ is an $n \\times n$ complex double-precision triangular band matrix with $k$ super-diagonals or sub-diagonals.
    /// The solution $x$ overwrites the input vector $b$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `k` - Specifies the number of super-diagonals (if uplo = CblasUpper) or sub-diagonals (if uplo = CblasLower) of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the complex double-precision triangular band matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $(k + 1)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the right-hand side complex double-precision vector $b$. On exit, overwritten by the solution vector $x$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_ztbsv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The ?tpmv routine performs a matrix-vector operation defined as
    ///
    /// $$x := A \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^T \\cdot x$$
    ///
    /// where $x$ is a vector and $A$ is an $n \\times n$ triangular matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the triangular matrix $A$ in packed storage.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the vector $x$. On exit, overwritten by the transformed vector.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_stpmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        ap: *const CBlasFloat,
        x: *mut CBlasFloat,
        incx: MklCBlasInt64,
    );

    /// The ?tpmv routine performs a matrix-vector operation defined as
    ///
    /// $$x := A \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^T \\cdot x$$
    ///
    /// where $x$ is a double-precision vector and $A$ is an $n \\times n$ double-precision triangular matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the double-precision triangular matrix $A$ in packed storage.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the double-precision vector $x$. On exit, overwritten by the transformed vector.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_dtpmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        ap: *const CBlasDouble,
        x: *mut CBlasDouble,
        incx: MklCBlasInt64,
    );

    /// The ?tpmv routine performs a matrix-vector operation defined as
    ///
    /// $$x := A \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^T \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^H \\cdot x$$
    ///
    /// where $x$ is a complex vector and $A$ is an $n \\times n$ complex triangular matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the complex triangular matrix $A$ in packed storage.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the complex vector $x$. On exit, overwritten by the transformed vector.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_ctpmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        ap: *const CBlasVoid,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The ?tpmv routine performs a matrix-vector operation defined as
    ///
    /// $$x := A \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^T \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^H \\cdot x$$
    ///
    /// where $x$ is a complex double-precision vector and $A$ is an $n \\times n$ complex double-precision triangular matrix stored in packed format.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the complex double-precision triangular matrix $A$ in packed storage.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the complex double-precision vector $x$. On exit, overwritten by the transformed vector.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_ztpmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        ap: *const CBlasVoid,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The ?tpsv routine solves a system of linear equations defined as
    ///
    /// $$A \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^T \\cdot x = b$$
    ///
    /// where $x$ and $b$ are vectors and $A$ is an $n \\times n$ triangular matrix stored in packed format.
    /// The solution $x$ overwrites the input vector $b$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the triangular matrix $A$ in packed storage.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the right-hand side vector $b$. On exit, overwritten by the solution vector $x$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_stpsv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        ap: *const CBlasFloat,
        x: *mut CBlasFloat,
        incx: MklCBlasInt64,
    );

    /// The ?tpsv routine solves a system of linear equations defined as
    ///
    /// $$A \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^T \\cdot x = b$$
    ///
    /// where $x$ and $b$ are double-precision vectors and $A$ is an $n \\times n$ double-precision triangular matrix stored in packed format.
    /// The solution $x$ overwrites the input vector $b$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the double-precision triangular matrix $A$ in packed storage.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the right-hand side double-precision vector $b$. On exit, overwritten by the solution vector $x$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_dtpsv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        ap: *const CBlasDouble,
        x: *mut CBlasDouble,
        incx: MklCBlasInt64,
    );

    /// The ?tpsv routine solves a system of linear equations defined as
    ///
    /// $$A \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^T \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^H \\cdot x = b$$
    ///
    /// where $x$ and $b$ are complex vectors and $A$ is an $n \\times n$ complex triangular matrix stored in packed format.
    /// The solution $x$ overwrites the input vector $b$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the complex triangular matrix $A$ in packed storage.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the right-hand side complex vector $b$. On exit, overwritten by the solution vector $x$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_ctpsv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        ap: *const CBlasVoid,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The ?tpsv routine solves a system of linear equations defined as
    ///
    /// $$A \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^T \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^H \\cdot x = b$$
    ///
    /// where $x$ and $b$ are complex double-precision vectors and $A$ is an $n \\times n$ complex double-precision triangular matrix stored in packed format.
    /// The solution $x$ overwrites the input vector $b$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `ap` - Array, size at least $(n(n+1)/2)$. On entry, the complex double-precision triangular matrix $A$ in packed storage.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the right-hand side complex double-precision vector $b$. On exit, overwritten by the solution vector $x$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_ztpsv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        ap: *const CBlasVoid,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The ?trmv routine performs a matrix-vector operation defined as
    ///
    /// $$x := A \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^T \\cdot x$$
    ///
    /// where $x$ is a vector and $A$ is an $n \\times n$ triangular matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the triangular matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the vector $x$. On exit, overwritten by the transformed vector.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_strmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        x: *mut CBlasFloat,
        incx: MklCBlasInt64,
    );

    /// The ?trmv routine performs a matrix-vector operation defined as
    ///
    /// $$x := A \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^T \\cdot x$$
    ///
    /// where $x$ is a double-precision vector and $A$ is an $n \\times n$ double-precision triangular matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the double-precision triangular matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the double-precision vector $x$. On exit, overwritten by the transformed vector.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_dtrmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        x: *mut CBlasDouble,
        incx: MklCBlasInt64,
    );

    /// The ?trmv routine performs a matrix-vector operation defined as
    ///
    /// $$x := A \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^T \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^H \\cdot x$$
    ///
    /// where $x$ is a complex vector and $A$ is an $n \\times n$ complex triangular matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the complex triangular matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the complex vector $x$. On exit, overwritten by the transformed vector.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_ctrmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The ?trmv routine performs a matrix-vector operation defined as
    ///
    /// $$x := A \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^T \\cdot x$$
    ///
    /// or
    ///
    /// $$x := A^H \\cdot x$$
    ///
    /// where $x$ is a complex double-precision vector and $A$ is an $n \\times n$ complex double-precision triangular matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the complex double-precision triangular matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the complex double-precision vector $x$. On exit, overwritten by the transformed vector.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_ztrmv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The ?trsv routine solves a system of linear equations defined as
    ///
    /// $$A \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^T \\cdot x = b$$
    ///
    /// where $x$ and $b$ are vectors and $A$ is an $n \\times n$ triangular matrix.
    /// The solution $x$ overwrites the input vector $b$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the triangular matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the right-hand side vector $b$. On exit, overwritten by the solution vector $x$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_strsv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        x: *mut CBlasFloat,
        incx: MklCBlasInt64,
    );

    /// The ?trsv routine solves a system of linear equations defined as
    ///
    /// $$A \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^T \\cdot x = b$$
    ///
    /// where $x$ and $b$ are double-precision vectors and $A$ is an $n \\times n$ double-precision triangular matrix.
    /// The solution $x$ overwrites the input vector $b$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the double-precision triangular matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the right-hand side double-precision vector $b$. On exit, overwritten by the solution vector $x$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_dtrsv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        x: *mut CBlasDouble,
        incx: MklCBlasInt64,
    );

    /// The ?trsv routine solves a system of linear equations defined as
    ///
    /// $$A \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^T \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^H \\cdot x = b$$
    ///
    /// where $x$ and $b$ are complex vectors and $A$ is an $n \\times n$ complex triangular matrix.
    /// The solution $x$ overwrites the input vector $b$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the complex triangular matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the right-hand side complex vector $b$. On exit, overwritten by the solution vector $x$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_ctrsv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The ?trsv routine solves a system of linear equations defined as
    ///
    /// $$A \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^T \\cdot x = b$$
    ///
    /// or
    ///
    /// $$A^H \\cdot x = b$$
    ///
    /// where $x$ and $b$ are complex double-precision vectors and $A$ is an $n \\times n$ complex double-precision triangular matrix.
    /// The solution $x$ overwrites the input vector $b$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the matrix $A$ is used: CblasUpper or CblasLower.
    /// * `trans` - Specifies the operation: CblasNoTrans, CblasTrans, or CblasConjTrans.
    /// * `diag` - Specifies whether the matrix $A$ is unit triangular (CblasUnit) or non-unit triangular (CblasNonUnit).
    /// * `n` - Specifies the order of the matrix $A$.
    /// * `a` - Array, size `lda * n`. On entry, the complex double-precision triangular matrix $A$.
    /// * `lda` - Specifies the leading dimension of `a` as declared in the calling program. Must be at least $\\max(1, n)$.
    /// * `x` - Array, size at least $(1 + (n-1) \cdot |\\text{incx}|)$. On entry, the right-hand side complex double-precision vector $b$. On exit, overwritten by the solution vector $x$.
    /// * `incx` - Specifies the increment for indexing vector `x`.
    pub fn cblas_ztrsv_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        diag: CBlasDiag,
        n: MklCBlasInt64,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

}
