//! Explicit Intel MKL `*_64` Level 3 CBLAS entry points.
//!
//! Every declaration in this module binds an actual suffixed MKL symbol and
//! uses `MKL_INT64`, independent of the ordinary ABI feature.

use crate::cblas::cblas_types::*;

unsafe extern "C" {

    /// The ?gemm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product.
    /// The operation is defined as
    ///
    /// $$C := \alpha \cdot \mathrm{op}(A) \cdot \mathrm{op}(B) + \beta \cdot C$$
    ///
    /// where $\mathrm{op}(X)$ is one of $X$, $X^{\top}$, or $X^{\mathrm{H}}$; $\alpha$ and $\beta$ are scalars;
    /// $A$ is an $m$-by-$k$ matrix, $B$ is a $k$-by-$n$ matrix, and $C$ is an $m$-by-$n$ matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `transa` - Specifies the form of $\mathrm{op}(A)$:
    ///     * CblasNoTrans - $\mathrm{op}(A) = A$
    ///     * CblasTrans - $\mathrm{op}(A) = A^{\top}$
    ///     * CblasConjTrans - $\mathrm{op}(A) = A^{\mathrm{H}}$
    /// * `transb` - Specifies the form of $\mathrm{op}(B)$.
    /// * `m` - Specifies the number of rows of $\mathrm{op}(A)$ and $C$. Must be at least zero.
    /// * `n` - Specifies the number of columns of $\mathrm{op}(B)$ and $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of $\mathrm{op}(A)$ and rows of $\mathrm{op}(B)$. Must be at least zero.
    /// * `alpha` - Specifies the scalar $\alpha$.
    /// * `a` - Array containing the matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array containing the matrix $B$.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, m)$.
    pub fn cblas_sgemm_64(
        layout: CBlasLayout,
        transa: CBlasTranspose,
        transb: CBlasTranspose,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        b: *const CBlasFloat,
        ldb: MklCBlasInt64,
        beta: CBlasFloat,
        c: *mut CBlasFloat,
        ldc: MklCBlasInt64,
    );

    /// The ?gemm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product.
    /// The operation is defined as
    ///
    /// $$C := \alpha \cdot \mathrm{op}(A) \cdot \mathrm{op}(B) + \beta \cdot C$$
    ///
    /// where $\mathrm{op}(X)$ is one of $X$, $X^{\top}$, or $X^{\mathrm{H}}$; $\alpha$ and $\beta$ are scalars;
    /// $A$ is an $m$-by-$k$ matrix, $B$ is a $k$-by-$n$ matrix, and $C$ is an $m$-by-$n$ matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `transa` - Specifies the form of $\mathrm{op}(A)$:
    ///     * CblasNoTrans - $\mathrm{op}(A) = A$
    ///     * CblasTrans - $\mathrm{op}(A) = A^{\top}$
    ///     * CblasConjTrans - $\mathrm{op}(A) = A^{\mathrm{H}}$
    /// * `transb` - Specifies the form of $\mathrm{op}(B)$.
    /// * `m` - Specifies the number of rows of $\mathrm{op}(A)$ and $C$. Must be at least zero.
    /// * `n` - Specifies the number of columns of $\mathrm{op}(B)$ and $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of $\mathrm{op}(A)$ and rows of $\mathrm{op}(B)$. Must be at least zero.
    /// * `alpha` - Specifies the scalar $\alpha$.
    /// * `a` - Array containing the matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array containing the matrix $B$.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, m)$.
    pub fn cblas_dgemm_64(
        layout: CBlasLayout,
        transa: CBlasTranspose,
        transb: CBlasTranspose,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        b: *const CBlasDouble,
        ldb: MklCBlasInt64,
        beta: CBlasDouble,
        c: *mut CBlasDouble,
        ldc: MklCBlasInt64,
    );

    /// The ?gemm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product.
    /// The operation is defined as
    ///
    /// $$C := \alpha \cdot \mathrm{op}(A) \cdot \mathrm{op}(B) + \beta \cdot C$$
    ///
    /// where $\mathrm{op}(X)$ is one of $X$, $X^{\top}$, or $X^{\mathrm{H}}$; $\alpha$ and $\beta$ are complex scalars;
    /// $A$ is an $m$-by-$k$ matrix, $B$ is a $k$-by-$n$ matrix, and $C$ is an $m$-by-$n$ matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `transa` - Specifies the form of $\mathrm{op}(A)$:
    ///     * CblasNoTrans - $\mathrm{op}(A) = A$
    ///     * CblasTrans - $\mathrm{op}(A) = A^{\top}$
    ///     * CblasConjTrans - $\mathrm{op}(A) = A^{\mathrm{H}}$
    /// * `transb` - Specifies the form of $\mathrm{op}(B)$.
    /// * `m` - Specifies the number of rows of $\mathrm{op}(A)$ and $C$. Must be at least zero.
    /// * `n` - Specifies the number of columns of $\mathrm{op}(B)$ and $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of $\mathrm{op}(A)$ and rows of $\mathrm{op}(B)$. Must be at least zero.
    /// * `alpha` - Specifies the complex scalar $\alpha$.
    /// * `a` - Array containing the matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array containing the matrix $B$.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the complex scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, m)$.
    pub fn cblas_cgemm_64(
        layout: CBlasLayout,
        transa: CBlasTranspose,
        transb: CBlasTranspose,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?gemm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product.
    /// The operation is defined as
    ///
    /// $$C := \alpha \cdot \mathrm{op}(A) \cdot \mathrm{op}(B) + \beta \cdot C$$
    ///
    /// where $\mathrm{op}(X)$ is one of $X$, $X^{\top}$, or $X^{\mathrm{H}}$; $\alpha$ and $\beta$ are complex double scalars;
    /// $A$ is an $m$-by-$k$ matrix, $B$ is a $k$-by-$n$ matrix, and $C$ is an $m$-by-$n$ matrix.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `transa` - Specifies the form of $\mathrm{op}(A)$:
    ///     * CblasNoTrans - $\mathrm{op}(A) = A$
    ///     * CblasTrans - $\mathrm{op}(A) = A^{\top}$
    ///     * CblasConjTrans - $\mathrm{op}(A) = A^{\mathrm{H}}$
    /// * `transb` - Specifies the form of $\mathrm{op}(B)$.
    /// * `m` - Specifies the number of rows of $\mathrm{op}(A)$ and $C$. Must be at least zero.
    /// * `n` - Specifies the number of columns of $\mathrm{op}(B)$ and $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of $\mathrm{op}(A)$ and rows of $\mathrm{op}(B)$. Must be at least zero.
    /// * `alpha` - Specifies the complex double scalar $\alpha$.
    /// * `a` - Array containing the matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array containing the matrix $B$.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the complex double scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, m)$.
    pub fn cblas_zgemm_64(
        layout: CBlasLayout,
        transa: CBlasTranspose,
        transb: CBlasTranspose,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?hemm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product,
    /// where one of the matrices is Hermitian. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot B + \beta \cdot C \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot B \cdot A + \beta \cdot C \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ and $\beta$ are scalars, $A$ is a Hermitian matrix, and $B$ and $C$ are $m$-by-$n$ matrices.
    /// The Hermitian matrix $A$ is stored using only the upper or lower triangular part.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the Hermitian matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the operation.
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used:
    ///     * CblasUpper - Upper triangular part.
    ///     * CblasLower - Lower triangular part.
    /// * `m` - Specifies the number of rows of matrix $C$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $C$. Must be at least zero.
    /// * `alpha` - Specifies the complex scalar $\alpha$.
    /// * `a` - Array of size lda by k, where k = m if side = CblasLeft, k = n if side = CblasRight.
    ///          Contains the Hermitian matrix $A$.
    /// * `lda` - Leading dimension of a. Must be at least $\max(1, m)$ if side = CblasLeft, or $\max(1, n)$ if side = CblasRight.
    /// * `b` - Array of size ldb by n. Contains the matrix $B$.
    /// * `ldb` - Leading dimension of b. Must be at least $\max(1, m)$.
    /// * `beta` - Specifies the complex scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, m)$.
    pub fn cblas_chemm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?hemm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product,
    /// where one of the matrices is Hermitian. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot B + \beta \cdot C \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot B \cdot A + \beta \cdot C \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ and $\beta$ are scalars, $A$ is a Hermitian matrix, and $B$ and $C$ are $m$-by-$n$ matrices.
    /// The Hermitian matrix $A$ is stored using only the upper or lower triangular part.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the Hermitian matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the operation.
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $A$ is used:
    ///     * CblasUpper - Upper triangular part.
    ///     * CblasLower - Lower triangular part.
    /// * `m` - Specifies the number of rows of matrix $C$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $C$. Must be at least zero.
    /// * `alpha` - Specifies the complex double scalar $\alpha$.
    /// * `a` - Array of size lda by k, where k = m if side = CblasLeft, k = n if side = CblasRight.
    ///          Contains the Hermitian matrix $A$.
    /// * `lda` - Leading dimension of a. Must be at least $\max(1, m)$ if side = CblasLeft, or $\max(1, n)$ if side = CblasRight.
    /// * `b` - Array of size ldb by n. Contains the matrix $B$.
    /// * `ldb` - Leading dimension of b. Must be at least $\max(1, m)$.
    /// * `beta` - Specifies the complex double scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, m)$.
    pub fn cblas_zhemm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?herk routines perform a rank-k update of a Hermitian matrix. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot A^{\mathrm{H}} + \beta \cdot C \quad \text{(if trans = CblasNoTrans)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot A^{\mathrm{H}} \cdot A + \beta \cdot C \quad \text{(if trans = CblasTrans or CblasConjTrans)}$$
    ///
    /// where $\alpha$ and $\beta$ are real scalars, $C$ is an $n$-by-$n$ Hermitian matrix, and $A$ is an $n$-by-$k$ matrix
    /// (if trans = CblasNoTrans) or $k$-by-$n$ matrix (if trans = CblasTrans).
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $C$ is used:
    ///     * CblasUpper - Upper triangular part.
    ///     * CblasLower - Lower triangular part.
    /// * `trans` - Specifies the operation:
    ///     * CblasNoTrans - $C := \alpha \cdot A \cdot A^{\mathrm{H}} + \beta \cdot C$
    ///     * CblasTrans or CblasConjTrans - $C := \alpha \cdot A^{\mathrm{H}} \cdot A + \beta \cdot C$
    /// * `n` - Specifies the order of matrix $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of matrix $A$ (if trans = CblasNoTrans) or rows of matrix $A$ (if trans = CblasTrans). Must be at least zero.
    /// * `alpha` - Specifies the real scalar $\alpha$.
    /// * `a` - Array of size lda by ka, where ka = k if trans = CblasNoTrans, ka = n if trans = CblasTrans.
    /// * `lda` - Leading dimension of a. Must be at least $\max(1, n)$ if trans = CblasNoTrans, or $\max(1, k)$ if trans = CblasTrans.
    /// * `beta` - Specifies the real scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains the Hermitian matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, n)$.
    pub fn cblas_cherk_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: CBlasFloat,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        beta: CBlasFloat,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?herk routines perform a rank-k update of a Hermitian matrix. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot A^{\mathrm{H}} + \beta \cdot C \quad \text{(if trans = CblasNoTrans)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot A^{\mathrm{H}} \cdot A + \beta \cdot C \quad \text{(if trans = CblasTrans or CblasConjTrans)}$$
    ///
    /// where $\alpha$ and $\beta$ are real scalars, $C$ is an $n$-by-$n$ Hermitian matrix, and $A$ is an $n$-by-$k$ matrix
    /// (if trans = CblasNoTrans) or $k$-by-$n$ matrix (if trans = CblasTrans).
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $C$ is used:
    ///     * CblasUpper - Upper triangular part.
    ///     * CblasLower - Lower triangular part.
    /// * `trans` - Specifies the operation:
    ///     * CblasNoTrans - $C := \alpha \cdot A \cdot A^{\mathrm{H}} + \beta \cdot C$
    ///     * CblasTrans or CblasConjTrans - $C := \alpha \cdot A^{\mathrm{H}} \cdot A + \beta \cdot C$
    /// * `n` - Specifies the order of matrix $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of matrix $A$ (if trans = CblasNoTrans) or rows of matrix $A$ (if trans = CblasTrans). Must be at least zero.
    /// * `alpha` - Specifies the real scalar $\alpha$.
    /// * `a` - Array of size lda by ka, where ka = k if trans = CblasNoTrans, ka = n if trans = CblasTrans.
    /// * `lda` - Leading dimension of a. Must be at least $\max(1, n)$ if trans = CblasNoTrans, or $\max(1, k)$ if trans = CblasTrans.
    /// * `beta` - Specifies the real scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains the Hermitian matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, n)$.
    pub fn cblas_zherk_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: CBlasDouble,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        beta: CBlasDouble,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?her2k routines perform a rank-2k update of a Hermitian matrix. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot B^{\mathrm{H}} + \overline{\alpha} \cdot B \cdot A^{\mathrm{H}} + \beta \cdot C \quad \text{(if trans = CblasNoTrans)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot A^{\mathrm{H}} \cdot B + \overline{\alpha} \cdot B^{\mathrm{H}} \cdot A + \beta \cdot C \quad \text{(if trans = CblasTrans or CblasConjTrans)}$$
    ///
    /// where $\alpha$ is a complex scalar, $\beta$ is a real scalar, $C$ is an $n$-by-$n$ Hermitian matrix,
    /// and $A$ and $B$ are $n$-by-$k$ matrices (if trans = CblasNoTrans) or $k$-by-$n$ matrices (if trans = CblasTrans).
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $C$ is used.
    /// * `trans` - Specifies the operation:
    ///     * CblasNoTrans - $C := \alpha \cdot A \cdot B^{\mathrm{H}} + \overline{\alpha} \cdot B \cdot A^{\mathrm{H}} + \beta \cdot C$
    ///     * CblasTrans or CblasConjTrans - $C := \alpha \cdot A^{\mathrm{H}} \cdot B + \overline{\alpha} \cdot B^{\mathrm{H}} \cdot A + \beta \cdot C$
    /// * `n` - Specifies the order of matrix $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of matrices $A$ and $B$ (if trans = CblasNoTrans) or rows (if trans = CblasTrans).
    /// * `alpha` - Specifies the complex scalar $\alpha$.
    /// * `a` - Array of size lda by ka.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by kb.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the real scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains the Hermitian matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, n)$.
    pub fn cblas_cher2k_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: CBlasFloat,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?her2k routines perform a rank-2k update of a Hermitian matrix. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot B^{\mathrm{H}} + \overline{\alpha} \cdot B \cdot A^{\mathrm{H}} + \beta \cdot C \quad \text{(if trans = CblasNoTrans)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot A^{\mathrm{H}} \cdot B + \overline{\alpha} \cdot B^{\mathrm{H}} \cdot A + \beta \cdot C \quad \text{(if trans = CblasTrans or CblasConjTrans)}$$
    ///
    /// where $\alpha$ is a complex scalar, $\beta$ is a real scalar, $C$ is an $n$-by-$n$ Hermitian matrix,
    /// and $A$ and $B$ are $n$-by-$k$ matrices (if trans = CblasNoTrans) or $k$-by-$n$ matrices (if trans = CblasTrans).
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the Hermitian matrix $C$ is used.
    /// * `trans` - Specifies the operation:
    ///     * CblasNoTrans - $C := \alpha \cdot A \cdot B^{\mathrm{H}} + \overline{\alpha} \cdot B \cdot A^{\mathrm{H}} + \beta \cdot C$
    ///     * CblasTrans or CblasConjTrans - $C := \alpha \cdot A^{\mathrm{H}} \cdot B + \overline{\alpha} \cdot B^{\mathrm{H}} \cdot A + \beta \cdot C$
    /// * `n` - Specifies the order of matrix $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of matrices $A$ and $B$ (if trans = CblasNoTrans) or rows (if trans = CblasTrans).
    /// * `alpha` - Specifies the complex double scalar $\alpha$.
    /// * `a` - Array of size lda by ka.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by kb.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the real scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains the Hermitian matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, n)$.
    pub fn cblas_zher2k_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: CBlasDouble,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?symm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product,
    /// where one of the matrices is symmetric. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot B + \beta \cdot C \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot B \cdot A + \beta \cdot C \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ and $\beta$ are scalars, $A$ is a symmetric matrix, and $B$ and $C$ are $m$-by-$n$ matrices.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the symmetric matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the operation.
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used.
    /// * `m` - Specifies the number of rows of matrix $C$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $C$. Must be at least zero.
    /// * `alpha` - Specifies the scalar $\alpha$.
    /// * `a` - Array containing the symmetric matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array containing the matrix $B$.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, m)$.
    pub fn cblas_ssymm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        b: *const CBlasFloat,
        ldb: MklCBlasInt64,
        beta: CBlasFloat,
        c: *mut CBlasFloat,
        ldc: MklCBlasInt64,
    );

    /// The ?symm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product,
    /// where one of the matrices is symmetric. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot B + \beta \cdot C \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot B \cdot A + \beta \cdot C \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ and $\beta$ are scalars, $A$ is a symmetric matrix, and $B$ and $C$ are $m$-by-$n$ matrices.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the symmetric matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the operation.
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used.
    /// * `m` - Specifies the number of rows of matrix $C$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $C$. Must be at least zero.
    /// * `alpha` - Specifies the scalar $\alpha$.
    /// * `a` - Array containing the symmetric matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array containing the matrix $B$.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, m)$.
    pub fn cblas_dsymm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        b: *const CBlasDouble,
        ldb: MklCBlasInt64,
        beta: CBlasDouble,
        c: *mut CBlasDouble,
        ldc: MklCBlasInt64,
    );

    /// The ?symm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product,
    /// where one of the matrices is symmetric. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot B + \beta \cdot C \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot B \cdot A + \beta \cdot C \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ and $\beta$ are complex scalars, $A$ is a symmetric matrix, and $B$ and $C$ are $m$-by-$n$ matrices.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the symmetric matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the operation.
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used.
    /// * `m` - Specifies the number of rows of matrix $C$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $C$. Must be at least zero.
    /// * `alpha` - Specifies the complex scalar $\alpha$.
    /// * `a` - Array containing the symmetric matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array containing the matrix $B$.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the complex scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, m)$.
    pub fn cblas_csymm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?symm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product,
    /// where one of the matrices is symmetric. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot B + \beta \cdot C \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot B \cdot A + \beta \cdot C \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ and $\beta$ are complex double scalars, $A$ is a symmetric matrix, and $B$ and $C$ are $m$-by-$n$ matrices.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the symmetric matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the operation.
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is used.
    /// * `m` - Specifies the number of rows of matrix $C$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $C$. Must be at least zero.
    /// * `alpha` - Specifies the complex double scalar $\alpha$.
    /// * `a` - Array containing the symmetric matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array containing the matrix $B$.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the complex double scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, m)$.
    pub fn cblas_zsymm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?syrk routines perform a rank-k update of a symmetric matrix. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot A^{\top} + \beta \cdot C \quad \text{(if trans = CblasNoTrans)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot A^{\top} \cdot A + \beta \cdot C \quad \text{(if trans = CblasTrans)}$$
    ///
    /// where $\alpha$ and $\beta$ are scalars, $C$ is an $n$-by-$n$ symmetric matrix, and $A$ is an $n$-by-$k$ matrix
    /// (if trans = CblasNoTrans) or $k$-by-$n$ matrix (if trans = CblasTrans).
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $C$ is used.
    /// * `trans` - Specifies the operation:
    ///     * CblasNoTrans - $C := \alpha \cdot A \cdot A^{\top} + \beta \cdot C$
    ///     * CblasTrans - $C := \alpha \cdot A^{\top} \cdot A + \beta \cdot C$
    /// * `n` - Specifies the order of matrix $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of matrix $A$ (if trans = CblasNoTrans) or rows of matrix $A$ (if trans = CblasTrans).
    /// * `alpha` - Specifies the scalar $\alpha$.
    /// * `a` - Array of size lda by ka.
    /// * `lda` - Leading dimension of a.
    /// * `beta` - Specifies the scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains the symmetric matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, n)$.
    pub fn cblas_ssyrk_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        beta: CBlasFloat,
        c: *mut CBlasFloat,
        ldc: MklCBlasInt64,
    );

    /// The ?syrk routines perform a rank-k update of a symmetric matrix. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot A^{\top} + \beta \cdot C \quad \text{(if trans = CblasNoTrans)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot A^{\top} \cdot A + \beta \cdot C \quad \text{(if trans = CblasTrans)}$$
    ///
    /// where $\alpha$ and $\beta$ are scalars, $C$ is an $n$-by-$n$ symmetric matrix, and $A$ is an $n$-by-$k$ matrix
    /// (if trans = CblasNoTrans) or $k$-by-$n$ matrix (if trans = CblasTrans).
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $C$ is used.
    /// * `trans` - Specifies the operation:
    ///     * CblasNoTrans - $C := \alpha \cdot A \cdot A^{\top} + \beta \cdot C$
    ///     * CblasTrans - $C := \alpha \cdot A^{\top} \cdot A + \beta \cdot C$
    /// * `n` - Specifies the order of matrix $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of matrix $A$ (if trans = CblasNoTrans) or rows of matrix $A$ (if trans = CblasTrans).
    /// * `alpha` - Specifies the scalar $\alpha$.
    /// * `a` - Array of size lda by ka.
    /// * `lda` - Leading dimension of a.
    /// * `beta` - Specifies the scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains the symmetric matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, n)$.
    pub fn cblas_dsyrk_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        beta: CBlasDouble,
        c: *mut CBlasDouble,
        ldc: MklCBlasInt64,
    );

    /// The ?syrk routines perform a rank-k update of a symmetric matrix. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot A^{\top} + \beta \cdot C \quad \text{(if trans = CblasNoTrans)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot A^{\top} \cdot A + \beta \cdot C \quad \text{(if trans = CblasTrans)}$$
    ///
    /// where $\alpha$ and $\beta$ are complex scalars, $C$ is an $n$-by-$n$ symmetric matrix, and $A$ is an $n$-by-$k$ matrix
    /// (if trans = CblasNoTrans) or $k$-by-$n$ matrix (if trans = CblasTrans).
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $C$ is used.
    /// * `trans` - Specifies the operation.
    /// * `n` - Specifies the order of matrix $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of matrix $A$ (if trans = CblasNoTrans) or rows of matrix $A$ (if trans = CblasTrans).
    /// * `alpha` - Specifies the complex scalar $\alpha$.
    /// * `a` - Array of size lda by ka.
    /// * `lda` - Leading dimension of a.
    /// * `beta` - Specifies the complex scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains the symmetric matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, n)$.
    pub fn cblas_csyrk_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?syrk routines perform a rank-k update of a symmetric matrix. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot A^{\top} + \beta \cdot C \quad \text{(if trans = CblasNoTrans)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot A^{\top} \cdot A + \beta \cdot C \quad \text{(if trans = CblasTrans)}$$
    ///
    /// where $\alpha$ and $\beta$ are complex double scalars, $C$ is an $n$-by-$n$ symmetric matrix, and $A$ is an $n$-by-$k$ matrix
    /// (if trans = CblasNoTrans) or $k$-by-$n$ matrix (if trans = CblasTrans).
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $C$ is used.
    /// * `trans` - Specifies the operation.
    /// * `n` - Specifies the order of matrix $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of matrix $A$ (if trans = CblasNoTrans) or rows of matrix $A$ (if trans = CblasTrans).
    /// * `alpha` - Specifies the complex double scalar $\alpha$.
    /// * `a` - Array of size lda by ka.
    /// * `lda` - Leading dimension of a.
    /// * `beta` - Specifies the complex double scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains the symmetric matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, n)$.
    pub fn cblas_zsyrk_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?syr2k routines perform a rank-2k update of a symmetric matrix. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot B^{\top} + \alpha \cdot B \cdot A^{\top} + \beta \cdot C \quad \text{(if trans = CblasNoTrans)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot A^{\top} \cdot B + \alpha \cdot B^{\top} \cdot A + \beta \cdot C \quad \text{(if trans = CblasTrans)}$$
    ///
    /// where $\alpha$ and $\beta$ are scalars, $C$ is an $n$-by-$n$ symmetric matrix,
    /// and $A$ and $B$ are $n$-by-$k$ matrices (if trans = CblasNoTrans) or $k$-by-$n$ matrices (if trans = CblasTrans).
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $C$ is used.
    /// * `trans` - Specifies the operation.
    /// * `n` - Specifies the order of matrix $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of matrices $A$ and $B$ (if trans = CblasNoTrans) or rows (if trans = CblasTrans).
    /// * `alpha` - Specifies the scalar $\alpha$.
    /// * `a` - Array of size lda by ka.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by kb.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains the symmetric matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, n)$.
    pub fn cblas_ssyr2k_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        b: *const CBlasFloat,
        ldb: MklCBlasInt64,
        beta: CBlasFloat,
        c: *mut CBlasFloat,
        ldc: MklCBlasInt64,
    );

    /// The ?syr2k routines perform a rank-2k update of a symmetric matrix. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot B^{\top} + \alpha \cdot B \cdot A^{\top} + \beta \cdot C \quad \text{(if trans = CblasNoTrans)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot A^{\top} \cdot B + \alpha \cdot B^{\top} \cdot A + \beta \cdot C \quad \text{(if trans = CblasTrans)}$$
    ///
    /// where $\alpha$ and $\beta$ are scalars, $C$ is an $n$-by-$n$ symmetric matrix,
    /// and $A$ and $B$ are $n$-by-$k$ matrices (if trans = CblasNoTrans) or $k$-by-$n$ matrices (if trans = CblasTrans).
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $C$ is used.
    /// * `trans` - Specifies the operation.
    /// * `n` - Specifies the order of matrix $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of matrices $A$ and $B$ (if trans = CblasNoTrans) or rows (if trans = CblasTrans).
    /// * `alpha` - Specifies the scalar $\alpha$.
    /// * `a` - Array of size lda by ka.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by kb.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains the symmetric matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, n)$.
    pub fn cblas_dsyr2k_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        b: *const CBlasDouble,
        ldb: MklCBlasInt64,
        beta: CBlasDouble,
        c: *mut CBlasDouble,
        ldc: MklCBlasInt64,
    );

    /// The ?syr2k routines perform a rank-2k update of a symmetric matrix. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot B^{\top} + \alpha \cdot B \cdot A^{\top} + \beta \cdot C \quad \text{(if trans = CblasNoTrans)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot A^{\top} \cdot B + \alpha \cdot B^{\top} \cdot A + \beta \cdot C \quad \text{(if trans = CblasTrans)}$$
    ///
    /// where $\alpha$ and $\beta$ are complex scalars, $C$ is an $n$-by-$n$ symmetric matrix,
    /// and $A$ and $B$ are $n$-by-$k$ matrices (if trans = CblasNoTrans) or $k$-by-$n$ matrices (if trans = CblasTrans).
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $C$ is used.
    /// * `trans` - Specifies the operation.
    /// * `n` - Specifies the order of matrix $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of matrices $A$ and $B$ (if trans = CblasNoTrans) or rows (if trans = CblasTrans).
    /// * `alpha` - Specifies the complex scalar $\alpha$.
    /// * `a` - Array of size lda by ka.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by kb.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the complex scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains the symmetric matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, n)$.
    pub fn cblas_csyr2k_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?syr2k routines perform a rank-2k update of a symmetric matrix. The operation is defined as
    ///
    /// $$C := \alpha \cdot A \cdot B^{\top} + \alpha \cdot B \cdot A^{\top} + \beta \cdot C \quad \text{(if trans = CblasNoTrans)}$$
    ///
    /// or
    ///
    /// $$C := \alpha \cdot A^{\top} \cdot B + \alpha \cdot B^{\top} \cdot A + \beta \cdot C \quad \text{(if trans = CblasTrans)}$$
    ///
    /// where $\alpha$ and $\beta$ are complex double scalars, $C$ is an $n$-by-$n$ symmetric matrix,
    /// and $A$ and $B$ are $n$-by-$k$ matrices (if trans = CblasNoTrans) or $k$-by-$n$ matrices (if trans = CblasTrans).
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `uplo` - Specifies whether the upper or lower triangular part of the symmetric matrix $C$ is used.
    /// * `trans` - Specifies the operation.
    /// * `n` - Specifies the order of matrix $C$. Must be at least zero.
    /// * `k` - Specifies the number of columns of matrices $A$ and $B$ (if trans = CblasNoTrans) or rows (if trans = CblasTrans).
    /// * `alpha` - Specifies the complex double scalar $\alpha$.
    /// * `a` - Array of size lda by ka.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by kb.
    /// * `ldb` - Leading dimension of b.
    /// * `beta` - Specifies the complex double scalar $\beta$. When zero, $C$ need not be set on entry.
    /// * `c` - Array of size ldc by n. On entry, contains the symmetric matrix $C$. On exit, overwritten by the result.
    /// * `ldc` - Leading dimension of c. Must be at least $\max(1, n)$.
    pub fn cblas_zsyr2k_64(
        layout: CBlasLayout,
        uplo: CBlasUplo,
        trans: CBlasTranspose,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    /// The ?trmm routines compute a scalar-matrix product where one of the matrices is triangular.
    /// The operation is defined as
    ///
    /// $$B := \alpha \cdot \mathrm{op}(A) \cdot B \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$B := \alpha \cdot B \cdot \mathrm{op}(A) \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ is a scalar, $B$ is an $m$-by-$n$ matrix, and $A$ is a unit or non-unit,
    /// upper or lower triangular matrix. $\mathrm{op}(A)$ is one of $A$, $A^{\top}$, or $A^{\mathrm{H}}$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the triangular matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the operation.
    /// * `uplo` - Specifies whether the triangular matrix $A$ is upper (CblasUpper) or lower (CblasLower) triangular.
    /// * `transa` - Specifies the form of $\mathrm{op}(A)$:
    ///     * CblasNoTrans - $\mathrm{op}(A) = A$
    ///     * CblasTrans - $\mathrm{op}(A) = A^{\top}$
    ///     * CblasConjTrans - $\mathrm{op}(A) = A^{\mathrm{H}}$
    /// * `diag` - Specifies whether the triangular matrix $A$ is unit triangular:
    ///     * CblasUnit - $A$ is unit triangular (diagonal elements are assumed to be 1).
    ///     * CblasNonUnit - $A$ is not unit triangular.
    /// * `m` - Specifies the number of rows of matrix $B$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $B$. Must be at least zero.
    /// * `alpha` - Specifies the scalar $\alpha$.
    /// * `a` - Array containing the triangular matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by n. On entry, contains matrix $B$. On exit, overwritten by the result.
    /// * `ldb` - Leading dimension of b. Must be at least $\max(1, m)$.
    pub fn cblas_strmm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        b: *mut CBlasFloat,
        ldb: MklCBlasInt64,
    );

    /// The ?trmm routines compute a scalar-matrix product where one of the matrices is triangular.
    /// The operation is defined as
    ///
    /// $$B := \alpha \cdot \mathrm{op}(A) \cdot B \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$B := \alpha \cdot B \cdot \mathrm{op}(A) \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ is a scalar, $B$ is an $m$-by-$n$ matrix, and $A$ is a unit or non-unit,
    /// upper or lower triangular matrix. $\mathrm{op}(A)$ is one of $A$, $A^{\top}$, or $A^{\mathrm{H}}$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the triangular matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the operation.
    /// * `uplo` - Specifies whether the triangular matrix $A$ is upper (CblasUpper) or lower (CblasLower) triangular.
    /// * `transa` - Specifies the form of $\mathrm{op}(A)$.
    /// * `diag` - Specifies whether the triangular matrix $A$ is unit triangular.
    /// * `m` - Specifies the number of rows of matrix $B$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $B$. Must be at least zero.
    /// * `alpha` - Specifies the scalar $\alpha$.
    /// * `a` - Array containing the triangular matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by n. On entry, contains matrix $B$. On exit, overwritten by the result.
    /// * `ldb` - Leading dimension of b. Must be at least $\max(1, m)$.
    pub fn cblas_dtrmm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        b: *mut CBlasDouble,
        ldb: MklCBlasInt64,
    );

    /// The ?trmm routines compute a scalar-matrix product where one of the matrices is triangular.
    /// The operation is defined as
    ///
    /// $$B := \alpha \cdot \mathrm{op}(A) \cdot B \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$B := \alpha \cdot B \cdot \mathrm{op}(A) \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ is a complex scalar, $B$ is an $m$-by-$n$ matrix, and $A$ is a unit or non-unit,
    /// upper or lower triangular matrix. $\mathrm{op}(A)$ is one of $A$, $A^{\top}$, or $A^{\mathrm{H}}$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the triangular matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the operation.
    /// * `uplo` - Specifies whether the triangular matrix $A$ is upper (CblasUpper) or lower (CblasLower) triangular.
    /// * `transa` - Specifies the form of $\mathrm{op}(A)$.
    /// * `diag` - Specifies whether the triangular matrix $A$ is unit triangular.
    /// * `m` - Specifies the number of rows of matrix $B$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $B$. Must be at least zero.
    /// * `alpha` - Specifies the complex scalar $\alpha$.
    /// * `a` - Array containing the triangular matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by n. On entry, contains matrix $B$. On exit, overwritten by the result.
    /// * `ldb` - Leading dimension of b. Must be at least $\max(1, m)$.
    pub fn cblas_ctrmm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *mut CBlasVoid,
        ldb: MklCBlasInt64,
    );

    /// The ?trmm routines compute a scalar-matrix product where one of the matrices is triangular.
    /// The operation is defined as
    ///
    /// $$B := \alpha \cdot \mathrm{op}(A) \cdot B \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$B := \alpha \cdot B \cdot \mathrm{op}(A) \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ is a complex double scalar, $B$ is an $m$-by-$n$ matrix, and $A$ is a unit or non-unit,
    /// upper or lower triangular matrix. $\mathrm{op}(A)$ is one of $A$, $A^{\top}$, or $A^{\mathrm{H}}$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the triangular matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the operation.
    /// * `uplo` - Specifies whether the triangular matrix $A$ is upper (CblasUpper) or lower (CblasLower) triangular.
    /// * `transa` - Specifies the form of $\mathrm{op}(A)$.
    /// * `diag` - Specifies whether the triangular matrix $A$ is unit triangular.
    /// * `m` - Specifies the number of rows of matrix $B$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $B$. Must be at least zero.
    /// * `alpha` - Specifies the complex double scalar $\alpha$.
    /// * `a` - Array containing the triangular matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by n. On entry, contains matrix $B$. On exit, overwritten by the result.
    /// * `ldb` - Leading dimension of b. Must be at least $\max(1, m)$.
    pub fn cblas_ztrmm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *mut CBlasVoid,
        ldb: MklCBlasInt64,
    );

    /// The ?trsm routines solve a triangular matrix equation. The operation is defined as
    ///
    /// $$\mathrm{op}(A) \cdot X = \alpha \cdot B \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$X \cdot \mathrm{op}(A) = \alpha \cdot B \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ is a scalar, $X$ and $B$ are $m$-by-$n$ matrices, and $A$ is a unit or non-unit,
    /// upper or lower triangular matrix. $\mathrm{op}(A)$ is one of $A$, $A^{\top}$, or $A^{\mathrm{H}}$.
    /// The matrix $B$ is overwritten by the solution matrix $X$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the triangular matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the equation.
    /// * `uplo` - Specifies whether the triangular matrix $A$ is upper (CblasUpper) or lower (CblasLower) triangular.
    /// * `transa` - Specifies the form of $\mathrm{op}(A)$:
    ///     * CblasNoTrans - $\mathrm{op}(A) = A$
    ///     * CblasTrans - $\mathrm{op}(A) = A^{\top}$
    ///     * CblasConjTrans - $\mathrm{op}(A) = A^{\mathrm{H}}$
    /// * `diag` - Specifies whether the triangular matrix $A$ is unit triangular:
    ///     * CblasUnit - $A$ is unit triangular (diagonal elements are assumed to be 1).
    ///     * CblasNonUnit - $A$ is not unit triangular.
    /// * `m` - Specifies the number of rows of matrix $B$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $B$. Must be at least zero.
    /// * `alpha` - Specifies the scalar $\alpha$.
    /// * `a` - Array containing the triangular matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by n. On entry, contains the right-hand side matrix $B$. On exit, overwritten by the solution matrix $X$.
    /// * `ldb` - Leading dimension of b. Must be at least $\max(1, m)$.
    pub fn cblas_strsm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        b: *mut CBlasFloat,
        ldb: MklCBlasInt64,
    );

    /// The ?trsm routines solve a triangular matrix equation. The operation is defined as
    ///
    /// $$\mathrm{op}(A) \cdot X = \alpha \cdot B \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$X \cdot \mathrm{op}(A) = \alpha \cdot B \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ is a scalar, $X$ and $B$ are $m$-by-$n$ matrices, and $A$ is a unit or non-unit,
    /// upper or lower triangular matrix. $\mathrm{op}(A)$ is one of $A$, $A^{\top}$, or $A^{\mathrm{H}}$.
    /// The matrix $B$ is overwritten by the solution matrix $X$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the triangular matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the equation.
    /// * `uplo` - Specifies whether the triangular matrix $A$ is upper (CblasUpper) or lower (CblasLower) triangular.
    /// * `transa` - Specifies the form of $\mathrm{op}(A)$.
    /// * `diag` - Specifies whether the triangular matrix $A$ is unit triangular.
    /// * `m` - Specifies the number of rows of matrix $B$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $B$. Must be at least zero.
    /// * `alpha` - Specifies the scalar $\alpha$.
    /// * `a` - Array containing the triangular matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by n. On entry, contains the right-hand side matrix $B$. On exit, overwritten by the solution matrix $X$.
    /// * `ldb` - Leading dimension of b. Must be at least $\max(1, m)$.
    pub fn cblas_dtrsm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        b: *mut CBlasDouble,
        ldb: MklCBlasInt64,
    );

    /// The ?trsm routines solve a triangular matrix equation. The operation is defined as
    ///
    /// $$\mathrm{op}(A) \cdot X = \alpha \cdot B \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$X \cdot \mathrm{op}(A) = \alpha \cdot B \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ is a complex scalar, $X$ and $B$ are $m$-by-$n$ matrices, and $A$ is a unit or non-unit,
    /// upper or lower triangular matrix. $\mathrm{op}(A)$ is one of $A$, $A^{\top}$, or $A^{\mathrm{H}}$.
    /// The matrix $B$ is overwritten by the solution matrix $X$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the triangular matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the equation.
    /// * `uplo` - Specifies whether the triangular matrix $A$ is upper (CblasUpper) or lower (CblasLower) triangular.
    /// * `transa` - Specifies the form of $\mathrm{op}(A)$.
    /// * `diag` - Specifies whether the triangular matrix $A$ is unit triangular.
    /// * `m` - Specifies the number of rows of matrix $B$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $B$. Must be at least zero.
    /// * `alpha` - Specifies the complex scalar $\alpha$.
    /// * `a` - Array containing the triangular matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by n. On entry, contains the right-hand side matrix $B$. On exit, overwritten by the solution matrix $X$.
    /// * `ldb` - Leading dimension of b. Must be at least $\max(1, m)$.
    pub fn cblas_ctrsm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *mut CBlasVoid,
        ldb: MklCBlasInt64,
    );

    /// The ?trsm routines solve a triangular matrix equation. The operation is defined as
    ///
    /// $$\mathrm{op}(A) \cdot X = \alpha \cdot B \quad \text{(if side = CblasLeft)}$$
    ///
    /// or
    ///
    /// $$X \cdot \mathrm{op}(A) = \alpha \cdot B \quad \text{(if side = CblasRight)}$$
    ///
    /// where $\alpha$ is a complex double scalar, $X$ and $B$ are $m$-by-$n$ matrices, and $A$ is a unit or non-unit,
    /// upper or lower triangular matrix. $\mathrm{op}(A)$ is one of $A$, $A^{\top}$, or $A^{\mathrm{H}}$.
    /// The matrix $B$ is overwritten by the solution matrix $X$.
    ///
    /// # Arguments
    /// * `layout` - Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    /// * `side` - Specifies whether the triangular matrix $A$ appears on the left (CblasLeft) or right (CblasRight) in the equation.
    /// * `uplo` - Specifies whether the triangular matrix $A$ is upper (CblasUpper) or lower (CblasLower) triangular.
    /// * `transa` - Specifies the form of $\mathrm{op}(A)$.
    /// * `diag` - Specifies whether the triangular matrix $A$ is unit triangular.
    /// * `m` - Specifies the number of rows of matrix $B$. Must be at least zero.
    /// * `n` - Specifies the number of columns of matrix $B$. Must be at least zero.
    /// * `alpha` - Specifies the complex double scalar $\alpha$.
    /// * `a` - Array containing the triangular matrix $A$.
    /// * `lda` - Leading dimension of a.
    /// * `b` - Array of size ldb by n. On entry, contains the right-hand side matrix $B$. On exit, overwritten by the solution matrix $X$.
    /// * `ldb` - Leading dimension of b. Must be at least $\max(1, m)$.
    pub fn cblas_ztrsm_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *mut CBlasVoid,
        ldb: MklCBlasInt64,
    );

    pub fn cblas_hgemm_64(
        layout: CBlasLayout,
        transa: CBlasTranspose,
        transb: CBlasTranspose,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        k: MklCBlasInt64,
        alpha: CBlasF16,
        a: *const CBlasF16,
        lda: MklCBlasInt64,
        b: *const CBlasF16,
        ldb: MklCBlasInt64,
        beta: CBlasF16,
        c: *mut CBlasF16,
        ldc: MklCBlasInt64,
    );

    pub fn cblas_strmm_oop_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        b: *const CBlasFloat,
        ldb: MklCBlasInt64,
        beta: CBlasFloat,
        c: *mut CBlasFloat,
        ldc: MklCBlasInt64,
    );
    pub fn cblas_dtrmm_oop_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        b: *const CBlasDouble,
        ldb: MklCBlasInt64,
        beta: CBlasDouble,
        c: *mut CBlasDouble,
        ldc: MklCBlasInt64,
    );
    pub fn cblas_ctrmm_oop_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );
    pub fn cblas_ztrmm_oop_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

    pub fn cblas_strsm_oop_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasFloat,
        a: *const CBlasFloat,
        lda: MklCBlasInt64,
        b: *const CBlasFloat,
        ldb: MklCBlasInt64,
        beta: CBlasFloat,
        c: *mut CBlasFloat,
        ldc: MklCBlasInt64,
    );
    pub fn cblas_dtrsm_oop_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: CBlasDouble,
        a: *const CBlasDouble,
        lda: MklCBlasInt64,
        b: *const CBlasDouble,
        ldb: MklCBlasInt64,
        beta: CBlasDouble,
        c: *mut CBlasDouble,
        ldc: MklCBlasInt64,
    );
    pub fn cblas_ctrsm_oop_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );
    pub fn cblas_ztrsm_oop_64(
        layout: CBlasLayout,
        side: CBlasSide,
        uplo: CBlasUplo,
        transa: CBlasTranspose,
        diag: CBlasDiag,
        m: MklCBlasInt64,
        n: MklCBlasInt64,
        alpha: *const CBlasVoid,
        a: *const CBlasVoid,
        lda: MklCBlasInt64,
        b: *const CBlasVoid,
        ldb: MklCBlasInt64,
        beta: *const CBlasVoid,
        c: *mut CBlasVoid,
        ldc: MklCBlasInt64,
    );

}
