//! Explicit Intel MKL `*_64` Level 1 CBLAS entry points.
//!
//! Every declaration in this module binds an actual suffixed MKL symbol and
//! uses `MKL_INT64`/`MKL_UINT64`, independent of the ordinary ABI feature.
//!
//! # Example
//!
//! ```no_run
//! use rivet_blas_sys::cblas::cblas_level_one_mkl_64::cblas_ddot_64;
//! use rivet_blas_sys::cblas::cblas_types::MklCBlasInt64;
//!
//! let x = [1.0_f64, 2.0, 3.0];
//! let y = [4.0_f64, 5.0, 6.0];
//! let dot = unsafe {
//!     cblas_ddot_64(3 as MklCBlasInt64, x.as_ptr(), 1, y.as_ptr(), 1)
//! };
//! assert_eq!(dot, 32.0);
//! ```

use crate::cblas::cblas_types::*;

unsafe extern "C" {

    /// The ?asum routine computes the sum of the magnitudes of elements of a real vector, or the sum of magnitudes of the real and imaginary parts of elements of a complex vector:
    ///
    /// $$\mathrm{res} =  | \mathrm{Re} x_1| + |\mathrm{Im} x_1| + | \mathrm{Re} x_2| + |\mathrm{Im} x_2|+ \cdots + | \mathrm{Re} x_n| + |\mathrm{Im} x_n|$$
    ///
    /// where $x$ is a vector with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    ///
    /// # Returns
    ///
    /// Contains the sum of magnitudes of real and imaginary parts of all elements of the vector.
    pub fn cblas_sasum_64(
        n: MklCBlasInt64,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
    ) -> CBlasFloat;

    /// The ?asum routine computes the sum of the magnitudes of elements of a real vector, or the sum of magnitudes of the real and imaginary parts of elements of a complex vector:
    ///
    /// $$\mathrm{res} =  | \mathrm{Re} x_1| + |\mathrm{Im} x_1| + | \mathrm{Re} x_2| + |\mathrm{Im} x_2|+ \cdots + | \mathrm{Re} x_n| + |\mathrm{Im} x_n|$$
    ///
    /// where $x$ is a vector with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    ///
    /// # Returns
    ///
    /// Contains the sum of magnitudes of real and imaginary parts of all elements of the vector.
    pub fn cblas_scasum_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
    ) -> CBlasFloat;

    /// The ?asum routine computes the sum of the magnitudes of elements of a real vector, or the sum of magnitudes of the real and imaginary parts of elements of a complex vector:
    ///
    /// $$\mathrm{res} =  | \mathrm{Re} x_1| + |\mathrm{Im} x_1| + | \mathrm{Re} x_2| + |\mathrm{Im} x_2|+ \cdots + | \mathrm{Re} x_n| + |\mathrm{Im} x_n|$$
    ///
    /// where $x$ is a vector with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    ///
    /// # Returns
    ///
    /// Contains the sum of magnitudes of real and imaginary parts of all elements of the vector.
    pub fn cblas_dasum_64(
        n: MklCBlasInt64,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
    ) -> CBlasDouble;

    /// The ?asum routine computes the sum of the magnitudes of elements of a real vector, or the sum of magnitudes of the real and imaginary parts of elements of a complex vector:
    ///
    /// $$\mathrm{res} =  | \mathrm{Re} x_1| + |\mathrm{Im} x_1| + | \mathrm{Re} x_2| + |\mathrm{Im} x_2|+ \cdots + | \mathrm{Re} x_n| + |\mathrm{Im} x_n|$$
    ///
    /// where $x$ is a vector with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    ///
    /// # Returns
    ///
    /// Contains the sum of magnitudes of real and imaginary parts of all elements of the vector.
    pub fn cblas_dzasum_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
    ) -> CBlasDouble;

    /// The ?axpy routine performs a vector-vector operation defined as
    ///
    /// $$y := a \cdot x + y$$
    ///
    /// where $a$ is a scalar, $x$ and $y$ are vectors each with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `a` - Specifies the scalar a.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    pub fn cblas_saxpy_64(
        n: MklCBlasInt64,
        a: CBlasFloat,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
        y: *mut CBlasFloat,
        incy: MklCBlasInt64,
    );

    /// The ?axpy routine performs a vector-vector operation defined as
    ///
    /// $$y := a \cdot x + y$$
    ///
    /// where $a$ is a scalar, $x$ and $y$ are vectors each with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `a` - Specifies the scalar a.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    pub fn cblas_daxpy_64(
        n: MklCBlasInt64,
        a: CBlasDouble,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
        y: *mut CBlasDouble,
        incy: MklCBlasInt64,
    );

    /// The ?axpy routine performs a vector-vector operation defined as
    ///
    /// $$y := a \cdot x + y$$
    ///
    /// where $a$ is a scalar, $x$ and $y$ are vectors each with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `a` - Specifies the scalar a (complex).
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    pub fn cblas_caxpy_64(
        n: MklCBlasInt64,
        a: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?axpy routine performs a vector-vector operation defined as
    ///
    /// $$y := a \cdot x + y$$
    ///
    /// where $a$ is a scalar, $x$ and $y$ are vectors each with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `a` - Specifies the scalar a (complex double).
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    pub fn cblas_zaxpy_64(
        n: MklCBlasInt64,
        a: *const CBlasVoid,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?copy routine performs a vector-vector operation defined as
    ///
    /// $$y := x$$
    ///
    /// where $x$ and $y$ are vectors of $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    pub fn cblas_scopy_64(
        n: MklCBlasInt64,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
        y: *mut CBlasFloat,
        incy: MklCBlasInt64,
    );

    /// The ?copy routine performs a vector-vector operation defined as
    ///
    /// $$y := x$$
    ///
    /// where $x$ and $y$ are vectors of $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    pub fn cblas_dcopy_64(
        n: MklCBlasInt64,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
        y: *mut CBlasDouble,
        incy: MklCBlasInt64,
    );

    /// The ?copy routine performs a vector-vector operation defined as
    ///
    /// $$y := x$$
    ///
    /// where $x$ and $y$ are vectors of $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    pub fn cblas_ccopy_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?copy routine performs a vector-vector operation defined as
    ///
    /// $$y := x$$
    ///
    /// where $x$ and $y$ are vectors of $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    pub fn cblas_zcopy_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?dot routine performs a vector-vector reduction operation defined as
    ///
    /// $$\mathrm{res} = \sum_{i=1}^{n} x_i \cdot y_i$$
    ///
    /// where $x$ and $y$ are vectors of $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    ///
    /// # Returns
    /// Returns the dot product of vectors x and y.
    pub fn cblas_sdot_64(
        n: MklCBlasInt64,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
        y: *const CBlasFloat,
        incy: MklCBlasInt64,
    ) -> CBlasFloat;

    /// The ?dot routine performs a vector-vector reduction operation defined as
    ///
    /// $$\mathrm{res} = \sum_{i=1}^{n} x_i \cdot y_i$$
    ///
    /// where $x$ and $y$ are vectors of $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    ///
    /// # Returns
    /// Returns the dot product of vectors x and y.
    pub fn cblas_ddot_64(
        n: MklCBlasInt64,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
        y: *const CBlasDouble,
        incy: MklCBlasInt64,
    ) -> CBlasDouble;

    /// The sdsdot routine performs a vector-vector operation defined as
    ///
    /// $$\mathrm{res} = sb + \sum_{i=1}^{n} sx_i \cdot sy_i$$
    ///
    /// where $sb$ is a scalar, $sx$ and $sy$ are single-precision vectors with $n$ elements.
    /// The computation is performed in double precision.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors sx and sy.
    /// * `sb` - Specifies the scalar sb to be added to the dot product.
    /// * `sx` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector sx.
    /// * `sy` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector sy.
    ///
    /// # Returns
    /// Returns the result of sb plus the dot product of sx and sy.
    pub fn cblas_sdsdot_64(
        n: MklCBlasInt64,
        sb: CBlasFloat,
        sx: *const CBlasFloat,
        incx: MklCBlasInt64,
        sy: *const CBlasFloat,
        incy: MklCBlasInt64,
    ) -> CBlasFloat;

    /// The dsdot routine performs a vector-vector operation defined as
    ///
    /// $$\mathrm{res} = \sum_{i=1}^{n} sx_i \cdot sy_i$$
    ///
    /// where $sx$ and $sy$ are single-precision vectors with $n$ elements.
    /// The computation is performed in double precision.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors sx and sy.
    /// * `sx` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector sx.
    /// * `sy` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector sy.
    ///
    /// # Returns
    /// Returns the dot product of vectors sx and sy in double precision.
    pub fn cblas_dsdot_64(
        n: MklCBlasInt64,
        sx: *const CBlasFloat,
        incx: MklCBlasInt64,
        sy: *const CBlasFloat,
        incy: MklCBlasInt64,
    ) -> CBlasDouble;

    /// The ?dotc routine performs a vector-vector operation defined as
    ///
    /// $$\mathrm{res} = \sum_{i=1}^{n} \overline{x_i} \cdot y_i$$
    ///
    /// where $x$ and $y$ are complex vectors of $n$ elements. The conjugate of $x$ is used.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    /// * `dotc` - Contains the result of the conjugate dot product.
    pub fn cblas_cdotc_sub_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *const CBlasVoid,
        incy: MklCBlasInt64,
        dotc: *mut CBlasVoid,
    );

    /// The ?dotc routine performs a vector-vector operation defined as
    ///
    /// $$\mathrm{res} = \sum_{i=1}^{n} \overline{x_i} \cdot y_i$$
    ///
    /// where $x$ and $y$ are complex vectors of $n$ elements. The conjugate of $x$ is used.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    /// * `dotc` - Contains the result of the conjugate dot product.
    pub fn cblas_zdotc_sub_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *const CBlasVoid,
        incy: MklCBlasInt64,
        dotc: *mut CBlasVoid,
    );

    /// The ?dotu routine performs a vector-vector operation defined as
    ///
    /// $$\mathrm{res} = \sum_{i=1}^{n} x_i \cdot y_i$$
    ///
    /// where $x$ and $y$ are complex vectors of $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    /// * `dotu` - Contains the result of the dot product.
    pub fn cblas_cdotu_sub_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *const CBlasVoid,
        incy: MklCBlasInt64,
        dotu: *mut CBlasVoid,
    );

    /// The ?dotu routine performs a vector-vector operation defined as
    ///
    /// $$\mathrm{res} = \sum_{i=1}^{n} x_i \cdot y_i$$
    ///
    /// where $x$ and $y$ are complex vectors of $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)).
    /// * `incy` - Specifies the increment for indexing vector y.
    /// * `dotu` - Contains the result of the dot product.
    pub fn cblas_zdotu_sub_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
        y: *const CBlasVoid,
        incy: MklCBlasInt64,
        dotu: *mut CBlasVoid,
    );

    /// The ?nrm2 routine performs a vector reduction operation defined as
    ///
    /// $$\mathrm{res} = \|x\|_2 = \sqrt{\sum_{i=1}^{n} |x_i|^2}$$
    ///
    /// where $x$ is a vector with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    ///
    /// # Returns
    /// Returns the Euclidean norm of vector x.
    pub fn cblas_snrm2_64(
        n: MklCBlasInt64,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
    ) -> CBlasFloat;

    /// The ?nrm2 routine performs a vector reduction operation defined as
    ///
    /// $$\mathrm{res} = \|x\|_2 = \sqrt{\sum_{i=1}^{n} |x_i|^2}$$
    ///
    /// where $x$ is a vector with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    ///
    /// # Returns
    /// Returns the Euclidean norm of vector x.
    pub fn cblas_dnrm2_64(
        n: MklCBlasInt64,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
    ) -> CBlasDouble;

    /// The ?nrm2 routine performs a vector reduction operation defined as
    ///
    /// $$\mathrm{res} = \|x\|_2 = \sqrt{\sum_{i=1}^{n} |x_i|^2}$$
    ///
    /// where $x$ is a complex vector with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    ///
    /// # Returns
    /// Returns the Euclidean norm of complex vector x.
    pub fn cblas_scnrm2_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
    ) -> CBlasFloat;

    /// The ?nrm2 routine performs a vector reduction operation defined as
    ///
    /// $$\mathrm{res} = \|x\|_2 = \sqrt{\sum_{i=1}^{n} |x_i|^2}$$
    ///
    /// where $x$ is a complex double vector with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    ///
    /// # Returns
    /// Returns the Euclidean norm of complex vector x.
    pub fn cblas_dznrm2_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
    ) -> CBlasDouble;

    /// The ?rot routine performs a vector-vector operation defined as
    ///
    /// $$\begin{pmatrix} x_i \\ y_i \end{pmatrix} := \begin{pmatrix} c & s \\ -s & c \end{pmatrix} \begin{pmatrix} x_i \\ y_i \end{pmatrix}$$
    ///
    /// where $c$ and $s$ form a plane rotation matrix. For complex versions, the operation uses appropriate complex arithmetic.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). On entry, contains the vector x. On exit, overwritten by the rotated vector.
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)). On entry, contains the vector y. On exit, overwritten by the rotated vector.
    /// * `incy` - Specifies the increment for indexing vector y.
    /// * `c` - Specifies the scalar c (cosine of rotation).
    /// * `s` - Specifies the scalar s (sine of rotation).
    pub fn cblas_srot_64(
        n: MklCBlasInt64,
        x: *mut CBlasFloat,
        incx: MklCBlasInt64,
        y: *mut CBlasFloat,
        incy: MklCBlasInt64,
        c: CBlasFloat,
        s: CBlasFloat,
    );

    /// The ?rot routine performs a vector-vector operation defined as
    ///
    /// $$\begin{pmatrix} x_i \\ y_i \end{pmatrix} := \begin{pmatrix} c & s \\ -s & c \end{pmatrix} \begin{pmatrix} x_i \\ y_i \end{pmatrix}$$
    ///
    /// where $c$ and $s$ form a plane rotation matrix. For complex versions, the operation uses appropriate complex arithmetic.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). On entry, contains the vector x. On exit, overwritten by the rotated vector.
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)). On entry, contains the vector y. On exit, overwritten by the rotated vector.
    /// * `incy` - Specifies the increment for indexing vector y.
    /// * `c` - Specifies the scalar c (cosine of rotation).
    /// * `s` - Specifies the scalar s (sine of rotation).
    pub fn cblas_drot_64(
        n: MklCBlasInt64,
        x: *mut CBlasDouble,
        incx: MklCBlasInt64,
        y: *mut CBlasDouble,
        incy: MklCBlasInt64,
        c: CBlasDouble,
        s: CBlasDouble,
    );

    /// The csrot routine performs a vector-vector operation defined as
    ///
    /// $$\begin{pmatrix} x_i \\ y_i \end{pmatrix} := \begin{pmatrix} c & s \\ -s & c \end{pmatrix} \begin{pmatrix} x_i \\ y_i \end{pmatrix}$$
    ///
    /// where $c$ and $s$ are real scalars, and $x$ and $y$ are complex single-precision vectors.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). Complex single-precision vector.
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)). Complex single-precision vector.
    /// * `incy` - Specifies the increment for indexing vector y.
    /// * `c` - Specifies the real scalar c (cosine of rotation).
    /// * `s` - Specifies the real scalar s (sine of rotation).
    pub fn cblas_csrot_64(
        n: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
        c: CBlasFloat,
        s: CBlasFloat,
    );

    /// The zdrot routine performs a vector-vector operation defined as
    ///
    /// $$\begin{pmatrix} x_i \\ y_i \end{pmatrix} := \begin{pmatrix} c & s \\ -s & c \end{pmatrix} \begin{pmatrix} x_i \\ y_i \end{pmatrix}$$
    ///
    /// where $c$ and $s$ are real scalars, and $x$ and $y$ are complex double-precision vectors.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). Complex double-precision vector.
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)). Complex double-precision vector.
    /// * `incy` - Specifies the increment for indexing vector y.
    /// * `c` - Specifies the real scalar c (cosine of rotation).
    /// * `s` - Specifies the real scalar s (sine of rotation).
    pub fn cblas_zdrot_64(
        n: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
        c: CBlasDouble,
        s: CBlasDouble,
    );

    /// The ?rotm routine performs a modified Givens rotation of a pair of vectors.
    /// The operation is defined as:
    ///
    /// $$\begin{pmatrix} x_i \\ y_i \end{pmatrix} := H \begin{pmatrix} x_i \\ y_i \end{pmatrix}$$
    ///
    /// where $H$ is a modified Givens transformation matrix defined by the param array.
    /// The param array contains $h_{11}, h_{21}, h_{12}, h_{22}$ and a flag defining the form of $H$.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). On exit, overwritten by the rotated vector.
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)). On exit, overwritten by the rotated vector.
    /// * `incy` - Specifies the increment for indexing vector y.
    /// * `param` - Array, size 5. Contains the modified Givens rotation parameters: flag, $h_{11}$, $h_{21}$, $h_{12}$, $h_{22}$.
    pub fn cblas_srotm_64(
        n: MklCBlasInt64,
        x: *mut CBlasFloat,
        incx: MklCBlasInt64,
        y: *mut CBlasFloat,
        incy: MklCBlasInt64,
        param: *const CBlasFloat,
    );

    /// The ?rotm routine performs a modified Givens rotation of a pair of vectors.
    /// The operation is defined as:
    ///
    /// $$\begin{pmatrix} x_i \\ y_i \end{pmatrix} := H \begin{pmatrix} x_i \\ y_i \end{pmatrix}$$
    ///
    /// where $H$ is a modified Givens transformation matrix defined by the param array.
    /// The param array contains $h_{11}, h_{21}, h_{12}, h_{22}$ and a flag defining the form of $H$.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). On exit, overwritten by the rotated vector.
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)). On exit, overwritten by the rotated vector.
    /// * `incy` - Specifies the increment for indexing vector y.
    /// * `param` - Array, size 5. Contains the modified Givens rotation parameters: flag, $h_{11}$, $h_{21}$, $h_{12}$, $h_{22}$.
    pub fn cblas_drotm_64(
        n: MklCBlasInt64,
        x: *mut CBlasDouble,
        incx: MklCBlasInt64,
        y: *mut CBlasDouble,
        incy: MklCBlasInt64,
        param: *const CBlasDouble,
    );

    /// The ?scal routine performs a vector operation defined as
    ///
    /// $$x := a \cdot x$$
    ///
    /// where $a$ is a scalar and $x$ is a vector with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `a` - Specifies the scalar a.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). On exit, overwritten by the scaled vector.
    /// * `incx` - Specifies the increment for indexing vector x.
    pub fn cblas_sscal_64(n: MklCBlasInt64, a: CBlasFloat, x: *mut CBlasFloat, incx: MklCBlasInt64);

    /// The ?scal routine performs a vector operation defined as
    ///
    /// $$x := a \cdot x$$
    ///
    /// where $a$ is a scalar and $x$ is a vector with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `a` - Specifies the scalar a.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). On exit, overwritten by the scaled vector.
    /// * `incx` - Specifies the increment for indexing vector x.
    pub fn cblas_dscal_64(
        n: MklCBlasInt64,
        a: CBlasDouble,
        x: *mut CBlasDouble,
        incx: MklCBlasInt64,
    );

    /// The ?scal routine performs a vector operation defined as
    ///
    /// $$x := a \cdot x$$
    ///
    /// where $a$ is a complex scalar and $x$ is a complex vector with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `a` - Specifies the complex scalar a.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). Complex vector. On exit, overwritten by the scaled vector.
    /// * `incx` - Specifies the increment for indexing vector x.
    pub fn cblas_cscal_64(
        n: MklCBlasInt64,
        a: *const CBlasVoid,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The ?scal routine performs a vector operation defined as
    ///
    /// $$x := a \cdot x$$
    ///
    /// where $a$ is a complex scalar and $x$ is a complex double-precision vector with $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `a` - Specifies the complex scalar a.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). Complex double-precision vector. On exit, overwritten by the scaled vector.
    /// * `incx` - Specifies the increment for indexing vector x.
    pub fn cblas_zscal_64(
        n: MklCBlasInt64,
        a: *const CBlasVoid,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The csscal routine performs a vector operation defined as
    ///
    /// $$x := a \cdot x$$
    ///
    /// where $a$ is a real scalar and $x$ is a complex single-precision vector with $n$ elements.
    /// This is a mixed-precision version that scales a complex vector by a real scalar.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `a` - Specifies the real scalar a.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). Complex vector. On exit, overwritten by the scaled vector.
    /// * `incx` - Specifies the increment for indexing vector x.
    pub fn cblas_csscal_64(n: MklCBlasInt64, a: CBlasFloat, x: *mut CBlasVoid, incx: MklCBlasInt64);

    /// The zdscal routine performs a vector operation defined as
    ///
    /// $$x := a \cdot x$$
    ///
    /// where $a$ is a real scalar and $x$ is a complex double-precision vector with $n$ elements.
    /// This is a mixed-precision version that scales a complex vector by a real scalar.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `a` - Specifies the real scalar a.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). Complex double-precision vector. On exit, overwritten by the scaled vector.
    /// * `incx` - Specifies the increment for indexing vector x.
    pub fn cblas_zdscal_64(
        n: MklCBlasInt64,
        a: CBlasDouble,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
    );

    /// The ?swap routine performs a vector-vector operation defined as
    ///
    /// $$\mathrm{swap}(x, y)$$
    ///
    /// where $x$ and $y$ are vectors of $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). On exit, contains the elements of y.
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)). On exit, contains the elements of x.
    /// * `incy` - Specifies the increment for indexing vector y.
    pub fn cblas_sswap_64(
        n: MklCBlasInt64,
        x: *mut CBlasFloat,
        incx: MklCBlasInt64,
        y: *mut CBlasFloat,
        incy: MklCBlasInt64,
    );

    /// The ?swap routine performs a vector-vector operation defined as
    ///
    /// $$\mathrm{swap}(x, y)$$
    ///
    /// where $x$ and $y$ are vectors of $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). On exit, contains the elements of y.
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)). On exit, contains the elements of x.
    /// * `incy` - Specifies the increment for indexing vector y.
    pub fn cblas_dswap_64(
        n: MklCBlasInt64,
        x: *mut CBlasDouble,
        incx: MklCBlasInt64,
        y: *mut CBlasDouble,
        incy: MklCBlasInt64,
    );

    /// The ?swap routine performs a vector-vector operation defined as
    ///
    /// $$\mathrm{swap}(x, y)$$
    ///
    /// where $x$ and $y$ are complex vectors of $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). Complex vector. On exit, contains the elements of y.
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)). Complex vector. On exit, contains the elements of x.
    /// * `incy` - Specifies the increment for indexing vector y.
    pub fn cblas_cswap_64(
        n: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The ?swap routine performs a vector-vector operation defined as
    ///
    /// $$\mathrm{swap}(x, y)$$
    ///
    /// where $x$ and $y$ are complex double-precision vectors of $n$ elements.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vectors x and y.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)). Complex double-precision vector. On exit, contains the elements of y.
    /// * `incx` - Specifies the increment for indexing vector x.
    /// * `y` - Array, size at least (1 + (n-1)*abs(incy)). Complex double-precision vector. On exit, contains the elements of x.
    /// * `incy` - Specifies the increment for indexing vector y.
    pub fn cblas_zswap_64(
        n: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
    );

    /// The i?amax routine performs a vector reduction operation defined as
    ///
    /// $$\mathrm{result} = \arg\max_{i=1,\ldots,n} |x_i|$$
    ///
    /// where $x$ is a vector with $n$ elements. Returns the index of the element with the largest absolute value.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    ///
    /// # Returns
    /// Returns the index (0-based) of the element with the largest absolute value. If n ≤ 0, returns 0.
    pub fn cblas_isamax_64(
        n: MklCBlasInt64,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
    ) -> MklCBlasIndex64;

    /// The i?amax routine performs a vector reduction operation defined as
    ///
    /// $$\mathrm{result} = \arg\max_{i=1,\ldots,n} |x_i|$$
    ///
    /// where $x$ is a vector with $n$ elements. Returns the index of the element with the largest absolute value.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    ///
    /// # Returns
    /// Returns the index (0-based) of the element with the largest absolute value. If n ≤ 0, returns 0.
    pub fn cblas_idamax_64(
        n: MklCBlasInt64,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
    ) -> MklCBlasIndex64;

    /// The i?amax routine performs a vector reduction operation defined as
    ///
    /// $$\mathrm{result} = \arg\max_{i=1,\ldots,n} |x_i|$$
    ///
    /// where $x$ is a complex vector with $n$ elements. Returns the index of the element with the largest absolute value.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    ///
    /// # Returns
    /// Returns the index (0-based) of the element with the largest absolute value. If n ≤ 0, returns 0.
    pub fn cblas_icamax_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
    ) -> MklCBlasIndex64;

    /// The i?amax routine performs a vector reduction operation defined as
    ///
    /// $$\mathrm{result} = \arg\max_{i=1,\ldots,n} |x_i|$$
    ///
    /// where $x$ is a complex double-precision vector with $n$ elements. Returns the index of the element with the largest absolute value.
    ///
    /// # Arguments
    /// * `n` - Specifies the number of elements in vector x.
    /// * `x` - Array, size at least (1 + (n-1)*abs(incx)).
    /// * `incx` - Specifies the increment for indexing vector x.
    ///
    /// # Returns
    /// Returns the index (0-based) of the element with the largest absolute value. If n ≤ 0, returns 0.
    pub fn cblas_izamax_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
    ) -> MklCBlasIndex64;

    /// 64-bit MKL variant of [`super::cblas_level_one_mkl::cblas_crot`].
    /// Applies a complex single-precision plane rotation in place.
    ///
    /// # Safety
    /// `x`, `y`, and `s` must point to valid, sufficiently sized storage, and
    /// both increments must be non-zero.
    pub fn cblas_crot_64(
        n: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
        c: CBlasFloat,
        s: *const CBlasVoid,
    );

    /// 64-bit MKL variant of [`super::cblas_level_one_mkl::cblas_zrot`].
    /// Applies a complex double-precision plane rotation in place.
    ///
    /// # Safety
    /// `x`, `y`, and `s` must point to valid, sufficiently sized storage, and
    /// both increments must be non-zero.
    pub fn cblas_zrot_64(
        n: MklCBlasInt64,
        x: *mut CBlasVoid,
        incx: MklCBlasInt64,
        y: *mut CBlasVoid,
        incy: MklCBlasInt64,
        c: CBlasDouble,
        s: *const CBlasVoid,
    );

    /// Returns the zero-based index of the smallest real single-precision
    /// element, using MKL's explicit 64-bit integer interface.
    ///
    /// # Safety
    /// `x` must point to at least `n` logically strided elements and `incx`
    /// must be non-zero.
    pub fn cblas_isamin_64(
        n: MklCBlasInt64,
        x: *const CBlasFloat,
        incx: MklCBlasInt64,
    ) -> MklCBlasIndex64;

    /// Returns the zero-based index of the smallest real double-precision
    /// element, using MKL's explicit 64-bit integer interface.
    ///
    /// # Safety
    /// `x` must point to at least `n` logically strided elements and `incx`
    /// must be non-zero.
    pub fn cblas_idamin_64(
        n: MklCBlasInt64,
        x: *const CBlasDouble,
        incx: MklCBlasInt64,
    ) -> MklCBlasIndex64;

    /// Returns the zero-based index of the smallest complex single-precision
    /// element, using MKL's explicit 64-bit integer interface.
    ///
    /// # Safety
    /// `x` must point to valid complex storage for `n` logically strided
    /// elements and `incx` must be non-zero.
    pub fn cblas_icamin_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
    ) -> MklCBlasIndex64;

    /// Returns the zero-based index of the smallest complex double-precision
    /// element, using MKL's explicit 64-bit integer interface.
    ///
    /// # Safety
    /// `x` must point to valid complex storage for `n` logically strided
    /// elements and `incx` must be non-zero.
    pub fn cblas_izamin_64(
        n: MklCBlasInt64,
        x: *const CBlasVoid,
        incx: MklCBlasInt64,
    ) -> MklCBlasIndex64;

}
