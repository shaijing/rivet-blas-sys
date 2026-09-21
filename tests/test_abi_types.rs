use std::mem::size_of;

use rivet_blas_sys::cblas::cblas_types::{CBlasIndex, CBlasInt};

#[test]
fn cblas_index_is_pointer_sized() {
    assert_eq!(size_of::<CBlasIndex>(), size_of::<usize>());
}

#[cfg(any(
    feature = "openblas-dynamic-ilp64",
    feature = "openblas-static-ilp64",
    feature = "flexiblas-dynamic-ilp64",
    feature = "flexiblas-static-ilp64",
    feature = "mkl-dynamic-ilp64-gomp",
    feature = "mkl-dynamic-ilp64-iomp",
    feature = "mkl-dynamic-ilp64-seq",
    feature = "mkl-dynamic-ilp64-tbb",
    feature = "mkl-static-ilp64-gomp",
    feature = "mkl-static-ilp64-iomp",
    feature = "mkl-static-ilp64-seq",
    feature = "mkl-static-ilp64-tbb",
    feature = "mkl-sdl"
))]
#[test]
fn ilp64_uses_64_bit_cblas_integers() {
    assert_eq!(size_of::<CBlasInt>(), 8);
}

#[cfg(any(
    feature = "openblas-dynamic-lp64",
    feature = "openblas-static-lp64",
    feature = "flexiblas-dynamic-lp64",
    feature = "flexiblas-static-lp64",
    feature = "mkl-dynamic-lp64-gomp",
    feature = "mkl-dynamic-lp64-iomp",
    feature = "mkl-dynamic-lp64-seq",
    feature = "mkl-dynamic-lp64-tbb",
    feature = "mkl-static-lp64-gomp",
    feature = "mkl-static-lp64-iomp",
    feature = "mkl-static-lp64-seq",
    feature = "mkl-static-lp64-tbb",
    feature = "netlib-dynamic-lp64",
    feature = "netlib-static-lp64",
    feature = "accelerate"
))]
#[test]
fn lp64_uses_32_bit_cblas_integers() {
    assert_eq!(size_of::<CBlasInt>(), 4);
}
