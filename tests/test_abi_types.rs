use std::mem::size_of;

use rivet_blas_sys::cblas::cblas_types::{CBlasIndex, CBlasInt};

#[test]
fn cblas_index_is_pointer_sized() {
    assert_eq!(size_of::<CBlasIndex>(), size_of::<usize>());
}

#[cfg(feature = "ilp64")]
#[test]
fn ilp64_uses_64_bit_cblas_integers() {
    assert_eq!(size_of::<CBlasInt>(), 8);
}

#[cfg(feature = "lp64")]
#[test]
fn lp64_uses_32_bit_cblas_integers() {
    assert_eq!(size_of::<CBlasInt>(), 4);
}
