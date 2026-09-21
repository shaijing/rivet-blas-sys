//! Minimal OpenBLAS static-link smoke test.
//!
//! ```text
//! cargo run --release --no-default-features \
//!     --features openblas-static-ilp64 --example static_openblas
//! ```

#[cfg(any(feature = "openblas-static-ilp64", feature = "openblas-static-lp64"))]
fn main() {
    use rivet_blas_sys::cblas::prelude::*;

    let x = [1.0_f64, 2.0, 3.0];
    let y = [4.0_f64, 5.0, 6.0];
    let dot = unsafe { cblas_ddot(3, x.as_ptr(), 1, y.as_ptr(), 1) };

    assert_eq!(dot, 32.0);
    println!("OpenBLAS static cblas_ddot: {dot}");
}

#[cfg(not(any(feature = "openblas-static-ilp64", feature = "openblas-static-lp64")))]
fn main() {
    eprintln!("This example requires the `openblas-static-*` feature.");
}
