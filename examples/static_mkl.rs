//! Minimal Intel MKL static-link smoke test.
//!
//! ```text
//! cargo run --release --no-default-features \
//!     --features mkl-static-ilp64-iomp --example static_mkl
//! ```

#[cfg(any(
    feature = "mkl-static-ilp64-gomp",
    feature = "mkl-static-ilp64-iomp",
    feature = "mkl-static-ilp64-seq",
    feature = "mkl-static-ilp64-tbb",
    feature = "mkl-static-lp64-gomp",
    feature = "mkl-static-lp64-iomp",
    feature = "mkl-static-lp64-seq",
    feature = "mkl-static-lp64-tbb",
))]
fn main() {
    use rivet_blas_sys::cblas::prelude::*;

    let x = [1.0_f64, 2.0, 3.0];
    let y = [4.0_f64, 5.0, 6.0];
    let dot = unsafe { cblas_ddot(3, x.as_ptr(), 1, y.as_ptr(), 1) };

    assert_eq!(dot, 32.0);
    println!("Intel MKL static cblas_ddot: {dot}");
}

#[cfg(not(any(
    feature = "mkl-static-ilp64-gomp",
    feature = "mkl-static-ilp64-iomp",
    feature = "mkl-static-ilp64-seq",
    feature = "mkl-static-ilp64-tbb",
    feature = "mkl-static-lp64-gomp",
    feature = "mkl-static-lp64-iomp",
    feature = "mkl-static-lp64-seq",
    feature = "mkl-static-lp64-tbb",
)))]
fn main() {
    eprintln!("This example requires an `mkl-static-*-*` feature.");
}
