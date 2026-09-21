#[cfg(any(
    docsrs,
    all(
        not(docsrs),
        any(
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
            feature = "mkl-sdl",
        )
    )
))]
pub type BlasInt = i64;
#[cfg(any(all(
    not(docsrs),
    any(
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
        feature = "accelerate",
    )
)))]
pub type BlasInt = i32;

pub type BlasIndex = usize;
pub type BlasFloat = f32;
pub type BlasDouble = f64;
pub type BlasF16 = u16;
