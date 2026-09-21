pub mod cblas_level_one;
pub mod cblas_level_three;
pub mod cblas_level_two;
pub mod cblas_types;

#[cfg(any(
    docsrs,
    feature = "mkl-dynamic-ilp64-gomp",
    feature = "mkl-dynamic-ilp64-iomp",
    feature = "mkl-dynamic-ilp64-seq",
    feature = "mkl-dynamic-ilp64-tbb",
    feature = "mkl-dynamic-lp64-gomp",
    feature = "mkl-dynamic-lp64-iomp",
    feature = "mkl-dynamic-lp64-seq",
    feature = "mkl-dynamic-lp64-tbb",
    feature = "mkl-static-ilp64-gomp",
    feature = "mkl-static-ilp64-iomp",
    feature = "mkl-static-ilp64-seq",
    feature = "mkl-static-ilp64-tbb",
    feature = "mkl-static-lp64-gomp",
    feature = "mkl-static-lp64-iomp",
    feature = "mkl-static-lp64-seq",
    feature = "mkl-static-lp64-tbb",
    feature = "mkl-sdl",
))]
pub mod cblas_level_one_mkl;
#[cfg(any(
    docsrs,
    feature = "mkl-dynamic-ilp64-gomp",
    feature = "mkl-dynamic-ilp64-iomp",
    feature = "mkl-dynamic-ilp64-seq",
    feature = "mkl-dynamic-ilp64-tbb",
    feature = "mkl-dynamic-lp64-gomp",
    feature = "mkl-dynamic-lp64-iomp",
    feature = "mkl-dynamic-lp64-seq",
    feature = "mkl-dynamic-lp64-tbb",
    feature = "mkl-static-ilp64-gomp",
    feature = "mkl-static-ilp64-iomp",
    feature = "mkl-static-ilp64-seq",
    feature = "mkl-static-ilp64-tbb",
    feature = "mkl-static-lp64-gomp",
    feature = "mkl-static-lp64-iomp",
    feature = "mkl-static-lp64-seq",
    feature = "mkl-static-lp64-tbb",
    feature = "mkl-sdl",
))]
pub mod cblas_level_one_mkl_64;
#[cfg(any(
    docsrs,
    feature = "mkl-dynamic-ilp64-gomp",
    feature = "mkl-dynamic-ilp64-iomp",
    feature = "mkl-dynamic-ilp64-seq",
    feature = "mkl-dynamic-ilp64-tbb",
    feature = "mkl-dynamic-lp64-gomp",
    feature = "mkl-dynamic-lp64-iomp",
    feature = "mkl-dynamic-lp64-seq",
    feature = "mkl-dynamic-lp64-tbb",
    feature = "mkl-static-ilp64-gomp",
    feature = "mkl-static-ilp64-iomp",
    feature = "mkl-static-ilp64-seq",
    feature = "mkl-static-ilp64-tbb",
    feature = "mkl-static-lp64-gomp",
    feature = "mkl-static-lp64-iomp",
    feature = "mkl-static-lp64-seq",
    feature = "mkl-static-lp64-tbb",
    feature = "mkl-sdl",
))]
pub mod cblas_level_three_mkl;
#[cfg(any(
    docsrs,
    feature = "mkl-dynamic-ilp64-gomp",
    feature = "mkl-dynamic-ilp64-iomp",
    feature = "mkl-dynamic-ilp64-seq",
    feature = "mkl-dynamic-ilp64-tbb",
    feature = "mkl-dynamic-lp64-gomp",
    feature = "mkl-dynamic-lp64-iomp",
    feature = "mkl-dynamic-lp64-seq",
    feature = "mkl-dynamic-lp64-tbb",
    feature = "mkl-static-ilp64-gomp",
    feature = "mkl-static-ilp64-iomp",
    feature = "mkl-static-ilp64-seq",
    feature = "mkl-static-ilp64-tbb",
    feature = "mkl-static-lp64-gomp",
    feature = "mkl-static-lp64-iomp",
    feature = "mkl-static-lp64-seq",
    feature = "mkl-static-lp64-tbb",
    feature = "mkl-sdl",
))]
pub mod cblas_level_three_mkl_64;
#[cfg(any(
    docsrs,
    feature = "mkl-dynamic-ilp64-gomp",
    feature = "mkl-dynamic-ilp64-iomp",
    feature = "mkl-dynamic-ilp64-seq",
    feature = "mkl-dynamic-ilp64-tbb",
    feature = "mkl-dynamic-lp64-gomp",
    feature = "mkl-dynamic-lp64-iomp",
    feature = "mkl-dynamic-lp64-seq",
    feature = "mkl-dynamic-lp64-tbb",
    feature = "mkl-static-ilp64-gomp",
    feature = "mkl-static-ilp64-iomp",
    feature = "mkl-static-ilp64-seq",
    feature = "mkl-static-ilp64-tbb",
    feature = "mkl-static-lp64-gomp",
    feature = "mkl-static-lp64-iomp",
    feature = "mkl-static-lp64-seq",
    feature = "mkl-static-lp64-tbb",
    feature = "mkl-sdl",
))]
pub mod cblas_level_two_mkl_64;

#[cfg(any(
    docsrs,
    feature = "openblas-dynamic-ilp64",
    feature = "openblas-dynamic-lp64",
    feature = "openblas-static-ilp64",
    feature = "openblas-static-lp64",
    feature = "flexiblas-dynamic-ilp64",
    feature = "flexiblas-dynamic-lp64",
    feature = "flexiblas-static-ilp64",
    feature = "flexiblas-static-lp64",
))]
pub mod cblas_level_one_openblas;
#[cfg(any(
    docsrs,
    feature = "openblas-dynamic-ilp64",
    feature = "openblas-dynamic-lp64",
    feature = "openblas-static-ilp64",
    feature = "openblas-static-lp64",
    feature = "flexiblas-dynamic-ilp64",
    feature = "flexiblas-dynamic-lp64",
    feature = "flexiblas-static-ilp64",
    feature = "flexiblas-static-lp64",
))]
pub mod cblas_level_three_openblas;

pub mod prelude {
    pub use crate::cblas::cblas_level_one::*;
    pub use crate::cblas::cblas_level_three::*;
    pub use crate::cblas::cblas_level_two::*;
    pub use crate::cblas::cblas_types::*;

    #[cfg(any(
        feature = "mkl-dynamic-ilp64-gomp",
        feature = "mkl-dynamic-ilp64-iomp",
        feature = "mkl-dynamic-ilp64-seq",
        feature = "mkl-dynamic-ilp64-tbb",
        feature = "mkl-dynamic-lp64-gomp",
        feature = "mkl-dynamic-lp64-iomp",
        feature = "mkl-dynamic-lp64-seq",
        feature = "mkl-dynamic-lp64-tbb",
        feature = "mkl-static-ilp64-gomp",
        feature = "mkl-static-ilp64-iomp",
        feature = "mkl-static-ilp64-seq",
        feature = "mkl-static-ilp64-tbb",
        feature = "mkl-static-lp64-gomp",
        feature = "mkl-static-lp64-iomp",
        feature = "mkl-static-lp64-seq",
        feature = "mkl-static-lp64-tbb",
        feature = "mkl-sdl",
    ))]
    pub use crate::cblas::cblas_level_one_mkl::*;
    #[cfg(any(
        feature = "mkl-dynamic-ilp64-gomp",
        feature = "mkl-dynamic-ilp64-iomp",
        feature = "mkl-dynamic-ilp64-seq",
        feature = "mkl-dynamic-ilp64-tbb",
        feature = "mkl-dynamic-lp64-gomp",
        feature = "mkl-dynamic-lp64-iomp",
        feature = "mkl-dynamic-lp64-seq",
        feature = "mkl-dynamic-lp64-tbb",
        feature = "mkl-static-ilp64-gomp",
        feature = "mkl-static-ilp64-iomp",
        feature = "mkl-static-ilp64-seq",
        feature = "mkl-static-ilp64-tbb",
        feature = "mkl-static-lp64-gomp",
        feature = "mkl-static-lp64-iomp",
        feature = "mkl-static-lp64-seq",
        feature = "mkl-static-lp64-tbb",
        feature = "mkl-sdl",
    ))]
    pub use crate::cblas::cblas_level_one_mkl_64::*;
    #[cfg(any(
        feature = "mkl-dynamic-ilp64-gomp",
        feature = "mkl-dynamic-ilp64-iomp",
        feature = "mkl-dynamic-ilp64-seq",
        feature = "mkl-dynamic-ilp64-tbb",
        feature = "mkl-dynamic-lp64-gomp",
        feature = "mkl-dynamic-lp64-iomp",
        feature = "mkl-dynamic-lp64-seq",
        feature = "mkl-dynamic-lp64-tbb",
        feature = "mkl-static-ilp64-gomp",
        feature = "mkl-static-ilp64-iomp",
        feature = "mkl-static-ilp64-seq",
        feature = "mkl-static-ilp64-tbb",
        feature = "mkl-static-lp64-gomp",
        feature = "mkl-static-lp64-iomp",
        feature = "mkl-static-lp64-seq",
        feature = "mkl-static-lp64-tbb",
        feature = "mkl-sdl",
    ))]
    pub use crate::cblas::cblas_level_three_mkl::*;
    #[cfg(any(
        feature = "mkl-dynamic-ilp64-gomp",
        feature = "mkl-dynamic-ilp64-iomp",
        feature = "mkl-dynamic-ilp64-seq",
        feature = "mkl-dynamic-ilp64-tbb",
        feature = "mkl-dynamic-lp64-gomp",
        feature = "mkl-dynamic-lp64-iomp",
        feature = "mkl-dynamic-lp64-seq",
        feature = "mkl-dynamic-lp64-tbb",
        feature = "mkl-static-ilp64-gomp",
        feature = "mkl-static-ilp64-iomp",
        feature = "mkl-static-ilp64-seq",
        feature = "mkl-static-ilp64-tbb",
        feature = "mkl-static-lp64-gomp",
        feature = "mkl-static-lp64-iomp",
        feature = "mkl-static-lp64-seq",
        feature = "mkl-static-lp64-tbb",
        feature = "mkl-sdl",
    ))]
    pub use crate::cblas::cblas_level_three_mkl_64::*;
    #[cfg(any(
        feature = "mkl-dynamic-ilp64-gomp",
        feature = "mkl-dynamic-ilp64-iomp",
        feature = "mkl-dynamic-ilp64-seq",
        feature = "mkl-dynamic-ilp64-tbb",
        feature = "mkl-dynamic-lp64-gomp",
        feature = "mkl-dynamic-lp64-iomp",
        feature = "mkl-dynamic-lp64-seq",
        feature = "mkl-dynamic-lp64-tbb",
        feature = "mkl-static-ilp64-gomp",
        feature = "mkl-static-ilp64-iomp",
        feature = "mkl-static-ilp64-seq",
        feature = "mkl-static-ilp64-tbb",
        feature = "mkl-static-lp64-gomp",
        feature = "mkl-static-lp64-iomp",
        feature = "mkl-static-lp64-seq",
        feature = "mkl-static-lp64-tbb",
        feature = "mkl-sdl",
    ))]
    pub use crate::cblas::cblas_level_two_mkl_64::*;

    #[cfg(any(
        feature = "openblas-dynamic-ilp64",
        feature = "openblas-dynamic-lp64",
        feature = "openblas-static-ilp64",
        feature = "openblas-static-lp64",
        feature = "flexiblas-dynamic-ilp64",
        feature = "flexiblas-dynamic-lp64",
        feature = "flexiblas-static-ilp64",
        feature = "flexiblas-static-lp64",
    ))]
    pub use crate::cblas::cblas_level_one_openblas::*;
    #[cfg(any(
        feature = "openblas-dynamic-ilp64",
        feature = "openblas-dynamic-lp64",
        feature = "openblas-static-ilp64",
        feature = "openblas-static-lp64",
        feature = "flexiblas-dynamic-ilp64",
        feature = "flexiblas-dynamic-lp64",
        feature = "flexiblas-static-ilp64",
        feature = "flexiblas-static-lp64",
    ))]
    pub use crate::cblas::cblas_level_three_openblas::*;
}
