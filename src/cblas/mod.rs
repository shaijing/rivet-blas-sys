pub mod cblas_level_one;
pub mod cblas_level_three;
pub mod cblas_level_two;
pub mod cblas_types;

#[cfg(any(feature = "intel-mkl", docsrs))]
pub mod cblas_level_one_mkl;
#[cfg(any(feature = "intel-mkl", docsrs))]
pub mod cblas_level_one_mkl_64;
#[cfg(any(feature = "intel-mkl", docsrs))]
pub mod cblas_level_three_mkl;
#[cfg(any(feature = "intel-mkl", docsrs))]
pub mod cblas_level_three_mkl_64;
#[cfg(any(feature = "intel-mkl", docsrs))]
pub mod cblas_level_two_mkl_64;

#[cfg(any(
    docsrs,
    all(
        any(feature = "openblas", feature = "flexiblas"),
        not(feature = "intel-mkl")
    )
))]
pub mod cblas_level_one_openblas;
#[cfg(any(
    docsrs,
    all(
        any(feature = "openblas", feature = "flexiblas"),
        not(feature = "intel-mkl")
    )
))]
pub mod cblas_level_three_openblas;

pub mod prelude {
    pub use crate::cblas::cblas_level_one::*;
    pub use crate::cblas::cblas_level_three::*;
    pub use crate::cblas::cblas_level_two::*;
    pub use crate::cblas::cblas_types::*;

    #[cfg(feature = "intel-mkl")]
    pub use crate::cblas::cblas_level_one_mkl::*;
    #[cfg(feature = "intel-mkl")]
    pub use crate::cblas::cblas_level_one_mkl_64::*;
    #[cfg(feature = "intel-mkl")]
    pub use crate::cblas::cblas_level_three_mkl::*;
    #[cfg(feature = "intel-mkl")]
    pub use crate::cblas::cblas_level_three_mkl_64::*;
    #[cfg(feature = "intel-mkl")]
    pub use crate::cblas::cblas_level_two_mkl_64::*;

    #[cfg(all(
        any(feature = "openblas", feature = "flexiblas"),
        not(feature = "intel-mkl")
    ))]
    pub use crate::cblas::cblas_level_one_openblas::*;
    #[cfg(all(
        any(feature = "openblas", feature = "flexiblas"),
        not(feature = "intel-mkl")
    ))]
    pub use crate::cblas::cblas_level_three_openblas::*;
}
