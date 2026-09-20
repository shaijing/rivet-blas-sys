pub mod cblas_level_one;
pub mod cblas_level_three;
pub mod cblas_level_two;
pub mod cblas_types;

#[cfg(feature = "intel-mkl")]
pub mod cblas_level_one_mkl;
#[cfg(feature = "intel-mkl")]
pub mod cblas_level_one_mkl_64;
#[cfg(feature = "intel-mkl")]
pub mod cblas_level_three_mkl;
#[cfg(feature = "intel-mkl")]
pub mod cblas_level_three_mkl_64;
#[cfg(feature = "intel-mkl")]
pub mod cblas_level_two_mkl_64;

#[cfg(all(feature = "openblas", not(feature = "intel-mkl")))]
pub mod cblas_level_one_openblas;
#[cfg(all(feature = "openblas", not(feature = "intel-mkl")))]
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

    #[cfg(all(feature = "openblas", not(feature = "intel-mkl")))]
    pub use crate::cblas::cblas_level_one_openblas::*;
    #[cfg(all(feature = "openblas", not(feature = "intel-mkl")))]
    pub use crate::cblas::cblas_level_three_openblas::*;
}
