pub mod cblas_level_one;
pub mod cblas_level_three;
pub mod cblas_level_two;
pub mod cblas_types;

#[cfg(any(rivet_blas_mkl, docsrs))]
pub mod cblas_level_one_mkl;
#[cfg(any(rivet_blas_mkl, docsrs))]
pub mod cblas_level_one_mkl_64;
#[cfg(any(rivet_blas_mkl, docsrs))]
pub mod cblas_level_three_mkl;
#[cfg(any(rivet_blas_mkl, docsrs))]
pub mod cblas_level_three_mkl_64;
#[cfg(any(rivet_blas_mkl, docsrs))]
pub mod cblas_level_two_mkl_64;

#[cfg(any(rivet_blas_openblas, rivet_blas_flexiblas, docsrs))]
pub mod cblas_level_one_openblas;
#[cfg(any(rivet_blas_openblas, rivet_blas_flexiblas, docsrs))]
pub mod cblas_level_three_openblas;

pub mod prelude {
    pub use crate::cblas::cblas_level_one::*;
    pub use crate::cblas::cblas_level_three::*;
    pub use crate::cblas::cblas_level_two::*;
    pub use crate::cblas::cblas_types::*;

    #[cfg(rivet_blas_mkl)]
    pub use crate::cblas::cblas_level_one_mkl::*;
    #[cfg(rivet_blas_mkl)]
    pub use crate::cblas::cblas_level_one_mkl_64::*;
    #[cfg(rivet_blas_mkl)]
    pub use crate::cblas::cblas_level_three_mkl::*;
    #[cfg(rivet_blas_mkl)]
    pub use crate::cblas::cblas_level_three_mkl_64::*;
    #[cfg(rivet_blas_mkl)]
    pub use crate::cblas::cblas_level_two_mkl_64::*;

    #[cfg(any(rivet_blas_openblas, rivet_blas_flexiblas))]
    pub use crate::cblas::cblas_level_one_openblas::*;
    #[cfg(any(rivet_blas_openblas, rivet_blas_flexiblas))]
    pub use crate::cblas::cblas_level_three_openblas::*;
}
