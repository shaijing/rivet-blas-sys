//! # rivet-blas-sys
//! The packages provides a BLAS binding.

//! ## Configuration
//! The following implementations are available:

//! Complete native link configurations are exposed as features, for example
//! `openblas-static-lp64`, `mkl-static-ilp64-iomp`, and `mkl-sdl`.
//! Exactly one complete configuration feature must be enabled.
//!
//! [intel mkl]: https://software.intel.com/en-us/mkl
//! [openblas]: https://github.com/OpenMathLib/OpenBLAS

pub mod blas_types;
pub mod cblas;
