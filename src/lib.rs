//! # rivet-blas-sys
//! The packages provides a BLAS binding.

//! ## Configuration
//! The following implementations are available:

//! - `intel-mkl`, which is the one in [Intel MKL]
//! - `openblas`, which is the one in [OpenBLAS]
//!
//! [intel mkl]: https://software.intel.com/en-us/mkl
//! [openblas]: https://github.com/OpenMathLib/OpenBLAS

pub mod blas_types;
pub mod cblas;
