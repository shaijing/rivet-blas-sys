//! # rivet-blas-sys
//! The packages provides a BLAS binding.

//! ## Configuration
//! The following implementations are available:

//! - `intel-mkl`, which is the one in [Intel MKL]
//! - `openblas`, which is the one in [OpenBLAS]
//! - `flexiblas`, which selects a system FlexiBLAS installation
//! - `netlib`, which selects the reference BLAS implementation
//! - `accelerate`, which selects Apple's Accelerate framework
//!
//! [intel mkl]: https://software.intel.com/en-us/mkl
//! [openblas]: https://github.com/OpenMathLib/OpenBLAS

#[cfg(all(
    not(docsrs),
    not(any(
        feature = "openblas",
        feature = "flexiblas",
        feature = "intel-mkl",
        feature = "netlib",
        feature = "accelerate",
        feature = "system-blas",
    )),
))]
compile_error!("enable exactly one BLAS backend feature");

#[cfg(all(
    not(docsrs),
    any(
        all(feature = "openblas", feature = "flexiblas"),
        all(feature = "openblas", feature = "intel-mkl"),
        all(feature = "openblas", feature = "netlib"),
        all(feature = "openblas", feature = "accelerate"),
        all(feature = "openblas", feature = "system-blas"),
        all(feature = "flexiblas", feature = "intel-mkl"),
        all(feature = "flexiblas", feature = "netlib"),
        all(feature = "flexiblas", feature = "accelerate"),
        all(feature = "flexiblas", feature = "system-blas"),
        all(feature = "intel-mkl", feature = "netlib"),
        all(feature = "intel-mkl", feature = "accelerate"),
        all(feature = "intel-mkl", feature = "system-blas"),
        all(feature = "netlib", feature = "accelerate"),
        all(feature = "netlib", feature = "system-blas"),
        all(feature = "accelerate", feature = "system-blas"),
    ),
))]
compile_error!("BLAS backend features are mutually exclusive; enable exactly one");

#[cfg(all(
    not(docsrs),
    any(
        all(feature = "ilp64", feature = "lp64"),
        not(any(feature = "ilp64", feature = "lp64")),
    ),
))]
compile_error!("enable exactly one integer ABI feature: `ilp64` or `lp64`");

#[cfg(all(
    not(docsrs),
    any(
        all(feature = "netlib", feature = "ilp64"),
        all(feature = "accelerate", feature = "ilp64"),
    ),
))]
compile_error!("the selected BLAS backend supports LP64 only");

#[cfg(all(not(docsrs), feature = "accelerate", not(target_os = "macos")))]
compile_error!("the `accelerate` backend is only available on macOS");

pub mod blas_types;
pub mod cblas;
