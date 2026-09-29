//! Intel oneMKL APIs beyond CBLAS.
//!
//! These modules require an MKL configuration. Ordinary Sparse BLAS functions
//! use the selected LP64/ILP64 ABI; [`sparse_64`] contains actual suffixed MKL
//! entry points and always uses 64-bit indices. Keep a handle and all operations
//! on it within the same interface, including export and destruction.
//!
//! The declarations are low-level FFI: callers manage storage and handle
//! lifetimes. In particular, creation borrows the supplied arrays, and export
//! returns borrowed pointers rather than transferring ownership.

pub mod sparse;
pub mod sparse_64;
pub mod types;

/// Ordinary Sparse BLAS declarations and MKL-specific types.
pub mod prelude {
    pub use super::sparse::*;
    pub use super::types::*;
}
