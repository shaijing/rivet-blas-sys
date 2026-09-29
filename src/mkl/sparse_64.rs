//! Intel oneMKL Inspector-Executor Sparse BLAS: explicit `*_64` symbols.
//!
//! These are actual declarations from oneMKL 2026.1 `mkl_spblas.h`, not renamed
//! ordinary symbols. Indices always use [`MklInt64`], including under LP64/SDL.
//! A compatible MKL version exporting these symbols is required; validated on
//! 2026.1. Keep creation, analysis, execution, export, and destruction within
//! this interface. See [`super::sparse`] for storage and lifetime requirements.
//!
//! ```no_run
//! # #[cfg(rivet_blas_mkl)]
//! # fn main() {
//! use rivet_blas_sys::mkl::{sparse_64::*, types::*};
//! let mut offsets: [MklInt64; 3] = [0, 1, 2];
//! let mut indices: [MklInt64; 2] = [0, 1];
//! let mut values = [2.0_f64, 3.0];
//! let x = [4.0_f64, 5.0];
//! let mut y = [0.0_f64; 2];
//! let mut a = std::ptr::null_mut();
//! unsafe {
//!     assert_eq!(mkl_sparse_d_create_csr_64(
//!         &mut a, SparseIndexBase::Zero, 2, 2,
//!         offsets.as_mut_ptr(), offsets.as_mut_ptr().add(1),
//!         indices.as_mut_ptr(), values.as_mut_ptr(),
//!     ), SparseStatus::SUCCESS);
//!     assert_eq!(mkl_sparse_d_mv_64(
//!         SparseOperation::NonTranspose, 1.0, a, MatrixDescr::default(),
//!         x.as_ptr(), 0.0, y.as_mut_ptr(),
//!     ), SparseStatus::SUCCESS);
//!     assert_eq!(mkl_sparse_destroy_64(a), SparseStatus::SUCCESS);
//! }
//! assert_eq!(y, [8.0, 15.0]);
//! # }
//! # #[cfg(not(rivet_blas_mkl))]
//! # fn main() {}
//! ```
use super::types::*;

unsafe extern "C" {
    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_create_csr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_create_csr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_create_csr_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt64,
        cols: MklInt64,
        rows_start: *mut MklInt64,
        rows_end: *mut MklInt64,
        col_indx: *mut MklInt64,
        values: *mut f32,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_create_csr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_create_csr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_create_csr_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt64,
        cols: MklInt64,
        rows_start: *mut MklInt64,
        rows_end: *mut MklInt64,
        col_indx: *mut MklInt64,
        values: *mut f64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_create_csr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_create_csr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_create_csr_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt64,
        cols: MklInt64,
        rows_start: *mut MklInt64,
        rows_end: *mut MklInt64,
        col_indx: *mut MklInt64,
        values: *mut MklComplex8,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_create_csr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_create_csr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_create_csr_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt64,
        cols: MklInt64,
        rows_start: *mut MklInt64,
        rows_end: *mut MklInt64,
        col_indx: *mut MklInt64,
        values: *mut MklComplex16,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_export_csr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_export_csr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_export_csr_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        rows_start: *mut *mut MklInt64,
        rows_end: *mut *mut MklInt64,
        col_indx: *mut *mut MklInt64,
        values: *mut *mut f32,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_export_csr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_export_csr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_export_csr_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        rows_start: *mut *mut MklInt64,
        rows_end: *mut *mut MklInt64,
        col_indx: *mut *mut MklInt64,
        values: *mut *mut f64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_export_csr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_export_csr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_export_csr_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        rows_start: *mut *mut MklInt64,
        rows_end: *mut *mut MklInt64,
        col_indx: *mut *mut MklInt64,
        values: *mut *mut MklComplex8,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_export_csr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_export_csr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_export_csr_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        rows_start: *mut *mut MklInt64,
        rows_end: *mut *mut MklInt64,
        col_indx: *mut *mut MklInt64,
        values: *mut *mut MklComplex16,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_create_csc`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_create_csc`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_create_csc_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt64,
        cols: MklInt64,
        cols_start: *mut MklInt64,
        cols_end: *mut MklInt64,
        row_indx: *mut MklInt64,
        values: *mut f32,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_create_csc`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_create_csc`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_create_csc_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt64,
        cols: MklInt64,
        cols_start: *mut MklInt64,
        cols_end: *mut MklInt64,
        row_indx: *mut MklInt64,
        values: *mut f64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_create_csc`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_create_csc`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_create_csc_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt64,
        cols: MklInt64,
        cols_start: *mut MklInt64,
        cols_end: *mut MklInt64,
        row_indx: *mut MklInt64,
        values: *mut MklComplex8,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_create_csc`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_create_csc`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_create_csc_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt64,
        cols: MklInt64,
        cols_start: *mut MklInt64,
        cols_end: *mut MklInt64,
        row_indx: *mut MklInt64,
        values: *mut MklComplex16,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_export_csc`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_export_csc`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_export_csc_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        cols_start: *mut *mut MklInt64,
        cols_end: *mut *mut MklInt64,
        row_indx: *mut *mut MklInt64,
        values: *mut *mut f32,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_export_csc`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_export_csc`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_export_csc_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        cols_start: *mut *mut MklInt64,
        cols_end: *mut *mut MklInt64,
        row_indx: *mut *mut MklInt64,
        values: *mut *mut f64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_export_csc`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_export_csc`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_export_csc_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        cols_start: *mut *mut MklInt64,
        cols_end: *mut *mut MklInt64,
        row_indx: *mut *mut MklInt64,
        values: *mut *mut MklComplex8,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_export_csc`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_export_csc`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_export_csc_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        cols_start: *mut *mut MklInt64,
        cols_end: *mut *mut MklInt64,
        row_indx: *mut *mut MklInt64,
        values: *mut *mut MklComplex16,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_create_coo`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_create_coo`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_create_coo_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt64,
        cols: MklInt64,
        nnz: MklInt64,
        row_indx: *mut MklInt64,
        col_indx: *mut MklInt64,
        values: *mut f32,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_create_coo`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_create_coo`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_create_coo_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt64,
        cols: MklInt64,
        nnz: MklInt64,
        row_indx: *mut MklInt64,
        col_indx: *mut MklInt64,
        values: *mut f64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_create_coo`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_create_coo`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_create_coo_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt64,
        cols: MklInt64,
        nnz: MklInt64,
        row_indx: *mut MklInt64,
        col_indx: *mut MklInt64,
        values: *mut MklComplex8,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_create_coo`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_create_coo`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_create_coo_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt64,
        cols: MklInt64,
        nnz: MklInt64,
        row_indx: *mut MklInt64,
        col_indx: *mut MklInt64,
        values: *mut MklComplex16,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_export_coo`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_export_coo`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_export_coo_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        nnz: *mut MklInt64,
        rows_indx: *mut *mut MklInt64,
        cols_indx: *mut *mut MklInt64,
        values: *mut *mut f32,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_export_coo`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_export_coo`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_export_coo_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        nnz: *mut MklInt64,
        rows_indx: *mut *mut MklInt64,
        cols_indx: *mut *mut MklInt64,
        values: *mut *mut f64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_export_coo`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_export_coo`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_export_coo_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        nnz: *mut MklInt64,
        rows_indx: *mut *mut MklInt64,
        cols_indx: *mut *mut MklInt64,
        values: *mut *mut MklComplex8,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_export_coo`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_export_coo`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_export_coo_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        nnz: *mut MklInt64,
        rows_indx: *mut *mut MklInt64,
        cols_indx: *mut *mut MklInt64,
        values: *mut *mut MklComplex16,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_create_bsr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_create_bsr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_create_bsr_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        block_layout: SparseLayout,
        rows: MklInt64,
        cols: MklInt64,
        block_size: MklInt64,
        rows_start: *mut MklInt64,
        rows_end: *mut MklInt64,
        col_indx: *mut MklInt64,
        values: *mut f32,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_create_bsr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_create_bsr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_create_bsr_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        block_layout: SparseLayout,
        rows: MklInt64,
        cols: MklInt64,
        block_size: MklInt64,
        rows_start: *mut MklInt64,
        rows_end: *mut MklInt64,
        col_indx: *mut MklInt64,
        values: *mut f64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_create_bsr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_create_bsr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_create_bsr_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        block_layout: SparseLayout,
        rows: MklInt64,
        cols: MklInt64,
        block_size: MklInt64,
        rows_start: *mut MklInt64,
        rows_end: *mut MklInt64,
        col_indx: *mut MklInt64,
        values: *mut MklComplex8,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_create_bsr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_create_bsr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_create_bsr_64(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        block_layout: SparseLayout,
        rows: MklInt64,
        cols: MklInt64,
        block_size: MklInt64,
        rows_start: *mut MklInt64,
        rows_end: *mut MklInt64,
        col_indx: *mut MklInt64,
        values: *mut MklComplex16,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_export_bsr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_export_bsr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_export_bsr_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        block_layout: *mut SparseLayout,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        block_size: *mut MklInt64,
        rows_start: *mut *mut MklInt64,
        rows_end: *mut *mut MklInt64,
        col_indx: *mut *mut MklInt64,
        values: *mut *mut f32,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_export_bsr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_export_bsr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_export_bsr_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        block_layout: *mut SparseLayout,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        block_size: *mut MklInt64,
        rows_start: *mut *mut MklInt64,
        rows_end: *mut *mut MklInt64,
        col_indx: *mut *mut MklInt64,
        values: *mut *mut f64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_export_bsr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_export_bsr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_export_bsr_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        block_layout: *mut SparseLayout,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        block_size: *mut MklInt64,
        rows_start: *mut *mut MklInt64,
        rows_end: *mut *mut MklInt64,
        col_indx: *mut *mut MklInt64,
        values: *mut *mut MklComplex8,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_export_bsr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_export_bsr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_export_bsr_64(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        block_layout: *mut SparseLayout,
        rows: *mut MklInt64,
        cols: *mut MklInt64,
        block_size: *mut MklInt64,
        rows_start: *mut *mut MklInt64,
        rows_end: *mut *mut MklInt64,
        col_indx: *mut *mut MklInt64,
        values: *mut *mut MklComplex16,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_copy`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_copy`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_copy_64(
        source: SparseMatrixHandle,
        descr: MatrixDescr,
        dest: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_destroy`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_destroy`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_destroy_64(a: SparseMatrixHandle) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_convert_csr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_convert_csr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_convert_csr_64(
        source: SparseMatrixHandle,
        operation: SparseOperation,
        dest: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_convert_csc`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_convert_csc`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_convert_csc_64(
        source: SparseMatrixHandle,
        operation: SparseOperation,
        dest: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_convert_coo`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_convert_coo`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_convert_coo_64(
        source: SparseMatrixHandle,
        operation: SparseOperation,
        dest: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_convert_bsr`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_convert_bsr`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_convert_bsr_64(
        source: SparseMatrixHandle,
        block_size: MklInt64,
        block_layout: SparseLayout,
        operation: SparseOperation,
        dest: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_order`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_order`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_order_64(a: SparseMatrixHandle) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_set_mv_hint`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_set_mv_hint`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_set_mv_hint_64(
        a: SparseMatrixHandle,
        operation: SparseOperation,
        descr: MatrixDescr,
        expected_calls: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_set_mm_hint`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_set_mm_hint`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_set_mm_hint_64(
        a: SparseMatrixHandle,
        operation: SparseOperation,
        descr: MatrixDescr,
        layout: SparseLayout,
        dense_matrix_size: MklInt64,
        expected_calls: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_set_sv_hint`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_set_sv_hint`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_set_sv_hint_64(
        a: SparseMatrixHandle,
        operation: SparseOperation,
        descr: MatrixDescr,
        expected_calls: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_set_sm_hint`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_set_sm_hint`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_set_sm_hint_64(
        a: SparseMatrixHandle,
        operation: SparseOperation,
        descr: MatrixDescr,
        layout: SparseLayout,
        dense_matrix_size: MklInt64,
        expected_calls: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_set_memory_hint`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_set_memory_hint`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_set_memory_hint_64(
        a: SparseMatrixHandle,
        policy: SparseMemoryUsage,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_optimize`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_optimize`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_optimize_64(a: SparseMatrixHandle) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_mv`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_mv`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_mv_64(
        operation: SparseOperation,
        alpha: f32,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const f32,
        beta: f32,
        y: *mut f32,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_mv`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_mv`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_mv_64(
        operation: SparseOperation,
        alpha: f64,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const f64,
        beta: f64,
        y: *mut f64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_mv`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_mv`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_mv_64(
        operation: SparseOperation,
        alpha: MklComplex8,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const MklComplex8,
        beta: MklComplex8,
        y: *mut MklComplex8,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_mv`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_mv`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_mv_64(
        operation: SparseOperation,
        alpha: MklComplex16,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const MklComplex16,
        beta: MklComplex16,
        y: *mut MklComplex16,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_mm`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_mm`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_mm_64(
        operation: SparseOperation,
        alpha: f32,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const f32,
        columns: MklInt64,
        ldx: MklInt64,
        beta: f32,
        y: *mut f32,
        ldy: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_mm`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_mm`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_mm_64(
        operation: SparseOperation,
        alpha: f64,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const f64,
        columns: MklInt64,
        ldx: MklInt64,
        beta: f64,
        y: *mut f64,
        ldy: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_mm`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_mm`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_mm_64(
        operation: SparseOperation,
        alpha: MklComplex8,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const MklComplex8,
        columns: MklInt64,
        ldx: MklInt64,
        beta: MklComplex8,
        y: *mut MklComplex8,
        ldy: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_mm`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_mm`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_mm_64(
        operation: SparseOperation,
        alpha: MklComplex16,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const MklComplex16,
        columns: MklInt64,
        ldx: MklInt64,
        beta: MklComplex16,
        y: *mut MklComplex16,
        ldy: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_trsv`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_trsv`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_trsv_64(
        operation: SparseOperation,
        alpha: f32,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const f32,
        y: *mut f32,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_trsv`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_trsv`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_trsv_64(
        operation: SparseOperation,
        alpha: f64,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const f64,
        y: *mut f64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_trsv`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_trsv`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_trsv_64(
        operation: SparseOperation,
        alpha: MklComplex8,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const MklComplex8,
        y: *mut MklComplex8,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_trsv`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_trsv`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_trsv_64(
        operation: SparseOperation,
        alpha: MklComplex16,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const MklComplex16,
        y: *mut MklComplex16,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_trsm`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_trsm`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_trsm_64(
        operation: SparseOperation,
        alpha: f32,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const f32,
        columns: MklInt64,
        ldx: MklInt64,
        y: *mut f32,
        ldy: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_trsm`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_trsm`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_trsm_64(
        operation: SparseOperation,
        alpha: f64,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const f64,
        columns: MklInt64,
        ldx: MklInt64,
        y: *mut f64,
        ldy: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_trsm`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_trsm`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_trsm_64(
        operation: SparseOperation,
        alpha: MklComplex8,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const MklComplex8,
        columns: MklInt64,
        ldx: MklInt64,
        y: *mut MklComplex8,
        ldy: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_trsm`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_trsm`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_trsm_64(
        operation: SparseOperation,
        alpha: MklComplex16,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const MklComplex16,
        columns: MklInt64,
        ldx: MklInt64,
        y: *mut MklComplex16,
        ldy: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_add`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_add`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_add_64(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        alpha: f32,
        b: SparseMatrixHandle,
        c: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_add`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_add`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_add_64(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        alpha: f64,
        b: SparseMatrixHandle,
        c: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_add`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_add`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_add_64(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        alpha: MklComplex8,
        b: SparseMatrixHandle,
        c: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_add`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_add`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_add_64(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        alpha: MklComplex16,
        b: SparseMatrixHandle,
        c: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_spmmd`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_spmmd`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_spmmd_64(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        b: SparseMatrixHandle,
        layout: SparseLayout,
        c: *mut f32,
        ldc: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_spmmd`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_spmmd`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_spmmd_64(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        b: SparseMatrixHandle,
        layout: SparseLayout,
        c: *mut f64,
        ldc: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_spmmd`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_spmmd`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_spmmd_64(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        b: SparseMatrixHandle,
        layout: SparseLayout,
        c: *mut MklComplex8,
        ldc: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_spmmd`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_spmmd`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_spmmd_64(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        b: SparseMatrixHandle,
        layout: SparseLayout,
        c: *mut MklComplex16,
        ldc: MklInt64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_set_value`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_set_value`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_set_value_64(
        a: SparseMatrixHandle,
        row: MklInt64,
        col: MklInt64,
        value: f32,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_set_value`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_set_value`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_set_value_64(
        a: SparseMatrixHandle,
        row: MklInt64,
        col: MklInt64,
        value: f64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_set_value`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_set_value`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_set_value_64(
        a: SparseMatrixHandle,
        row: MklInt64,
        col: MklInt64,
        value: MklComplex8,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_set_value`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_set_value`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_set_value_64(
        a: SparseMatrixHandle,
        row: MklInt64,
        col: MklInt64,
        value: MklComplex16,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_s_update_values`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_s_update_values`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_s_update_values_64(
        a: SparseMatrixHandle,
        nvalues: MklInt64,
        indx: *const MklInt64,
        indy: *const MklInt64,
        values: *mut f32,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_d_update_values`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_d_update_values`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_d_update_values_64(
        a: SparseMatrixHandle,
        nvalues: MklInt64,
        indx: *const MklInt64,
        indy: *const MklInt64,
        values: *mut f64,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_c_update_values`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_c_update_values`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_c_update_values_64(
        a: SparseMatrixHandle,
        nvalues: MklInt64,
        indx: *const MklInt64,
        indy: *const MklInt64,
        values: *mut MklComplex8,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_z_update_values`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_z_update_values`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_z_update_values_64(
        a: SparseMatrixHandle,
        nvalues: MklInt64,
        indx: *const MklInt64,
        indy: *const MklInt64,
        values: *mut MklComplex16,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_spmm`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_spmm`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_spmm_64(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        b: SparseMatrixHandle,
        c: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Explicit 64-bit version of [`super::sparse::mkl_sparse_sp2m`].
    /// All indices and dimensions use `MKL_INT64`; the handle must belong to
    /// the explicit 64-bit interface. Requires an MKL version exporting this symbol.
    ///
    /// # Safety
    /// All pointer, storage, ownership, and staging preconditions of
    /// [`super::sparse::mkl_sparse_sp2m`] apply, with 64-bit index arrays
    /// and a handle created and used exclusively through the `*_64` interface.
    pub fn mkl_sparse_sp2m_64(
        transa: SparseOperation,
        descra: MatrixDescr,
        a: SparseMatrixHandle,
        transb: SparseOperation,
        descrb: MatrixDescr,
        b: SparseMatrixHandle,
        request: SparseRequest,
        c: *mut SparseMatrixHandle,
    ) -> SparseStatus;

}
