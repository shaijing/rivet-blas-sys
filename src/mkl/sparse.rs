//! Intel oneMKL Inspector-Executor Sparse BLAS: ordinary LP64/ILP64 symbols.
//!
//! Indices use [`MklInt`], selected by the complete MKL feature. SDL uses LP64;
//! its runtime interface must remain LP64 when calling these declarations.
//! Complex scalars use C-layout [`MklComplex8`] / [`MklComplex16`] by value.
//!
//! Creation borrows arrays: keep them alive at stable addresses until the
//! handle no longer references them. Creation takes mutable pointers because
//! explicit ordering and update operations can modify the arrays. Export
//! borrows storage and does not transfer ownership. Successful new handles
//! must be destroyed exactly once. Never mix ordinary and explicit 64-bit
//! operations on a handle, even if their integer widths happen to match.
//!
//! Format, precision, transpose, and descriptor combinations are limited by
//! MKL; functions report [`SparseStatus::NOT_SUPPORTED`] for unsupported cases.
//! See the [Intel reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/inspector-executor-sparse-blas-routines.html).
//!
//! ```no_run
//! # #[cfg(rivet_blas_mkl)]
//! # fn main() {
//! use rivet_blas_sys::mkl::{sparse::*, types::*};
//! let mut offsets: [MklInt; 3] = [0, 1, 2];
//! let mut indices: [MklInt; 2] = [0, 1];
//! let mut values = [2.0_f64, 3.0];
//! let x = [4.0_f64, 5.0];
//! let mut y = [0.0_f64; 2];
//! let mut a = std::ptr::null_mut();
//! unsafe {
//!     assert_eq!(mkl_sparse_d_create_csr(
//!         &mut a, SparseIndexBase::Zero, 2, 2,
//!         offsets.as_mut_ptr(), offsets.as_mut_ptr().add(1),
//!         indices.as_mut_ptr(), values.as_mut_ptr(),
//!     ), SparseStatus::SUCCESS);
//!     assert_eq!(mkl_sparse_d_mv(
//!         SparseOperation::NonTranspose, 1.0, a, MatrixDescr::default(),
//!         x.as_ptr(), 0.0, y.as_mut_ptr(),
//!     ), SparseStatus::SUCCESS);
//!     assert_eq!(mkl_sparse_destroy(a), SparseStatus::SUCCESS);
//! }
//! assert_eq!(y, [8.0, 15.0]);
//! # }
//! # #[cfg(not(rivet_blas_mkl))]
//! # fn main() {}
//! ```
use super::types::*;

unsafe extern "C" {
    /// Creates a CSR matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` are scalar dimensions. `rows_start`/`rows_end` contain
    /// the inclusive/exclusive offsets of each row; `col_indx` contains column indices.
    /// For conventional CSR offsets `row_ptr`, pass `row_ptr` and `row_ptr + 1`.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_s_create_csr(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt,
        cols: MklInt,
        rows_start: *mut MklInt,
        rows_end: *mut MklInt,
        col_indx: *mut MklInt,
        values: *mut f32,
    ) -> SparseStatus;

    /// Creates a CSR matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` are scalar dimensions. `rows_start`/`rows_end` contain
    /// the inclusive/exclusive offsets of each row; `col_indx` contains column indices.
    /// For conventional CSR offsets `row_ptr`, pass `row_ptr` and `row_ptr + 1`.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_d_create_csr(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt,
        cols: MklInt,
        rows_start: *mut MklInt,
        rows_end: *mut MklInt,
        col_indx: *mut MklInt,
        values: *mut f64,
    ) -> SparseStatus;

    /// Creates a CSR matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` are scalar dimensions. `rows_start`/`rows_end` contain
    /// the inclusive/exclusive offsets of each row; `col_indx` contains column indices.
    /// For conventional CSR offsets `row_ptr`, pass `row_ptr` and `row_ptr + 1`.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_c_create_csr(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt,
        cols: MklInt,
        rows_start: *mut MklInt,
        rows_end: *mut MklInt,
        col_indx: *mut MklInt,
        values: *mut MklComplex8,
    ) -> SparseStatus;

    /// Creates a CSR matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` are scalar dimensions. `rows_start`/`rows_end` contain
    /// the inclusive/exclusive offsets of each row; `col_indx` contains column indices.
    /// For conventional CSR offsets `row_ptr`, pass `row_ptr` and `row_ptr + 1`.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_z_create_csr(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt,
        cols: MklInt,
        rows_start: *mut MklInt,
        rows_end: *mut MklInt,
        col_indx: *mut MklInt,
        values: *mut MklComplex16,
    ) -> SparseStatus;

    /// Exports the CSR representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` are scalar dimensions; `rows_start`/`rows_end` delimit rows.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_s_export_csr(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt,
        cols: *mut MklInt,
        rows_start: *mut *mut MklInt,
        rows_end: *mut *mut MklInt,
        col_indx: *mut *mut MklInt,
        values: *mut *mut f32,
    ) -> SparseStatus;

    /// Exports the CSR representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` are scalar dimensions; `rows_start`/`rows_end` delimit rows.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_d_export_csr(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt,
        cols: *mut MklInt,
        rows_start: *mut *mut MklInt,
        rows_end: *mut *mut MklInt,
        col_indx: *mut *mut MklInt,
        values: *mut *mut f64,
    ) -> SparseStatus;

    /// Exports the CSR representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` are scalar dimensions; `rows_start`/`rows_end` delimit rows.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_c_export_csr(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt,
        cols: *mut MklInt,
        rows_start: *mut *mut MklInt,
        rows_end: *mut *mut MklInt,
        col_indx: *mut *mut MklInt,
        values: *mut *mut MklComplex8,
    ) -> SparseStatus;

    /// Exports the CSR representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` are scalar dimensions; `rows_start`/`rows_end` delimit rows.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_z_export_csr(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt,
        cols: *mut MklInt,
        rows_start: *mut *mut MklInt,
        rows_end: *mut *mut MklInt,
        col_indx: *mut *mut MklInt,
        values: *mut *mut MklComplex16,
    ) -> SparseStatus;

    /// Creates a CSC matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` are scalar dimensions. `cols_start`/`cols_end` contain
    /// the inclusive/exclusive offsets of each column; `row_indx` contains row indices.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_s_create_csc(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt,
        cols: MklInt,
        cols_start: *mut MklInt,
        cols_end: *mut MklInt,
        row_indx: *mut MklInt,
        values: *mut f32,
    ) -> SparseStatus;

    /// Creates a CSC matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` are scalar dimensions. `cols_start`/`cols_end` contain
    /// the inclusive/exclusive offsets of each column; `row_indx` contains row indices.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_d_create_csc(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt,
        cols: MklInt,
        cols_start: *mut MklInt,
        cols_end: *mut MklInt,
        row_indx: *mut MklInt,
        values: *mut f64,
    ) -> SparseStatus;

    /// Creates a CSC matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` are scalar dimensions. `cols_start`/`cols_end` contain
    /// the inclusive/exclusive offsets of each column; `row_indx` contains row indices.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_c_create_csc(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt,
        cols: MklInt,
        cols_start: *mut MklInt,
        cols_end: *mut MklInt,
        row_indx: *mut MklInt,
        values: *mut MklComplex8,
    ) -> SparseStatus;

    /// Creates a CSC matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` are scalar dimensions. `cols_start`/`cols_end` contain
    /// the inclusive/exclusive offsets of each column; `row_indx` contains row indices.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_z_create_csc(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt,
        cols: MklInt,
        cols_start: *mut MklInt,
        cols_end: *mut MklInt,
        row_indx: *mut MklInt,
        values: *mut MklComplex16,
    ) -> SparseStatus;

    /// Exports the CSC representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` are scalar dimensions; `cols_start`/`cols_end` delimit columns.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_s_export_csc(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt,
        cols: *mut MklInt,
        cols_start: *mut *mut MklInt,
        cols_end: *mut *mut MklInt,
        row_indx: *mut *mut MklInt,
        values: *mut *mut f32,
    ) -> SparseStatus;

    /// Exports the CSC representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` are scalar dimensions; `cols_start`/`cols_end` delimit columns.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_d_export_csc(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt,
        cols: *mut MklInt,
        cols_start: *mut *mut MklInt,
        cols_end: *mut *mut MklInt,
        row_indx: *mut *mut MklInt,
        values: *mut *mut f64,
    ) -> SparseStatus;

    /// Exports the CSC representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` are scalar dimensions; `cols_start`/`cols_end` delimit columns.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_c_export_csc(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt,
        cols: *mut MklInt,
        cols_start: *mut *mut MklInt,
        cols_end: *mut *mut MklInt,
        row_indx: *mut *mut MklInt,
        values: *mut *mut MklComplex8,
    ) -> SparseStatus;

    /// Exports the CSC representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` are scalar dimensions; `cols_start`/`cols_end` delimit columns.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_z_export_csc(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt,
        cols: *mut MklInt,
        cols_start: *mut *mut MklInt,
        cols_end: *mut *mut MklInt,
        row_indx: *mut *mut MklInt,
        values: *mut *mut MklComplex16,
    ) -> SparseStatus;

    /// Creates a COO matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` are scalar dimensions. `nnz` is the number of entries;
    /// `row_indx`, `col_indx`, and `values` each contain `nnz` entries.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_s_create_coo(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt,
        cols: MklInt,
        nnz: MklInt,
        row_indx: *mut MklInt,
        col_indx: *mut MklInt,
        values: *mut f32,
    ) -> SparseStatus;

    /// Creates a COO matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` are scalar dimensions. `nnz` is the number of entries;
    /// `row_indx`, `col_indx`, and `values` each contain `nnz` entries.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_d_create_coo(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt,
        cols: MklInt,
        nnz: MklInt,
        row_indx: *mut MklInt,
        col_indx: *mut MklInt,
        values: *mut f64,
    ) -> SparseStatus;

    /// Creates a COO matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` are scalar dimensions. `nnz` is the number of entries;
    /// `row_indx`, `col_indx`, and `values` each contain `nnz` entries.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_c_create_coo(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt,
        cols: MklInt,
        nnz: MklInt,
        row_indx: *mut MklInt,
        col_indx: *mut MklInt,
        values: *mut MklComplex8,
    ) -> SparseStatus;

    /// Creates a COO matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` are scalar dimensions. `nnz` is the number of entries;
    /// `row_indx`, `col_indx`, and `values` each contain `nnz` entries.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_z_create_coo(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        rows: MklInt,
        cols: MklInt,
        nnz: MklInt,
        row_indx: *mut MklInt,
        col_indx: *mut MklInt,
        values: *mut MklComplex16,
    ) -> SparseStatus;

    /// Exports the COO representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` are scalar dimensions and `nnz` is the number of entries.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_s_export_coo(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt,
        cols: *mut MklInt,
        nnz: *mut MklInt,
        rows_indx: *mut *mut MklInt,
        cols_indx: *mut *mut MklInt,
        values: *mut *mut f32,
    ) -> SparseStatus;

    /// Exports the COO representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` are scalar dimensions and `nnz` is the number of entries.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_d_export_coo(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt,
        cols: *mut MklInt,
        nnz: *mut MklInt,
        rows_indx: *mut *mut MklInt,
        cols_indx: *mut *mut MklInt,
        values: *mut *mut f64,
    ) -> SparseStatus;

    /// Exports the COO representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` are scalar dimensions and `nnz` is the number of entries.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_c_export_coo(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt,
        cols: *mut MklInt,
        nnz: *mut MklInt,
        rows_indx: *mut *mut MklInt,
        cols_indx: *mut *mut MklInt,
        values: *mut *mut MklComplex8,
    ) -> SparseStatus;

    /// Exports the COO representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` are scalar dimensions and `nnz` is the number of entries.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_z_export_coo(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        rows: *mut MklInt,
        cols: *mut MklInt,
        nnz: *mut MklInt,
        rows_indx: *mut *mut MklInt,
        cols_indx: *mut *mut MklInt,
        values: *mut *mut MklComplex16,
    ) -> SparseStatus;

    /// Creates a BSR matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` count block rows and block columns. Each stored block
    /// contains `block_size * block_size` values in `block_layout` order;
    /// `rows_start`/`rows_end` delimit blocks and `col_indx` contains block columns.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_s_create_bsr(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        block_layout: SparseLayout,
        rows: MklInt,
        cols: MklInt,
        block_size: MklInt,
        rows_start: *mut MklInt,
        rows_end: *mut MklInt,
        col_indx: *mut MklInt,
        values: *mut f32,
    ) -> SparseStatus;

    /// Creates a BSR matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` count block rows and block columns. Each stored block
    /// contains `block_size * block_size` values in `block_layout` order;
    /// `rows_start`/`rows_end` delimit blocks and `col_indx` contains block columns.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_d_create_bsr(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        block_layout: SparseLayout,
        rows: MklInt,
        cols: MklInt,
        block_size: MklInt,
        rows_start: *mut MklInt,
        rows_end: *mut MklInt,
        col_indx: *mut MklInt,
        values: *mut f64,
    ) -> SparseStatus;

    /// Creates a BSR matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` count block rows and block columns. Each stored block
    /// contains `block_size * block_size` values in `block_layout` order;
    /// `rows_start`/`rows_end` delimit blocks and `col_indx` contains block columns.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_c_create_bsr(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        block_layout: SparseLayout,
        rows: MklInt,
        cols: MklInt,
        block_size: MklInt,
        rows_start: *mut MklInt,
        rows_end: *mut MklInt,
        col_indx: *mut MklInt,
        values: *mut MklComplex8,
    ) -> SparseStatus;

    /// Creates a BSR matrix handle in `A`, borrowing the supplied index and value arrays.
    /// `rows` and `cols` count block rows and block columns. Each stored block
    /// contains `block_size * block_size` values in `block_layout` order;
    /// `rows_start`/`rows_end` delimit blocks and `col_indx` contains block columns.
    /// `indexing` selects the base of both offsets and indices. Explicit ordering
    /// or value updates may mutate these arrays. Destroy the resulting handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be writable. Offsets and indices must describe valid matrix storage,
    /// and values must cover all referenced entries. Keep all input arrays alive and at
    /// stable addresses while the handle borrows them. Prevent conflicting access
    /// during calls, including explicit mutations by ordering or value updates.
    pub fn mkl_sparse_z_create_bsr(
        a: *mut SparseMatrixHandle,
        indexing: SparseIndexBase,
        block_layout: SparseLayout,
        rows: MklInt,
        cols: MklInt,
        block_size: MklInt,
        rows_start: *mut MklInt,
        rows_end: *mut MklInt,
        col_indx: *mut MklInt,
        values: *mut MklComplex16,
    ) -> SparseStatus;

    /// Exports the BSR representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` count blocks. `block_size` and `block_layout` describe each block.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_s_export_bsr(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        block_layout: *mut SparseLayout,
        rows: *mut MklInt,
        cols: *mut MklInt,
        block_size: *mut MklInt,
        rows_start: *mut *mut MklInt,
        rows_end: *mut *mut MklInt,
        col_indx: *mut *mut MklInt,
        values: *mut *mut f32,
    ) -> SparseStatus;

    /// Exports the BSR representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` count blocks. `block_size` and `block_layout` describe each block.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_d_export_bsr(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        block_layout: *mut SparseLayout,
        rows: *mut MklInt,
        cols: *mut MklInt,
        block_size: *mut MklInt,
        rows_start: *mut *mut MklInt,
        rows_end: *mut *mut MklInt,
        col_indx: *mut *mut MklInt,
        values: *mut *mut f64,
    ) -> SparseStatus;

    /// Exports the BSR representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` count blocks. `block_size` and `block_layout` describe each block.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_c_export_bsr(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        block_layout: *mut SparseLayout,
        rows: *mut MklInt,
        cols: *mut MklInt,
        block_size: *mut MklInt,
        rows_start: *mut *mut MklInt,
        rows_end: *mut *mut MklInt,
        col_indx: *mut *mut MklInt,
        values: *mut *mut MklComplex8,
    ) -> SparseStatus;

    /// Exports the BSR representation of `source` through borrowed output pointers.
    /// No array ownership is transferred. Do not free the returned arrays; read them
    /// only while their backing storage remains valid and before an operation that
    /// changes or destroys that storage. `indexing` reports the base of offsets/indices.
    /// `rows`/`cols` count blocks. `block_size` and `block_layout` describe each block.
    /// A partially computed staged product may have offsets but no indices or values.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` must be a live handle of the matching precision, format, and ABI.
    /// Every output parameter must point to writable storage of its declared type.
    /// Do not dereference returned pointers unless the successful operation and its
    /// computation stage guarantee the corresponding arrays are present.
    pub fn mkl_sparse_z_export_bsr(
        source: SparseMatrixHandle,
        indexing: *mut SparseIndexBase,
        block_layout: *mut SparseLayout,
        rows: *mut MklInt,
        cols: *mut MklInt,
        block_size: *mut MklInt,
        rows_start: *mut *mut MklInt,
        rows_end: *mut *mut MklInt,
        col_indx: *mut *mut MklInt,
        values: *mut *mut MklComplex16,
    ) -> SparseStatus;

    /// Deep-copies the matrix represented by `source` into a new handle in `dest`.
    /// Current MKL ignores `descr` and copies the complete matrix. The copy owns
    /// independent storage and may outlive the source and its borrowed arrays.
    /// Destroy the output handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` and its backing arrays must be valid. `dest` must be writable
    /// and must not overwrite an owned live handle. The descriptor must agree
    /// with the stored matrix structure.
    pub fn mkl_sparse_copy(
        source: SparseMatrixHandle,
        descr: MatrixDescr,
        dest: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Destroys `A` and releases MKL-owned storage associated with it.
    /// User-supplied creation arrays remain owned by the caller. Exported pointers
    /// into released storage become invalid.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be a live handle from this ABI, destroyed exactly once, with no
    /// concurrent calls or dependent handles still using the released storage.
    pub fn mkl_sparse_destroy(a: SparseMatrixHandle) -> SparseStatus;

    /// Creates a CSR representation of `op(source)` in `dest`.
    /// `operation` selects the transpose/conjugate transpose. Destroy the output handle
    /// after use. Supported source formats and operations depend on the MKL version.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` and its borrowed storage must remain valid for this call.
    /// `dest` must point to writable handle storage and must not overwrite a live
    /// handle whose ownership would be lost. All format constraints must hold.
    pub fn mkl_sparse_convert_csr(
        source: SparseMatrixHandle,
        operation: SparseOperation,
        dest: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Creates a CSC representation of `op(source)` in `dest`.
    /// `operation` selects the transpose/conjugate transpose. Destroy the output handle
    /// after use. Supported source formats and operations depend on the MKL version.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` and its borrowed storage must remain valid for this call.
    /// `dest` must point to writable handle storage and must not overwrite a live
    /// handle whose ownership would be lost. All format constraints must hold.
    pub fn mkl_sparse_convert_csc(
        source: SparseMatrixHandle,
        operation: SparseOperation,
        dest: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Creates a COO representation of `op(source)` in `dest`.
    /// `operation` selects the transpose/conjugate transpose. Destroy the output handle
    /// after use. Supported source formats and operations depend on the MKL version.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` and its borrowed storage must remain valid for this call.
    /// `dest` must point to writable handle storage and must not overwrite a live
    /// handle whose ownership would be lost. All format constraints must hold.
    pub fn mkl_sparse_convert_coo(
        source: SparseMatrixHandle,
        operation: SparseOperation,
        dest: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Creates a BSR representation of `op(source)` in `dest`.
    /// `operation` selects the transpose/conjugate transpose. Destroy the output handle
    /// after use. Supported source formats and operations depend on the MKL version.
    /// `block_size` is the number of scalar rows/columns per block;
    /// `block_layout` selects the storage order of values within each block.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `source` and its borrowed storage must remain valid for this call.
    /// `dest` must point to writable handle storage and must not overwrite a live
    /// handle whose ownership would be lost. All format constraints must hold.
    pub fn mkl_sparse_convert_bsr(
        source: SparseMatrixHandle,
        block_size: MklInt,
        block_layout: SparseLayout,
        operation: SparseOperation,
        dest: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Orders column indices within CSR rows or row indices within CSC columns.
    /// Corresponding values are reordered in place; borrowed creation arrays may change.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be a live handle with writable index and value storage.
    /// No other access to its arrays or exported pointers may conflict with reordering.
    pub fn mkl_sparse_order(a: SparseMatrixHandle) -> SparseStatus;

    /// Records the expected matrix-vector multiplication workload for `A`.
    /// `operation` and `descr` describe future calls; `expected_calls` is their expected
    /// count. Call `mkl_sparse_optimize` after setting hints to prepare execution.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` and its backing arrays must remain valid, and must not be accessed
    /// concurrently in a way that conflicts with MKL updating inspector metadata.
    /// The descriptor, dimensions, and workload must satisfy MKL constraints.
    pub fn mkl_sparse_set_mv_hint(
        a: SparseMatrixHandle,
        operation: SparseOperation,
        descr: MatrixDescr,
        expected_calls: MklInt,
    ) -> SparseStatus;

    /// Records the expected matrix-matrix multiplication workload for `A`.
    /// `operation` and `descr` describe future calls; `expected_calls` is their expected
    /// count. Call `mkl_sparse_optimize` after setting hints to prepare execution.
    /// `layout` and `dense_matrix_size` describe the dense operand storage and its number of columns.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` and its backing arrays must remain valid, and must not be accessed
    /// concurrently in a way that conflicts with MKL updating inspector metadata.
    /// The descriptor, dimensions, and workload must satisfy MKL constraints.
    pub fn mkl_sparse_set_mm_hint(
        a: SparseMatrixHandle,
        operation: SparseOperation,
        descr: MatrixDescr,
        layout: SparseLayout,
        dense_matrix_size: MklInt,
        expected_calls: MklInt,
    ) -> SparseStatus;

    /// Records the expected triangular vector solve workload for `A`.
    /// `operation` and `descr` describe future calls; `expected_calls` is their expected
    /// count. Call `mkl_sparse_optimize` after setting hints to prepare execution.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` and its backing arrays must remain valid, and must not be accessed
    /// concurrently in a way that conflicts with MKL updating inspector metadata.
    /// The descriptor, dimensions, and workload must satisfy MKL constraints.
    pub fn mkl_sparse_set_sv_hint(
        a: SparseMatrixHandle,
        operation: SparseOperation,
        descr: MatrixDescr,
        expected_calls: MklInt,
    ) -> SparseStatus;

    /// Records the expected triangular matrix solve workload for `A`.
    /// `operation` and `descr` describe future calls; `expected_calls` is their expected
    /// count. Call `mkl_sparse_optimize` after setting hints to prepare execution.
    /// `layout` and `dense_matrix_size` describe the dense operand storage and its number of columns.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` and its backing arrays must remain valid, and must not be accessed
    /// concurrently in a way that conflicts with MKL updating inspector metadata.
    /// The descriptor, dimensions, and workload must satisfy MKL constraints.
    pub fn mkl_sparse_set_sm_hint(
        a: SparseMatrixHandle,
        operation: SparseOperation,
        descr: MatrixDescr,
        layout: SparseLayout,
        dense_matrix_size: MklInt,
        expected_calls: MklInt,
    ) -> SparseStatus;

    /// Sets the inspector memory policy for `A`.
    /// `policy` determines whether optimization may allocate new matrix representations.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` and its backing arrays must remain valid, and must not be accessed
    /// concurrently in a way that conflicts with MKL updating inspector metadata.
    /// The descriptor, dimensions, and workload must satisfy MKL constraints.
    pub fn mkl_sparse_set_memory_hint(
        a: SparseMatrixHandle,
        policy: SparseMemoryUsage,
    ) -> SparseStatus;

    /// Inspects `A` and prepares optimizations for the recorded workload hints.
    /// Later execution calls reuse this metadata; optimization may allocate internal storage.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` and all arrays borrowed by it must be valid. Prevent concurrent
    /// operations that conflict with changes to the handle or its internal storage.
    pub fn mkl_sparse_optimize(a: SparseMatrixHandle) -> SparseStatus;

    /// Computes `y := alpha * op(A) * x + beta * y`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `x` has as many elements as columns of `op(A)`; `y` has as many as rows.
    /// Dense vectors are contiguous; this API has no stride parameter.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_s_mv(
        operation: SparseOperation,
        alpha: f32,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const f32,
        beta: f32,
        y: *mut f32,
    ) -> SparseStatus;

    /// Computes `y := alpha * op(A) * x + beta * y`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `x` has as many elements as columns of `op(A)`; `y` has as many as rows.
    /// Dense vectors are contiguous; this API has no stride parameter.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_d_mv(
        operation: SparseOperation,
        alpha: f64,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const f64,
        beta: f64,
        y: *mut f64,
    ) -> SparseStatus;

    /// Computes `y := alpha * op(A) * x + beta * y`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `x` has as many elements as columns of `op(A)`; `y` has as many as rows.
    /// Dense vectors are contiguous; this API has no stride parameter.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_c_mv(
        operation: SparseOperation,
        alpha: MklComplex8,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const MklComplex8,
        beta: MklComplex8,
        y: *mut MklComplex8,
    ) -> SparseStatus;

    /// Computes `y := alpha * op(A) * x + beta * y`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `x` has as many elements as columns of `op(A)`; `y` has as many as rows.
    /// Dense vectors are contiguous; this API has no stride parameter.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_z_mv(
        operation: SparseOperation,
        alpha: MklComplex16,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const MklComplex16,
        beta: MklComplex16,
        y: *mut MklComplex16,
    ) -> SparseStatus;

    /// Computes `Y := alpha * op(A) * X + beta * Y`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `X` and `Y` have `columns` dense columns. Their row counts match the
    /// columns and rows of `op(A)`, respectively. `layout` selects storage order.
    /// `ldx`/`ldy` are distances between consecutive rows (row major) or columns
    /// (column major), at least the corresponding logical row/column length.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_s_mm(
        operation: SparseOperation,
        alpha: f32,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const f32,
        columns: MklInt,
        ldx: MklInt,
        beta: f32,
        y: *mut f32,
        ldy: MklInt,
    ) -> SparseStatus;

    /// Computes `Y := alpha * op(A) * X + beta * Y`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `X` and `Y` have `columns` dense columns. Their row counts match the
    /// columns and rows of `op(A)`, respectively. `layout` selects storage order.
    /// `ldx`/`ldy` are distances between consecutive rows (row major) or columns
    /// (column major), at least the corresponding logical row/column length.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_d_mm(
        operation: SparseOperation,
        alpha: f64,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const f64,
        columns: MklInt,
        ldx: MklInt,
        beta: f64,
        y: *mut f64,
        ldy: MklInt,
    ) -> SparseStatus;

    /// Computes `Y := alpha * op(A) * X + beta * Y`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `X` and `Y` have `columns` dense columns. Their row counts match the
    /// columns and rows of `op(A)`, respectively. `layout` selects storage order.
    /// `ldx`/`ldy` are distances between consecutive rows (row major) or columns
    /// (column major), at least the corresponding logical row/column length.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_c_mm(
        operation: SparseOperation,
        alpha: MklComplex8,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const MklComplex8,
        columns: MklInt,
        ldx: MklInt,
        beta: MklComplex8,
        y: *mut MklComplex8,
        ldy: MklInt,
    ) -> SparseStatus;

    /// Computes `Y := alpha * op(A) * X + beta * Y`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `X` and `Y` have `columns` dense columns. Their row counts match the
    /// columns and rows of `op(A)`, respectively. `layout` selects storage order.
    /// `ldx`/`ldy` are distances between consecutive rows (row major) or columns
    /// (column major), at least the corresponding logical row/column length.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_z_mm(
        operation: SparseOperation,
        alpha: MklComplex16,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const MklComplex16,
        columns: MklInt,
        ldx: MklInt,
        beta: MklComplex16,
        y: *mut MklComplex16,
        ldy: MklInt,
    ) -> SparseStatus;

    /// Computes `op(A) * y = alpha * x`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `A` must be square triangular; `descr.mode` and `descr.diag` select its
    /// stored triangle and explicit or implicit unit diagonal.
    /// `x` has as many elements as columns of `op(A)`; `y` has as many as rows.
    /// Dense vectors are contiguous; this API has no stride parameter.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_s_trsv(
        operation: SparseOperation,
        alpha: f32,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const f32,
        y: *mut f32,
    ) -> SparseStatus;

    /// Computes `op(A) * y = alpha * x`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `A` must be square triangular; `descr.mode` and `descr.diag` select its
    /// stored triangle and explicit or implicit unit diagonal.
    /// `x` has as many elements as columns of `op(A)`; `y` has as many as rows.
    /// Dense vectors are contiguous; this API has no stride parameter.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_d_trsv(
        operation: SparseOperation,
        alpha: f64,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const f64,
        y: *mut f64,
    ) -> SparseStatus;

    /// Computes `op(A) * y = alpha * x`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `A` must be square triangular; `descr.mode` and `descr.diag` select its
    /// stored triangle and explicit or implicit unit diagonal.
    /// `x` has as many elements as columns of `op(A)`; `y` has as many as rows.
    /// Dense vectors are contiguous; this API has no stride parameter.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_c_trsv(
        operation: SparseOperation,
        alpha: MklComplex8,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const MklComplex8,
        y: *mut MklComplex8,
    ) -> SparseStatus;

    /// Computes `op(A) * y = alpha * x`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `A` must be square triangular; `descr.mode` and `descr.diag` select its
    /// stored triangle and explicit or implicit unit diagonal.
    /// `x` has as many elements as columns of `op(A)`; `y` has as many as rows.
    /// Dense vectors are contiguous; this API has no stride parameter.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_z_trsv(
        operation: SparseOperation,
        alpha: MklComplex16,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        x: *const MklComplex16,
        y: *mut MklComplex16,
    ) -> SparseStatus;

    /// Computes `op(A) * Y = alpha * X`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `A` must be square triangular; `descr.mode` and `descr.diag` select its
    /// stored triangle and explicit or implicit unit diagonal.
    /// `X` and `Y` have `columns` dense columns. Their row counts match the
    /// columns and rows of `op(A)`, respectively. `layout` selects storage order.
    /// `ldx`/`ldy` are distances between consecutive rows (row major) or columns
    /// (column major), at least the corresponding logical row/column length.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_s_trsm(
        operation: SparseOperation,
        alpha: f32,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const f32,
        columns: MklInt,
        ldx: MklInt,
        y: *mut f32,
        ldy: MklInt,
    ) -> SparseStatus;

    /// Computes `op(A) * Y = alpha * X`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `A` must be square triangular; `descr.mode` and `descr.diag` select its
    /// stored triangle and explicit or implicit unit diagonal.
    /// `X` and `Y` have `columns` dense columns. Their row counts match the
    /// columns and rows of `op(A)`, respectively. `layout` selects storage order.
    /// `ldx`/`ldy` are distances between consecutive rows (row major) or columns
    /// (column major), at least the corresponding logical row/column length.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_d_trsm(
        operation: SparseOperation,
        alpha: f64,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const f64,
        columns: MklInt,
        ldx: MklInt,
        y: *mut f64,
        ldy: MklInt,
    ) -> SparseStatus;

    /// Computes `op(A) * Y = alpha * X`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `A` must be square triangular; `descr.mode` and `descr.diag` select its
    /// stored triangle and explicit or implicit unit diagonal.
    /// `X` and `Y` have `columns` dense columns. Their row counts match the
    /// columns and rows of `op(A)`, respectively. `layout` selects storage order.
    /// `ldx`/`ldy` are distances between consecutive rows (row major) or columns
    /// (column major), at least the corresponding logical row/column length.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_c_trsm(
        operation: SparseOperation,
        alpha: MklComplex8,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const MklComplex8,
        columns: MklInt,
        ldx: MklInt,
        y: *mut MklComplex8,
        ldy: MklInt,
    ) -> SparseStatus;

    /// Computes `op(A) * Y = alpha * X`.
    /// `operation` selects `op(A)` and `descr` describes the sparse structure.
    /// `A` must be square triangular; `descr.mode` and `descr.diag` select its
    /// stored triangle and explicit or implicit unit diagonal.
    /// `X` and `Y` have `columns` dense columns. Their row counts match the
    /// columns and rows of `op(A)`, respectively. `layout` selects storage order.
    /// `ldx`/`ldy` are distances between consecutive rows (row major) or columns
    /// (column major), at least the corresponding logical row/column length.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_z_trsm(
        operation: SparseOperation,
        alpha: MklComplex16,
        a: SparseMatrixHandle,
        descr: MatrixDescr,
        layout: SparseLayout,
        x: *const MklComplex16,
        columns: MklInt,
        ldx: MklInt,
        y: *mut MklComplex16,
        ldy: MklInt,
    ) -> SparseStatus;

    /// Computes `C := alpha * op(A) + B` with a sparse result handle in `C`.
    /// Inputs must have compatible dimensions and matching scalar precision.
    /// Destroy the result handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// Input handles and their backing arrays must be valid and use this ABI.
    /// `C` must be writable and must not overwrite an unrelated live handle.
    /// Dimensions, formats, descriptors, and operations must satisfy MKL requirements.
    pub fn mkl_sparse_s_add(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        alpha: f32,
        b: SparseMatrixHandle,
        c: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Computes `C := alpha * op(A) + B` with a sparse result handle in `C`.
    /// Inputs must have compatible dimensions and matching scalar precision.
    /// Destroy the result handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// Input handles and their backing arrays must be valid and use this ABI.
    /// `C` must be writable and must not overwrite an unrelated live handle.
    /// Dimensions, formats, descriptors, and operations must satisfy MKL requirements.
    pub fn mkl_sparse_d_add(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        alpha: f64,
        b: SparseMatrixHandle,
        c: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Computes `C := alpha * op(A) + B` with a sparse result handle in `C`.
    /// Inputs must have compatible dimensions and matching scalar precision.
    /// Destroy the result handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// Input handles and their backing arrays must be valid and use this ABI.
    /// `C` must be writable and must not overwrite an unrelated live handle.
    /// Dimensions, formats, descriptors, and operations must satisfy MKL requirements.
    pub fn mkl_sparse_c_add(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        alpha: MklComplex8,
        b: SparseMatrixHandle,
        c: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Computes `C := alpha * op(A) + B` with a sparse result handle in `C`.
    /// Inputs must have compatible dimensions and matching scalar precision.
    /// Destroy the result handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// Input handles and their backing arrays must be valid and use this ABI.
    /// `C` must be writable and must not overwrite an unrelated live handle.
    /// Dimensions, formats, descriptors, and operations must satisfy MKL requirements.
    pub fn mkl_sparse_z_add(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        alpha: MklComplex16,
        b: SparseMatrixHandle,
        c: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Computes `C := op(A) * B`.
    /// `A` and `B` are sparse matrices; `C` is dense in `layout` order.
    /// `ldc` is the distance between consecutive dense rows (row major) or columns
    /// (column major), at least the corresponding logical row/column length.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_s_spmmd(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        b: SparseMatrixHandle,
        layout: SparseLayout,
        c: *mut f32,
        ldc: MklInt,
    ) -> SparseStatus;

    /// Computes `C := op(A) * B`.
    /// `A` and `B` are sparse matrices; `C` is dense in `layout` order.
    /// `ldc` is the distance between consecutive dense rows (row major) or columns
    /// (column major), at least the corresponding logical row/column length.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_d_spmmd(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        b: SparseMatrixHandle,
        layout: SparseLayout,
        c: *mut f64,
        ldc: MklInt,
    ) -> SparseStatus;

    /// Computes `C := op(A) * B`.
    /// `A` and `B` are sparse matrices; `C` is dense in `layout` order.
    /// `ldc` is the distance between consecutive dense rows (row major) or columns
    /// (column major), at least the corresponding logical row/column length.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_c_spmmd(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        b: SparseMatrixHandle,
        layout: SparseLayout,
        c: *mut MklComplex8,
        ldc: MklInt,
    ) -> SparseStatus;

    /// Computes `C := op(A) * B`.
    /// `A` and `B` are sparse matrices; `C` is dense in `layout` order.
    /// `ldc` is the distance between consecutive dense rows (row major) or columns
    /// (column major), at least the corresponding logical row/column length.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// All sparse handles must be live, of the matching scalar precision and ABI,
    /// with valid backing arrays. Dense input/output pointers must cover the dimensions
    /// and leading dimensions of the operation; outputs must be writable. Prevent
    /// conflicting access and unsupported overlap. The operation, format, and
    /// descriptor must be supported by the linked MKL version.
    pub fn mkl_sparse_z_spmmd(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        b: SparseMatrixHandle,
        layout: SparseLayout,
        c: *mut MklComplex16,
        ldc: MklInt,
    ) -> SparseStatus;

    /// Updates the existing matrix entry at (`row`, `col`) to `value`.
    /// Indices use the base of the matrix. This operation does not insert a new entry.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be a live handle of the matching precision with writable value
    /// storage. The entry at (`row`, `col`) must already exist. Prevent
    /// conflicting access to the matrix arrays.
    pub fn mkl_sparse_s_set_value(
        a: SparseMatrixHandle,
        row: MklInt,
        col: MklInt,
        value: f32,
    ) -> SparseStatus;

    /// Updates the existing matrix entry at (`row`, `col`) to `value`.
    /// Indices use the base of the matrix. This operation does not insert a new entry.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be a live handle of the matching precision with writable value
    /// storage. The entry at (`row`, `col`) must already exist. Prevent
    /// conflicting access to the matrix arrays.
    pub fn mkl_sparse_d_set_value(
        a: SparseMatrixHandle,
        row: MklInt,
        col: MklInt,
        value: f64,
    ) -> SparseStatus;

    /// Updates the existing matrix entry at (`row`, `col`) to `value`.
    /// Indices use the base of the matrix. This operation does not insert a new entry.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be a live handle of the matching precision with writable value
    /// storage. The entry at (`row`, `col`) must already exist. Prevent
    /// conflicting access to the matrix arrays.
    pub fn mkl_sparse_c_set_value(
        a: SparseMatrixHandle,
        row: MklInt,
        col: MklInt,
        value: MklComplex8,
    ) -> SparseStatus;

    /// Updates the existing matrix entry at (`row`, `col`) to `value`.
    /// Indices use the base of the matrix. This operation does not insert a new entry.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be a live handle of the matching precision with writable value
    /// storage. The entry at (`row`, `col`) must already exist. Prevent
    /// conflicting access to the matrix arrays.
    pub fn mkl_sparse_z_set_value(
        a: SparseMatrixHandle,
        row: MklInt,
        col: MklInt,
        value: MklComplex16,
    ) -> SparseStatus;

    /// Replaces values of a BSR matrix without changing its sparsity structure.
    /// MKL currently supports full-matrix updates only: pass null `indx` and
    /// `indy`, set `nvalues` to zero or the actual number of scalar values, and
    /// supply all block values in the existing storage order. CSR and selected
    /// entry updates report `NOT_SUPPORTED`; use `set_value` for a single entry.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be a live handle of the matching precision with writable value
    /// storage. `values` must cover every stored scalar in every BSR block, even
    /// when `nvalues` is zero. MKL can retain this pointer instead of copying;
    /// keep the replacement array alive at a stable address while the handle
    /// references it. Prevent conflicting access to matrix storage.
    pub fn mkl_sparse_s_update_values(
        a: SparseMatrixHandle,
        nvalues: MklInt,
        indx: *const MklInt,
        indy: *const MklInt,
        values: *mut f32,
    ) -> SparseStatus;

    /// Replaces values of a BSR matrix without changing its sparsity structure.
    /// MKL currently supports full-matrix updates only: pass null `indx` and
    /// `indy`, set `nvalues` to zero or the actual number of scalar values, and
    /// supply all block values in the existing storage order. CSR and selected
    /// entry updates report `NOT_SUPPORTED`; use `set_value` for a single entry.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be a live handle of the matching precision with writable value
    /// storage. `values` must cover every stored scalar in every BSR block, even
    /// when `nvalues` is zero. MKL can retain this pointer instead of copying;
    /// keep the replacement array alive at a stable address while the handle
    /// references it. Prevent conflicting access to matrix storage.
    pub fn mkl_sparse_d_update_values(
        a: SparseMatrixHandle,
        nvalues: MklInt,
        indx: *const MklInt,
        indy: *const MklInt,
        values: *mut f64,
    ) -> SparseStatus;

    /// Replaces values of a BSR matrix without changing its sparsity structure.
    /// MKL currently supports full-matrix updates only: pass null `indx` and
    /// `indy`, set `nvalues` to zero or the actual number of scalar values, and
    /// supply all block values in the existing storage order. CSR and selected
    /// entry updates report `NOT_SUPPORTED`; use `set_value` for a single entry.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be a live handle of the matching precision with writable value
    /// storage. `values` must cover every stored scalar in every BSR block, even
    /// when `nvalues` is zero. MKL can retain this pointer instead of copying;
    /// keep the replacement array alive at a stable address while the handle
    /// references it. Prevent conflicting access to matrix storage.
    pub fn mkl_sparse_c_update_values(
        a: SparseMatrixHandle,
        nvalues: MklInt,
        indx: *const MklInt,
        indy: *const MklInt,
        values: *mut MklComplex8,
    ) -> SparseStatus;

    /// Replaces values of a BSR matrix without changing its sparsity structure.
    /// MKL currently supports full-matrix updates only: pass null `indx` and
    /// `indy`, set `nvalues` to zero or the actual number of scalar values, and
    /// supply all block values in the existing storage order. CSR and selected
    /// entry updates report `NOT_SUPPORTED`; use `set_value` for a single entry.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// `A` must be a live handle of the matching precision with writable value
    /// storage. `values` must cover every stored scalar in every BSR block, even
    /// when `nvalues` is zero. MKL can retain this pointer instead of copying;
    /// keep the replacement array alive at a stable address while the handle
    /// references it. Prevent conflicting access to matrix storage.
    pub fn mkl_sparse_z_update_values(
        a: SparseMatrixHandle,
        nvalues: MklInt,
        indx: *const MklInt,
        indy: *const MklInt,
        values: *mut MklComplex16,
    ) -> SparseStatus;

    /// Computes `C := op(A) * B` with a sparse result handle in `C`.
    /// Inputs must have compatible dimensions and matching scalar precision.
    /// Destroy the result handle after use.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// Input handles and their backing arrays must be valid and use this ABI.
    /// `C` must be writable and must not overwrite an unrelated live handle.
    /// Dimensions, formats, descriptors, and operations must satisfy MKL requirements.
    pub fn mkl_sparse_spmm(
        operation: SparseOperation,
        a: SparseMatrixHandle,
        b: SparseMatrixHandle,
        c: *mut SparseMatrixHandle,
    ) -> SparseStatus;

    /// Computes `C := op(A) * op(B)` with a sparse result handle in `C`.
    /// Inputs must have compatible dimensions and matching scalar precision.
    /// Destroy the result handle after use.
    /// `transA`/`transB` and their descriptors select the operands. `request` can
    /// compute everything in one call (`FullMult`) or first build offsets (`NnzCount`)
    /// and then complete a product (`FinalizeMult`). Finalization reuses the handle in
    /// `C` from the earlier stage; value-only recomputation requires unchanged structure.
    /// Structure-only requests leave values unavailable for numerical execution.
    ///
    /// Requires Intel oneMKL; integer parameters and arrays use the selected MKL ABI.
    ///
    /// # Safety
    /// Input handles and their backing arrays must be valid and use this ABI.
    /// `C` must be writable and must not overwrite an unrelated live handle.
    /// Dimensions, formats, descriptors, and operations must satisfy MKL requirements.
    /// For staged execution, pass the correct previous result handle and preserve
    /// input structure between stages. Do not use incomplete outputs numerically.
    pub fn mkl_sparse_sp2m(
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
