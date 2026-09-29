//! C-compatible types from Intel oneMKL's `mkl_types.h` and `mkl_spblas.h`.

use std::ffi::c_int;

/// MKL_INT: 32 bits for LP64 (including SDL), 64 bits for ILP64.
pub type MklInt = crate::cblas::cblas_types::CBlasInt;
/// MKL_INT64: always 64 bits, independent of the ordinary interface.
pub type MklInt64 = crate::cblas::cblas_types::MklCBlasInt64;

/// MKL_Complex8. Sparse BLAS passes complex scalars by value.
#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct MklComplex8 {
    /// Real component.
    pub real: f32,
    /// Imaginary component.
    pub imag: f32,
}

/// MKL_Complex16. Sparse BLAS passes complex scalars by value.
#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct MklComplex16 {
    /// Real component.
    pub real: f64,
    /// Imaginary component.
    pub imag: f64,
}

/// Opaque MKL sparse matrix implementation; only use pointers to this type.
#[repr(C)]
pub struct SparseMatrix {
    _private: [u8; 0],
}

/// `sparse_matrix_t`. Creation does not transfer ownership of input arrays.
/// Destroy successful output handles with the matching Sparse BLAS interface.
pub type SparseMatrixHandle = *mut SparseMatrix;

/// `sparse_status_t`. A transparent integer also preserves future status codes.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SparseStatus(pub c_int);

impl SparseStatus {
    /// The operation succeeded.
    pub const SUCCESS: Self = Self(0);
    /// An empty handle or matrix array was encountered.
    pub const NOT_INITIALIZED: Self = Self(1);
    /// Internal allocation failed.
    pub const ALLOC_FAILED: Self = Self(2);
    /// An input parameter is invalid.
    pub const INVALID_VALUE: Self = Self(3);
    /// Execution failed, for example because a triangular diagonal is zero.
    pub const EXECUTION_FAILED: Self = Self(4);
    /// An internal error occurred.
    pub const INTERNAL_ERROR: Self = Self(5);
    /// The requested type, format, or operation is unsupported.
    pub const NOT_SUPPORTED: Self = Self(6);
}

/// `sparse_operation_t`; these values differ from CBLAS transpose values.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SparseOperation {
    /// Use A unchanged.
    NonTranspose = 10,
    /// Use A transposed.
    Transpose = 11,
    /// Use A conjugate transposed.
    ConjugateTranspose = 12,
}

/// `sparse_matrix_type_t`.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SparseMatrixType {
    /// General matrix.
    General = 20,
    /// Symmetric matrix, stored by triangle.
    Symmetric = 21,
    /// Hermitian matrix, stored by triangle.
    Hermitian = 22,
    /// Triangular matrix.
    Triangular = 23,
    /// Only the diagonal is processed.
    Diagonal = 24,
    /// Block triangular matrix (BSR).
    BlockTriangular = 25,
    /// Only diagonal blocks are processed (BSR).
    BlockDiagonal = 26,
}

/// `sparse_index_base_t`; applies to both offsets and row/column indices.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SparseIndexBase {
    /// C-style zero-based indices.
    Zero = 0,
    /// Fortran-style one-based indices.
    One = 1,
}

/// `sparse_fill_mode_t`.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SparseFillMode {
    /// Lower triangle is stored.
    Lower = 40,
    /// Upper triangle is stored.
    Upper = 41,
    /// Full matrix is stored, when supported by the operation.
    Full = 42,
}

/// `sparse_diag_type_t`.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SparseDiagType {
    /// Use the explicitly stored diagonal.
    NonUnit = 50,
    /// Treat the diagonal as ones.
    Unit = 51,
}

/// `sparse_layout_t`; used for dense operands and values within BSR blocks.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SparseLayout {
    /// Consecutive elements belong to a row.
    RowMajor = 101,
    /// Consecutive elements belong to a column.
    ColumnMajor = 102,
}

/// `sparse_memory_usage_t` for inspector optimization hints.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SparseMemoryUsage {
    /// Avoid allocation for new matrix values/structure; auxiliary data is allowed.
    None = 80,
    /// Permit conversion to internal formats for optimization.
    Aggressive = 81,
}

/// `sparse_request_t` for staged sparse matrix multiplication.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SparseRequest {
    /// Compute structure and values in one call.
    FullMult = 90,
    /// Allocate and compute row/column offsets, without indices or values.
    NnzCount = 91,
    /// Complete a previous stage, or recompute values for unchanged structure.
    FinalizeMult = 92,
    /// Compute complete structure without values.
    FullMultNoVal = 93,
    /// Complete structure without values after NnzCount.
    FinalizeMultNoVal = 94,
}

/// `struct matrix_descr`, passed by value to MKL.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MatrixDescr {
    /// Matrix structure interpreted by the operation.
    pub matrix_type: SparseMatrixType,
    /// Stored triangle for triangular, symmetric, or Hermitian matrices.
    pub mode: SparseFillMode,
    /// Whether the diagonal is implicitly one.
    pub diag: SparseDiagType,
}

impl Default for MatrixDescr {
    /// A general matrix; mode and diagonal are initialized but ignored by MKL.
    fn default() -> Self {
        Self {
            matrix_type: SparseMatrixType::General,
            mode: SparseFillMode::Lower,
            diag: SparseDiagType::NonUnit,
        }
    }
}
