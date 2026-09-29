//! Opt-in Linux ABI audit against the installed MKL C headers (GCC or Clang).
#![cfg(all(rivet_blas_mkl, target_os = "linux"))]

use rivet_blas_sys::mkl::types::*;
use std::{
    fmt::Write,
    mem::{align_of, offset_of, size_of},
    path::PathBuf,
    process::Command,
};

fn c_type(rust: &str) -> String {
    let rust = rust.trim();
    if let Some(inner) = rust.strip_prefix("*mut ") {
        return format!("{} *", c_type(inner));
    }
    if let Some(inner) = rust.strip_prefix("*const ") {
        return format!("const {} *", c_type(inner));
    }
    match rust {
        "MklInt" => "MKL_INT",
        "MklInt64" => "MKL_INT64",
        "MklComplex8" => "MKL_Complex8",
        "MklComplex16" => "MKL_Complex16",
        "SparseMatrixHandle" => "sparse_matrix_t",
        "MatrixDescr" => "struct matrix_descr",
        "SparseOperation" => "sparse_operation_t",
        "SparseIndexBase" => "sparse_index_base_t",
        "SparseLayout" => "sparse_layout_t",
        "SparseMemoryUsage" => "sparse_memory_usage_t",
        "SparseRequest" => "sparse_request_t",
        "f32" => "float",
        "f64" => "double",
        other => panic!("Unrecognized Rust FFI parameter type: {other}"),
    }
    .to_owned()
}

struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "requires GCC/Clang and MKL_INCLUDE_DIR pointing to oneMKL C headers"]
fn all_signatures_layouts_and_enum_values_match_mkl_headers() {
    let include = std::env::var_os("MKL_INCLUDE_DIR")
        .expect("Set MKL_INCLUDE_DIR to the directory containing mkl_spblas.h");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut source = String::from("#include <stddef.h>\n#include <mkl_spblas.h>\n");
    let mut count = 0;
    for module in ["sparse", "sparse_64"] {
        let declarations =
            std::fs::read_to_string(root.join(format!("src/mkl/{module}.rs"))).unwrap();
        for declaration in declarations.split("pub fn ").skip(1) {
            let (name, rest) = declaration.split_once('(').unwrap();
            let (params, tail) = rest.split_once(')').unwrap();
            assert!(tail.trim_start().starts_with("-> SparseStatus;"));
            let params: Vec<_> = params
                .split(',')
                .filter(|p| !p.trim().is_empty())
                .map(|p| c_type(p.split_once(':').unwrap().1))
                .collect();
            writeln!(
                source,
                "typedef sparse_status_t (*expected_{name})({});",
                params.join(", ")
            )
            .unwrap();
            writeln!(source, "_Static_assert(__builtin_types_compatible_p(__typeof__(&{name}), expected_{name}), \"{name} signature\");").unwrap();
            count += 1;
        }
    }
    assert_eq!(count, 158, "Audit every exported Sparse BLAS declaration");

    macro_rules! layout {
        ($rust:ty, $c:expr) => {{
            writeln!(
                source,
                "_Static_assert(sizeof({}) == {}, \"{} size\");",
                $c,
                size_of::<$rust>(),
                $c
            )
            .unwrap();
            writeln!(
                source,
                "_Static_assert(_Alignof({}) == {}, \"{} alignment\");",
                $c,
                align_of::<$rust>(),
                $c
            )
            .unwrap();
        }};
    }
    layout!(MklInt, "MKL_INT");
    layout!(MklInt64, "MKL_INT64");
    layout!(MklComplex8, "MKL_Complex8");
    layout!(MklComplex16, "MKL_Complex16");
    layout!(SparseStatus, "sparse_status_t");
    layout!(SparseMatrixHandle, "sparse_matrix_t");
    layout!(SparseOperation, "sparse_operation_t");
    layout!(SparseMatrixType, "sparse_matrix_type_t");
    layout!(SparseIndexBase, "sparse_index_base_t");
    layout!(SparseFillMode, "sparse_fill_mode_t");
    layout!(SparseDiagType, "sparse_diag_type_t");
    layout!(SparseLayout, "sparse_layout_t");
    layout!(SparseMemoryUsage, "sparse_memory_usage_t");
    layout!(SparseRequest, "sparse_request_t");
    layout!(MatrixDescr, "struct matrix_descr");

    for (c_type, c_field, offset) in [
        ("MKL_Complex8", "real", offset_of!(MklComplex8, real)),
        ("MKL_Complex8", "imag", offset_of!(MklComplex8, imag)),
        ("MKL_Complex16", "real", offset_of!(MklComplex16, real)),
        ("MKL_Complex16", "imag", offset_of!(MklComplex16, imag)),
        (
            "struct matrix_descr",
            "type",
            offset_of!(MatrixDescr, matrix_type),
        ),
        ("struct matrix_descr", "mode", offset_of!(MatrixDescr, mode)),
        ("struct matrix_descr", "diag", offset_of!(MatrixDescr, diag)),
    ] {
        writeln!(source, "_Static_assert(offsetof({c_type}, {c_field}) == {offset}, \"{c_type}.{c_field} offset\");").unwrap();
    }
    macro_rules! value {
        ($rust:expr, $c:ident) => {
            writeln!(
                source,
                "_Static_assert({} == {}, \"{} value\");",
                stringify!($c),
                $rust,
                stringify!($c)
            )
            .unwrap();
        };
    }
    value!(SparseStatus::SUCCESS.0, SPARSE_STATUS_SUCCESS);
    value!(
        SparseStatus::NOT_INITIALIZED.0,
        SPARSE_STATUS_NOT_INITIALIZED
    );
    value!(SparseStatus::ALLOC_FAILED.0, SPARSE_STATUS_ALLOC_FAILED);
    value!(SparseStatus::INVALID_VALUE.0, SPARSE_STATUS_INVALID_VALUE);
    value!(
        SparseStatus::EXECUTION_FAILED.0,
        SPARSE_STATUS_EXECUTION_FAILED
    );
    value!(SparseStatus::INTERNAL_ERROR.0, SPARSE_STATUS_INTERNAL_ERROR);
    value!(SparseStatus::NOT_SUPPORTED.0, SPARSE_STATUS_NOT_SUPPORTED);
    value!(
        SparseOperation::NonTranspose as i32,
        SPARSE_OPERATION_NON_TRANSPOSE
    );
    value!(
        SparseOperation::Transpose as i32,
        SPARSE_OPERATION_TRANSPOSE
    );
    value!(
        SparseOperation::ConjugateTranspose as i32,
        SPARSE_OPERATION_CONJUGATE_TRANSPOSE
    );
    value!(SparseMatrixType::General as i32, SPARSE_MATRIX_TYPE_GENERAL);
    value!(
        SparseMatrixType::Symmetric as i32,
        SPARSE_MATRIX_TYPE_SYMMETRIC
    );
    value!(
        SparseMatrixType::Hermitian as i32,
        SPARSE_MATRIX_TYPE_HERMITIAN
    );
    value!(
        SparseMatrixType::Triangular as i32,
        SPARSE_MATRIX_TYPE_TRIANGULAR
    );
    value!(
        SparseMatrixType::Diagonal as i32,
        SPARSE_MATRIX_TYPE_DIAGONAL
    );
    value!(
        SparseMatrixType::BlockTriangular as i32,
        SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR
    );
    value!(
        SparseMatrixType::BlockDiagonal as i32,
        SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL
    );
    value!(SparseIndexBase::Zero as i32, SPARSE_INDEX_BASE_ZERO);
    value!(SparseIndexBase::One as i32, SPARSE_INDEX_BASE_ONE);
    value!(SparseFillMode::Lower as i32, SPARSE_FILL_MODE_LOWER);
    value!(SparseFillMode::Upper as i32, SPARSE_FILL_MODE_UPPER);
    value!(SparseFillMode::Full as i32, SPARSE_FILL_MODE_FULL);
    value!(SparseDiagType::NonUnit as i32, SPARSE_DIAG_NON_UNIT);
    value!(SparseDiagType::Unit as i32, SPARSE_DIAG_UNIT);
    value!(SparseLayout::RowMajor as i32, SPARSE_LAYOUT_ROW_MAJOR);
    value!(SparseLayout::ColumnMajor as i32, SPARSE_LAYOUT_COLUMN_MAJOR);
    value!(SparseMemoryUsage::None as i32, SPARSE_MEMORY_NONE);
    value!(
        SparseMemoryUsage::Aggressive as i32,
        SPARSE_MEMORY_AGGRESSIVE
    );
    value!(SparseRequest::FullMult as i32, SPARSE_STAGE_FULL_MULT);
    value!(SparseRequest::NnzCount as i32, SPARSE_STAGE_NNZ_COUNT);
    value!(
        SparseRequest::FinalizeMult as i32,
        SPARSE_STAGE_FINALIZE_MULT
    );
    value!(
        SparseRequest::FullMultNoVal as i32,
        SPARSE_STAGE_FULL_MULT_NO_VAL
    );
    value!(
        SparseRequest::FinalizeMultNoVal as i32,
        SPARSE_STAGE_FINALIZE_MULT_NO_VAL
    );

    let path = std::env::temp_dir().join(format!(
        "rivet-mkl-header-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&path).unwrap();
    let temp = TempDir(path);
    let file = temp.0.join("audit.c");
    std::fs::write(&file, source).unwrap();
    let mut compiler = Command::new(std::env::var_os("CC").unwrap_or_else(|| "cc".into()));
    compiler
        .args([
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            "-I",
        ])
        .arg(include);
    if size_of::<MklInt>() == 8 {
        compiler.arg("-DMKL_ILP64");
    }
    let result = compiler
        .arg(file)
        .output()
        .expect("Run a GCC-compatible C compiler via CC");
    assert!(
        result.status.success(),
        "C header ABI audit failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
