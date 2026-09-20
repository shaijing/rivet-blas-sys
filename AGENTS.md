# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# macOS (default: Accelerate framework)
cargo build

# macOS with OpenBLAS (requires pkg-config)
cargo build -F openblas

# Linux with Intel MKL (requires MKLROOT environment variable)
cargo build -F intel-mkl

# Linux with OpenBLAS
cargo build -F openblas

# Windows with Intel MKL (requires MKLROOT environment variable)
cargo build -F intel-mkl

# Windows with OpenBLAS (via vcpkg)
cargo build -F openblas

# Run examples
cargo run --release --example mat_blas

# Run tests (requires BLAS implementation feature)
cargo test -F intel-mkl
```

## Architecture

This crate provides Rust FFI bindings to CBLAS (C interface to BLAS). The structure:

- `src/lib.rs` - Crate root, exports `blas_types` and `cblas` modules
- `src/blas_types.rs` - Rust-native type aliases (`BlasInt`, `BlasFloat`, etc.)
- `src/cblas/mod.rs` - Re-exports all CBLAS functions via `prelude`
- `src/cblas/cblas_types.rs` - C-compatible types (`CBlasInt`, `CBlasLayout`, `CBlasTranspose`, etc.) and enums for BLAS parameters
- `src/cblas/cblas_level_one.rs` - BLAS Level 1 (vector-vector ops: dot, axpy, scal, nrm2, etc.)
- `src/cblas/cblas_level_two.rs` - BLAS Level 2 (matrix-vector ops: gemv, ger, etc.)
- `src/cblas/cblas_level_three.rs` - BLAS Level 3 (matrix-matrix ops: gemm, symm, trmm, etc.)
- `build.rs` - Platform-specific library linking (vcpkg/pkg-config on Windows, MKL via MKLROOT)
- `examples/mat_blas.rs` - Example wrappers and usage

## Feature Flags

- `ilp64` (default) - 64-bit integer API for large arrays
- `lp64` - 32-bit integer API
- `intel-mkl` - Link against Intel MKL
- `openblas` - Link against OpenBLAS
- `static` - Static linking
- `system` - Use system-provided BLAS

## BLAS Implementation Requirements

- **Intel MKL on Windows**: Set `MKLROOT` environment variable to MKL installation directory
- **OpenBLAS on Windows**: Install via vcpkg (`vcpkg install openblas:x64-windows`)

## API Pattern

All functions are `unsafe extern "C"` FFI bindings matching CBLAS naming conventions:
- Prefix: `cblas_`
- Precision indicator: `s` (f32), `d` (f64), `c` (complex f32), `z` (complex f64)
- Operation: `gemm`, `dot`, `axpy`, etc.

Example: `cblas_dgemm` = double-precision general matrix multiply