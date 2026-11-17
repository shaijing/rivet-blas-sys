# rivet-blas-sys

This crate provides low-level Rust FFI bindings to CBLAS (C interface to BLAS)
and Intel oneMKL Inspector-Executor Sparse BLAS.

## Features

Features describe complete native link configurations. Enable exactly one
configuration feature; do not combine backend, ABI, static, or threading
features. The default is `flexiblas-dynamic-lp64`.

- OpenBLAS: `openblas-{dynamic,static}-{ilp64,lp64}`.
- FlexiBLAS: `flexiblas-dynamic-{ilp64,lp64}` (dynamic linking only).
- Intel oneMKL: `mkl-{dynamic,static}-{ilp64,lp64}-{gomp,iomp,seq,tbb}`.
- Intel oneMKL SDL: `mkl-sdl` (the oneAPI SDL runtime configuration, using the
  default LP64 ordinary ABI).
- Netlib: `netlib-{dynamic,static}-lp64`.
- Apple Accelerate: `accelerate` (macOS LP64).

FlexiBLAS is supported only on Linux. Intel oneMKL is supported only on
Windows and Linux.

Ordinary `cblas_*` functions use the ABI encoded by the selected feature.
Intel MKL's explicit 64-bit symbols are exposed separately under
`cblas_level_one_mkl_64`, `cblas_level_two_mkl_64`, and
`cblas_level_three_mkl_64`, and always use `MKL_INT64`/`MKL_UINT64`.

Backend-specific extensions are kept out of the common modules:

- `cblas_level_one_mkl` and `cblas_level_three_mkl` contain Intel MKL-only APIs.
- `cblas_level_one_openblas` and `cblas_level_three_openblas` contain
  OpenBLAS/FlexiBLAS extensions.
- `mkl::sparse` contains ordinary Inspector-Executor Sparse BLAS APIs.
- `mkl::sparse_64` contains actual `mkl_sparse_*_64` symbols, using 64-bit indices
  independently of the ordinary ABI. These bindings were validated against
  oneMKL 2026.1; using them requires a version exporting the selected symbols.

The non-standard `cblas_*geadd` extension has backend-specific ABIs: the
OpenBLAS form accepts transpose arguments, while the FlexiBLAS form does not.
The Rust declaration selected by the configuration feature matches that
backend's ABI.

An `openblas-*` configuration links OpenBLAS, while a `flexiblas-*`
configuration links FlexiBLAS. Do not use `--all-features` for a normal build,
because configuration features are intentionally mutually exclusive.

When selecting another backend, disable the defaults explicitly, for example:

```bash
cargo test --no-default-features -F mkl-dynamic-ilp64-seq
cargo test --no-default-features -F mkl-dynamic-lp64-iomp
cargo test --no-default-features -F flexiblas-dynamic-ilp64
cargo test --no-default-features -F openblas-dynamic-lp64
```

## Intel oneMKL Sparse BLAS

Use `rivet_blas_sys::mkl::prelude::*` for ordinary Sparse BLAS declarations and
types, or import `mkl::sparse_64` explicitly for suffixed entry points. No
additional feature is needed beyond a complete MKL configuration.

The bindings cover all four scalar precisions (`s`, `d`, `c`, `z`) for CSR,
CSC, COO, and BSR creation/export; copy, format conversion, ordering, value
updates, workload hints and optimization; matrix-vector/matrix multiplication,
triangular solves, addition, and sparse matrix products, including staged
`sp2m`. Supported format/operation combinations depend on MKL. In particular,
`update_values` currently supports only full BSR updates with null index arrays;
use `set_value` to change a single existing entry. Legacy deprecated sparse
Level 2/3 interfaces and sparse solvers are not included.

All calls are unsafe. Creation borrows the input arrays: keep them alive at
stable addresses while the handle references them. Ordering and value updates
can mutate those arrays. Export returns borrowed pointers, never ownership;
do not free them or use them after their backing storage changes or is destroyed.
Destroy each successfully created result handle exactly once. Use a single ABI
for a handle's entire lifetime, including analysis, export, and destruction.
Full BSR `update_values` can retain the replacement array pointer; keep that
array alive at a stable address while the handle references it. `copy` currently
creates an independent deep copy of the entire matrix and ignores its descriptor.

`MklInt` follows the selected LP64/ILP64 interface; `MklInt64` always uses 64 bits.
Sparse descriptors and enums are distinct from CBLAS enums. Complex scalars
use `#[repr(C)]` `MklComplex8`/`MklComplex16` structures passed by value.

**SDL ABI correction:** `mkl-sdl` now uses 32-bit ordinary integers to match
MKL's default LP64 interface, replacing this crate's earlier ILP64 mapping.
Keep `MKL_INTERFACE_LAYER` unset or set to `LP64` when using ordinary SDL calls;
do not switch the process runtime to ILP64. Migrate callers needing 64-bit
ordinary indices to an explicit `mkl-*-ilp64-*` configuration, or use the
separate `*_64` APIs with 64-bit arrays and handles.

For Linux, expose the matching pkg-config profile and runtime libraries:

```bash
export PKG_CONFIG_PATH=/opt/intel/oneapi/mkl/latest/lib/pkgconfig
export LD_LIBRARY_PATH="/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/latest/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cargo test --no-default-features -F mkl-dynamic-lp64-seq --test test_mkl_sparse
cargo test --no-default-features -F mkl-dynamic-ilp64-seq --test test_mkl_sparse
# Optional Linux audit of all 158 C signatures, layouts, and enum values (GCC/Clang).
MKL_INCLUDE_DIR=/opt/intel/oneapi/mkl/latest/include \
  cargo test --no-default-features -F mkl-dynamic-lp64-seq --test test_mkl_sparse_header -- --ignored
MKL_INCLUDE_DIR=/opt/intel/oneapi/mkl/latest/include \
  cargo test --no-default-features -F mkl-dynamic-ilp64-seq --test test_mkl_sparse_header -- --ignored
cargo run --release --no-default-features -F mkl-dynamic-lp64-seq --example mkl_sparse
cargo bench --no-default-features -F mkl-dynamic-lp64-seq --bench cblas -- mkl/sparse
```

The sparse tests use independent Rust dense reference calculations for all four
precisions in both interfaces, covering all bound symbols, transposes, layouts,
padding, descriptors, format round trips, staged products, and handle lifetimes.
The opt-in header audit compiles C static assertions without adding a C compiler
requirement to normal builds. Set `CC` to a GCC-compatible compiler if needed.
The sparse benchmark measures repeated execution after inspector optimization.
See the [Intel Sparse BLAS reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/inspector-executor-sparse-blas-routines.html)
for supported operations and format restrictions.

## Supported Platforms

The test matrix uses Fedora 44 on Linux, macOS 26 Tahoe, and Windows 11.

The Fedora 44 local test environment uses these backend versions:

| Backend | Version |
| :------ | :------ |
| FlexiBLAS | 3.5.0 |
| OpenBLAS | 0.3.29 |
| Intel oneMKL | 2026.1.0 |

The local macOS test environment uses these backend versions:

| Backend | Version |
| :------ | :------ |
| Accelerate | 1.11 |
| OpenBLAS | 0.3.34 |

| BLAS        | Windows | Linux | macOS |
| :-----------| :------: | :----: | :----: |
| `intel-mkl` |   ✅    |  ✅   |       |
| `openblas`  |   ✅    |  ✅   |  ✅   |
| `flexiblas` |         |  ✅   |       |
| `netlib`    |         |       |       |
| `accelerate`|         |       |  ✅   |

## Usage

```bash
# Default: FlexiBLAS dynamic LP64
cargo build

# macOS with OpenBLAS (requires pkg-config)
cargo build --no-default-features -F openblas-dynamic-lp64

# Linux with explicit FlexiBLAS (requires pkg-config)
cargo build --no-default-features -F flexiblas-dynamic-ilp64

# Linux with Intel MKL
cargo build --no-default-features -F mkl-dynamic-ilp64-seq

# Linux OpenBLAS static-link smoke test (requires an ABI-matching .pc and .a)
cargo run --release --no-default-features \
  --features openblas-static-ilp64 --example static_openblas

# Linux Intel MKL static-link smoke test (requires the MKL pkg-config profile)
cargo run --release --no-default-features \
  --features mkl-static-ilp64-iomp --example static_mkl

# Windows with Intel MKL (requires the matching pkg-config profile)
cargo build --no-default-features -F mkl-dynamic-lp64-iomp

# Windows with OpenBLAS (requires vcpkg)
cargo build --no-default-features -F openblas-dynamic-lp64

# Run example
cargo run --release --example mat_blas

# Run common Level 1/2/3 examples
cargo run --release --example blas_levels

# Run OpenBLAS/FlexiBLAS extensions
cargo run --release --example openblas_extensions

# Run Intel MKL explicit 64-bit APIs
cargo run --release --no-default-features -F mkl-dynamic-ilp64-seq --example mkl_64

# Run Intel MKL ordinary and explicit 64-bit Sparse BLAS
cargo run --release --no-default-features -F mkl-dynamic-lp64-seq --example mkl_sparse

# Run Criterion benchmarks
cargo bench --bench cblas

# The static smoke tests above are also available through just.
just static-openblas
just static-mkl

# Locally render all backend API pages like docs.rs
DOCS_RS=1 RUSTDOCFLAGS="--cfg docsrs" cargo doc --no-deps --all-features
```

docs.rs uses a documentation-only all-feature configuration so feature-gated
MKL, MKL `_64`, and OpenBLAS/FlexiBLAS modules are visible together. This mode
does not represent a valid link configuration; normal builds still require
exactly one ABI and one backend.

## Requirements

| Platform | BLAS | Requirements |
| :--------| :----| :------------ |
| macOS | Accelerate | None (built-in) |
| macOS | OpenBLAS | Install via Homebrew: `brew install openblas` |
| Linux | Intel MKL | Expose the selected `mkl-*.pc` profile through `pkg-config` |
| Linux | OpenBLAS | Install an ABI-matching OpenBLAS static archive and `.pc` file |
| Linux | FlexiBLAS | Install ABI-matching FlexiBLAS shared libraries and a `.pc` file |
| Windows | Intel MKL | Expose the selected `mkl-*.pc` profile through `pkg-config` |
| Windows | OpenBLAS | Install via vcpkg: `vcpkg install openblas:x64-windows` |



## Fedora OpenBLAS pkg-config profiles

Fedora packages OpenBLAS libraries separately from FlexiBLAS and omits the
OpenBLAS `.pc` files. Generate user-local profiles for the installed serial
LP64 and ILP64 libraries (with ordinary, unsuffixed CBLAS symbols):

```bash
sudo dnf install openblas-devel openblas-serial openblas-serial64
# Also install this package for static linking:
sudo dnf install openblas-static gcc-gfortran
python3 scripts/fedora-openblas-pc.py
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
pkg-config --libs --static openblas64
cargo run --release --no-default-features -F openblas-static-ilp64 --example static_openblas
```

The script checks the runtime ABI and selects the single-threaded libraries
`libopenblas` and `libopenblas64`. It adds `OPENBLAS_USE64BITINT` to the ILP64
C compiler flags and records the Fortran, math, and pthread dependencies for
static linking. It does not make the entire executable static. Use
`--output-dir PATH` for another destination, or `--force` to regenerate existing
profiles. It does not modify system files or install packages.

## License
This Rust bindings library is licensed under the MIT or Apache-2.0 license, at your option.

Note: This project only provides bindings to the following libraries:
- Intel MKL (Intel® oneAPI Math Kernel Library) — proprietary license: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
- OpenBLAS — BSD 3-Clause license: https://github.com/xianyi/OpenBLAS/blob/develop/LICENSE
- Netlib BLAS — public domain
- Apple Accelerate — proprietary license: https://developer.apple.com/documentation/accelerate

You must comply with the licenses of these libraries when using this crate.

[intel mkl]: https://software.intel.com/en-us/mkl
[openblas]: https://github.com/OpenMathLib/OpenBLAS
[netlib]: http://www.netlib.org/blas/
