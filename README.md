# rivet-blas-sys

This crate provides Rust FFI bindings to CBLAS (C interface to BLAS).

## Features

- ABI features: `ilp64` (default) or `lp64` — exactly one is required.
- Backend features: `openblas` (default), `flexiblas`, `intel-mkl`, `netlib`,
  `accelerate`, or `system-blas` — exactly one is required.
- `system` is a backwards-compatible alias for `system-blas`.
- `static` requests static linking where the platform and backend support it.

Enable exactly one of `ilp64` and `lp64`. Ordinary `cblas_*` functions use the
selected ABI. Intel MKL's explicit 64-bit symbols are exposed separately under
`cblas_level_one_mkl_64`, `cblas_level_two_mkl_64`, and
`cblas_level_three_mkl_64`, and always use `MKL_INT64`/`MKL_UINT64` regardless
of the selected ordinary ABI.

Backend-specific extensions are kept out of the common modules:

- `cblas_level_one_mkl` and `cblas_level_three_mkl` contain Intel MKL-only APIs.
- `cblas_level_one_openblas` and `cblas_level_three_openblas` contain
  OpenBLAS/FlexiBLAS extensions.

The default feature set is `openblas` + `ilp64`. On Linux, the `openblas`
backend may use an ABI-compatible FlexiBLAS package as a compatibility fallback
when the OpenBLAS pkg-config package is unavailable; use `flexiblas` when that
backend choice should be explicit. Do not use `--all-features`, because backend
and ABI features are intentionally mutually exclusive.

When selecting another backend, disable the defaults explicitly, for example:

```bash
cargo test --no-default-features -F intel-mkl -F ilp64
cargo test --no-default-features -F intel-mkl -F lp64
cargo test --no-default-features -F flexiblas -F ilp64
cargo test --no-default-features -F openblas -F lp64
```

## Supported Platforms

| BLAS        | Windows | Linux | macOS |
| :-----------| :------: | :----: | :----: |
| `intel-mkl` |   ✅    |  ✅   |       |
| `openblas`  |   ✅    |  ✅   |  ✅   |
| `flexiblas` |         |  ✅   |  ✅   |
| `netlib`    |   ✅    |  ✅   |  ✅   |
| `accelerate`|         |       |  ✅   |
| `system-blas` | ✅    |  ✅   |  ✅   |

## Usage

```bash
# Default: OpenBLAS with ILP64
cargo build

# macOS with OpenBLAS (requires pkg-config)
cargo build -F openblas

# Linux with explicit FlexiBLAS (requires pkg-config)
cargo build --no-default-features -F flexiblas -F ilp64

# Linux with Intel MKL
cargo build --no-default-features -F intel-mkl -F ilp64

# Linux with OpenBLAS
cargo build -F openblas

# Windows with Intel MKL (requires MKLROOT)
cargo build --no-default-features -F intel-mkl -F ilp64

# Windows with OpenBLAS (requires vcpkg)
cargo build -F openblas

# Run example
cargo run --release --example mat_blas

# Run common Level 1/2/3 examples
cargo run --release --example blas_levels

# Run OpenBLAS/FlexiBLAS extensions
cargo run --release --example openblas_extensions

# Run Intel MKL explicit 64-bit APIs
cargo run --release --no-default-features -F intel-mkl -F ilp64 --example mkl_64

# Run Criterion benchmarks
cargo bench --bench cblas

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
| Linux | Intel MKL | Set `MKLROOT` environment variable |
| Linux | OpenBLAS | Install via package manager |
| Windows | Intel MKL | Set `MKLROOT` environment variable |
| Windows | OpenBLAS | Install via vcpkg: `vcpkg install openblas:x64-windows` |



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
