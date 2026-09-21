# rivet-blas-sys

This crate provides Rust FFI bindings to CBLAS (C interface to BLAS).

## Features

Features describe complete native link configurations. Enable exactly one
configuration feature; do not combine backend, ABI, static, or threading
features. The default is `openblas-static-lp64`.

- OpenBLAS: `openblas-{dynamic,static}-{ilp64,lp64}`.
- FlexiBLAS: `flexiblas-{dynamic,static}-{ilp64,lp64}`.
- Intel oneMKL: `mkl-{dynamic,static}-{ilp64,lp64}-{gomp,iomp,seq,tbb}`.
- Intel oneMKL SDL: `mkl-sdl` (the oneAPI SDL runtime configuration, using the
  ILP64 ordinary ABI in this crate).
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

On Linux, an OpenBLAS dynamic configuration may use an ABI-compatible
FlexiBLAS package as a compatibility fallback when the OpenBLAS pkg-config
package is unavailable; use a `flexiblas-*` feature when that backend choice
should be explicit. Do not use `--all-features` for a normal build, because
configuration features are intentionally mutually exclusive.

When selecting another backend, disable the defaults explicitly, for example:

```bash
cargo test --no-default-features -F mkl-dynamic-ilp64-seq
cargo test --no-default-features -F mkl-dynamic-lp64-iomp
cargo test --no-default-features -F flexiblas-dynamic-ilp64
cargo test --no-default-features -F openblas-dynamic-lp64
```

## Supported Platforms

| BLAS        | Windows | Linux | macOS |
| :-----------| :------: | :----: | :----: |
| `intel-mkl` |   ✅    |  ✅   |       |
| `openblas`  |   ✅    |  ✅   |  ✅   |
| `flexiblas` |         |  ✅   |       |
| `netlib`    |   ✅    |  ✅   |  ✅   |
| `accelerate`|         |       |  ✅   |

## Usage

```bash
# Default: OpenBLAS static LP64
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

# Linux FlexiBLAS static-link smoke test (requires an ABI-matching .pc and .a)
cargo run --release --no-default-features \
  --features flexiblas-static-ilp64 --example static_flexiblas

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

# Run Criterion benchmarks
cargo bench --bench cblas

# The static smoke tests above are also available through just.
just static-openblas
just static-flexiblas
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
| Linux | FlexiBLAS | Install an ABI-matching FlexiBLAS static archive and `.pc` file |
| Windows | Intel MKL | Expose the selected `mkl-*.pc` profile through `pkg-config` |
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
