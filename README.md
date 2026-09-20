# rivet-blas-sys

This crate provides Rust FFI bindings to CBLAS (C interface to BLAS).

## Features

- `intel-mkl` — Link against Intel MKL
- `openblas` — Link against OpenBLAS
- `ilp64` (default) — 64-bit integer API for large arrays
- `lp64` — 32-bit integer API

Enable exactly one of `ilp64` and `lp64`. Ordinary `cblas_*` functions use the
selected ABI. Intel MKL's explicit 64-bit symbols are exposed separately under
`cblas_level_one_mkl_64`, `cblas_level_two_mkl_64`, and
`cblas_level_three_mkl_64`, and always use `MKL_INT64`/`MKL_UINT64` regardless
of the selected ordinary ABI.

Backend-specific extensions are kept out of the common modules:

- `cblas_level_one_mkl` and `cblas_level_three_mkl` contain Intel MKL-only APIs.
- `cblas_level_one_openblas` and `cblas_level_three_openblas` contain
  OpenBLAS/FlexiBLAS extensions.

When using MKL, disable the default OpenBLAS feature explicitly, for example:

```bash
cargo test --no-default-features -F intel-mkl -F ilp64
cargo test --no-default-features -F intel-mkl -F lp64
```

## Supported Platforms

| BLAS        | Windows | Linux | macOS |
| :-----------| :------: | :----: | :----: |
| `intel-mkl` |   ✅    |  ✅   |       |
| `openblas`  |   ✅    |  ✅   |  ✅   |
| `accelerate`|         |       |  ✅ (default) |

## Usage

```bash
# macOS (default: Accelerate framework)
cargo build

# macOS with OpenBLAS (requires pkg-config)
cargo build -F openblas

# Linux with Intel MKL
cargo build -F intel-mkl

# Linux with OpenBLAS
cargo build -F openblas

# Windows with Intel MKL (requires MKLROOT)
cargo build -F intel-mkl

# Windows with OpenBLAS (requires vcpkg)
cargo build -F openblas

# Run example
cargo run --release --example mat_blas
```

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
