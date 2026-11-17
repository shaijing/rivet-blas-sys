# AGENTS.md

This file is the repository guide for coding agents working on
`rivet-blas-sys`. Keep it synchronized with `Cargo.toml`, `build.rs`, and
`README.md` when changing features or supported backends.

## Project purpose

`rivet-blas-sys` provides low-level Rust FFI bindings to CBLAS and Intel oneMKL
Inspector-Executor Sparse BLAS. It does not
provide safe matrix abstractions. Every exported BLAS function is an
`unsafe extern "C"` declaration, so callers must uphold the backend's pointer,
dimension, layout, and stride requirements.

## Repository layout

- `src/lib.rs` — crate root, feature validation, and public module exports.
- `src/blas_types.rs` — Rust-native aliases such as `BlasInt` and `BlasFloat`.
- `src/cblas/mod.rs` — common/backend-specific module declarations and
  `prelude` re-exports.
- `src/cblas/cblas_types.rs` — C-compatible scalar aliases and BLAS enums.
- `src/cblas/cblas_level_one.rs` — common Level 1 vector operations.
- `src/cblas/cblas_level_two.rs` — common Level 2 matrix-vector operations.
- `src/cblas/cblas_level_three.rs` — common Level 3 matrix-matrix operations.
- `src/cblas/cblas_level_*_mkl.rs` — Intel MKL-only ordinary-ABI extensions.
- `src/cblas/cblas_level_*_mkl_64.rs` — Intel MKL suffixed `*_64` symbols.
- `src/cblas/cblas_level_*_openblas.rs` — OpenBLAS/FlexiBLAS extensions.
- `src/mkl/types.rs` — MKL complex scalars, sparse handles, enums, and descriptors.
- `src/mkl/sparse.rs` — ordinary Inspector-Executor Sparse BLAS declarations.
- `src/mkl/sparse_64.rs` — actual suffixed `mkl_sparse_*_64` declarations.
- `build.rs` — platform-specific library discovery and linker configuration.
- `build_support/openblas.rs` — GEADD header/version ABI detection.
- `tests/` — ABI, common Level 1/2/3, and backend-specific integration tests.
- `tests/support/mkl_sparse_*` — shared ordinary/explicit-64 behavioral cases
  and independent dense reference calculations for every scalar precision.
- `tests/test_mkl_sparse_header.rs` — opt-in Linux audit against installed C
  headers for all sparse signatures, layouts, field offsets, and enum values.
- `examples/` — runnable common, OpenBLAS/FlexiBLAS, and MKL examples.
- `benches/cblas.rs` — Criterion benchmarks for common and backend-specific APIs.

## Feature model

The default configuration is `flexiblas-dynamic-lp64`.

Exactly one complete native configuration feature must be selected. The
feature name encodes backend, link mode, ABI, and (for MKL) threading:

- `openblas-{dynamic,static}-{ilp64,lp64}`
- `flexiblas-dynamic-{ilp64,lp64}`
- `mkl-{dynamic,static}-{ilp64,lp64}-{gomp,iomp,seq,tbb}`
- `mkl-sdl`
- `netlib-{dynamic,static}-lp64`
- `accelerate` (macOS LP64)

The build script rejects missing or multiple configuration features. Do not
use `--all-features` for a normal build; it intentionally enables invalid
combinations. When selecting a non-default configuration, add
`--no-default-features`.

Backend restrictions:

- `accelerate` is available only on macOS and supports LP64 only.
- FlexiBLAS configurations are available only on Linux and support dynamic linking only.
- Intel MKL configurations are available only on Windows and Linux.
- `netlib` supports LP64 only in this crate.
- Intel MKL ordinary symbols use `CBlasInt`, selected by the ABI in the
  complete MKL feature.
- MKL `*_64` symbols always use `MklCBlasInt64`/`MklCBlasIndex64`, independent
  of the ordinary ABI feature.
- `mkl-sdl` uses LP64 ordinary integers to match the runtime's default interface.
  Keep `MKL_INTERFACE_LAYER` unset or `LP64`; do not select ILP64 at runtime
  for ordinary SDL declarations. Use a complete ILP64 configuration or explicit
  `*_64` symbols when 64-bit indices are needed.
- MKL Sparse BLAS modules use the existing complete MKL configurations and
  `rivet_blas_mkl` cfg. Their explicit `*_64` symbols were validated on oneMKL
  2026.1 and require a library version exporting those symbols.
- OpenBLAS and FlexiBLAS configurations link their selected backend; an
  OpenBLAS configuration never falls back to FlexiBLAS.
- Linux MKL configurations use the exact selected pkg-config profile. No
  `MKLROOT` environment variable is required.

Test platforms:

- Linux: Fedora 44.
- macOS: macOS 26 Tahoe.
- Windows: Windows 11.

## Build, test, and documentation commands

Format and validate the default configuration:

```bash
cargo fmt --all
cargo check
cargo test
cargo doc --no-deps
# Render all feature-gated API modules locally, without linking a backend.
DOCS_RS=1 RUSTDOCFLAGS="--cfg docsrs" cargo doc --no-deps --all-features
```

Examples and benchmarks:

```bash
cargo check --examples
cargo run --release --example mat_blas
cargo run --release --example blas_levels
cargo run --release --example openblas_extensions
cargo bench --no-run
cargo bench --bench cblas

# Linux static-link smoke tests (requires matching static BLAS installations)
just static-openblas
just static-mkl
```

The common feature matrix can be checked with locally installed BLAS
implementations:

```bash
cargo test --no-default-features -F openblas-dynamic-lp64
cargo test --no-default-features -F flexiblas-dynamic-ilp64
cargo test --no-default-features -F flexiblas-dynamic-lp64
```

For Linux Intel MKL, make the selected MKL pkg-config profile visible to
`pkg-config` (for example by sourcing oneAPI's environment setup). For dynamic
MKL, make the MKL runtime libraries visible at runtime. The exact compiler
runtime path depends on the oneAPI installation:

```bash
export PKG_CONFIG_PATH=/path/to/intel/oneapi/mkl/latest/lib/pkgconfig
export LD_LIBRARY_PATH="/path/to/intel/oneapi/mkl/latest/lib:/path/to/oneapi/compiler/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cargo test --no-default-features -F mkl-dynamic-ilp64-seq
cargo test --no-default-features -F mkl-dynamic-lp64-seq
cargo run --release --no-default-features -F mkl-dynamic-ilp64-seq --example mkl_64
cargo test --no-default-features -F mkl-dynamic-lp64-seq --test test_mkl_sparse
cargo test --no-default-features -F mkl-dynamic-ilp64-seq --test test_mkl_sparse
cargo run --release --no-default-features -F mkl-dynamic-lp64-seq --example mkl_sparse
cargo bench --no-default-features -F mkl-dynamic-lp64-seq --bench cblas -- mkl/sparse
cargo doc --no-deps --no-default-features -F mkl-dynamic-ilp64-seq
```

Do not omit `--no-default-features` when selecting another configuration,
because the default FlexiBLAS configuration would remain enabled.


OpenBLAS GEADD changed its C ABI in 0.3.34: `cblas_{s,d,c,z}geadd` gained
`trans_a` and `trans_c` arguments after `layout`. The build script inspects all
four declarations in the discovered `cblas.h`; if no header is readable, it
uses the pkg-config version (0.3.34 is the boundary). Unknown or inconsistent
header signatures fail the build. FlexiBLAS keeps the non-transposing ABI.
Headers and libraries must come from the same installation. docs.rs shows the
new OpenBLAS signature; builds with older OpenBLAS expose the old signature.
The internal `rivet_blas_geadd_transpose` cfg is local to this crate and its
tests/examples, not automatically propagated to dependent crates. Applications
must use the signature matching their selected installation. Detection does
not execute target binaries and can be used when cross-compiling.

## Backend linking

`build.rs` selects libraries according to the target and enabled feature:

- Linux uses pkg-config for `openblas`/`openblas64`,
  `flexiblas`/`flexiblas64`, LP64 Netlib `blas`, and Intel MKL.
- Windows MSVC uses vcpkg or pkg-config for OpenBLAS and pkg-config for MKL.
- Windows GNU uses pkg-config for OpenBLAS, Netlib, and MKL.
- macOS supports pkg-config OpenBLAS, the Accelerate framework, and Netlib.
  `accelerate` is LP64-only.
When adding a backend, update all of the following together:

1. `Cargo.toml` feature definitions and `src/lib.rs` feature validation.
2. The relevant target branches in `build.rs`.
3. Conditional module exports in `src/cblas/mod.rs`.
4. Backend-specific tests, examples, and benchmark cfgs.
5. README and this file with working commands and requirements.

## API and FFI conventions

CBLAS names follow the standard pattern:

- `s` — `f32`, `d` — `f64`.
- `c` — complex single precision, `z` — complex double precision.
- The operation follows the precision prefix, for example `cblas_dgemm`.

Common functions belong in the common Level 1/2/3 modules. MKL-only symbols
must stay in MKL modules, and OpenBLAS/FlexiBLAS extensions must stay in the
OpenBLAS-named extension modules. Do not expose backend-specific functions
through common modules just because another backend happens to provide a
symbol with the same name.

Use `CBlasInt` for ordinary entry points. Use `MklCBlasInt64` only for MKL's
actual suffixed `*_64` declarations. Complex arguments currently use
`CBlasVoid` pointers in CBLAS to match its C ABI. Sparse BLAS instead passes
`MklComplex8`/`MklComplex16` scalars by value and uses typed complex pointers.
Its `MklInt` and `MklInt64` aliases map to the ordinary and explicit integer
types above. Sparse enums must not reuse CBLAS enums with different values.

Keep `mkl_sparse_*` functions in `src/mkl/`, separate from CBLAS modules.
Creation borrows user arrays; export borrows storage without transferring
ownership. Document pointer lifetimes and explicit mutations, and keep each
handle entirely within one ABI, including destruction. `update_values`
currently supports full BSR updates only and may retain the replacement pointer;
keep that array alive while referenced. `copy` currently ignores its descriptor
and makes an independent deep copy. Do not add deprecated legacy
Sparse BLAS Level 2/3 routines or unsupported header-only APIs.

For sparse changes, run `test_mkl_sparse` under both LP64 and ILP64; also test
SDL and static configurations when installed. On Linux with GCC/Clang, run the
header audit separately for LP64 and ILP64, for example:

```bash
MKL_INCLUDE_DIR=/path/to/oneapi/mkl/latest/include \
  cargo test --no-default-features -F mkl-dynamic-lp64-seq --test test_mkl_sparse_header -- --ignored
MKL_INCLUDE_DIR=/path/to/oneapi/mkl/latest/include \
  cargo test --no-default-features -F mkl-dynamic-ilp64-seq --test test_mkl_sparse_header -- --ignored
```

The header audit is intentionally ignored by default so ordinary consumers do
not need a C compiler or headers to run numerical tests.

Every newly added public FFI function should have Rustdoc that explains:

- the mathematical operation and output update, such as
  `C := alpha * op(A) * op(B) + beta * C`;
- the meaning of important dimensions, strides, layouts, and leading
  dimensions;
- backend/ABI requirements when applicable; and
- a `# Safety` section describing pointer and storage preconditions.

Prefer a module-level `no_run` example for each backend-specific API group.
Keep examples small, assert their expected result, and use
`--no-default-features` whenever they select a non-default backend.

## Change checklist

Before handing off a change:

```bash
cargo fmt --all
git diff --check
cargo check
cargo test
cargo check --examples
```

For feature-specific changes, also run the relevant backend test/example and
documentation command. For documentation-only cfg changes, verify that the
all-feature docs command above exposes every backend module. Verify that
`git status` contains only the intended changes and that no generated `target/`
artifacts are added.
