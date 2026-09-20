# AGENTS.md

This file is the repository guide for coding agents working on
`rivet-blas-sys`. Keep it synchronized with `Cargo.toml`, `build.rs`, and
`README.md` when changing features or supported backends.

## Project purpose

`rivet-blas-sys` provides low-level Rust FFI bindings to CBLAS. It does not
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
- `build.rs` — platform-specific library discovery and linker configuration.
- `tests/` — ABI, common Level 1/2/3, and backend-specific integration tests.
- `examples/` — runnable common, OpenBLAS/FlexiBLAS, and MKL examples.
- `benches/cblas.rs` — Criterion benchmarks for common and backend-specific APIs.

## Feature model

The default configuration is `openblas` + `ilp64`.

Exactly one integer ABI and exactly one backend must be selected:

- ABI: `ilp64` or `lp64`.
- Backend: `openblas`, `flexiblas`, `intel-mkl`, `netlib`, `accelerate`, or
  `system-blas`.
- `system` is a compatibility alias for `system-blas`.
- `static` requests static linking where supported by the platform/backend.

The crate emits a compile error for conflicting or missing ABI/backend
features. Do not use `--all-features`; it intentionally enables invalid
combinations. When selecting a non-default backend, always add
`--no-default-features`.

Backend restrictions:

- `accelerate` is available only on macOS and supports LP64 only.
- `netlib` supports LP64 only in this crate.
- `intel-mkl` ordinary symbols use `CBlasInt`, selected by `lp64`/`ilp64`.
- MKL `*_64` symbols always use `MklCBlasInt64`/`MklCBlasIndex64`, independent
  of the ordinary ABI feature.
- On Linux, an `openblas` build may use an ABI-compatible FlexiBLAS pkg-config
  package as a compatibility fallback. Use `flexiblas` when that choice must
  be explicit.

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
```

The common feature matrix can be checked with locally installed BLAS
implementations:

```bash
cargo test --no-default-features -F openblas -F lp64
cargo test --no-default-features -F flexiblas -F ilp64
cargo test --no-default-features -F flexiblas -F lp64
cargo test --no-default-features -F system-blas -F lp64
```

For Intel MKL, set `MKLROOT` and make the MKL runtime libraries visible at
runtime. The exact compiler runtime path depends on the oneAPI installation:

```bash
export MKLROOT=/path/to/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH="$MKLROOT/lib:/path/to/oneapi/compiler/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cargo test --no-default-features -F intel-mkl -F ilp64
cargo test --no-default-features -F intel-mkl -F lp64
cargo run --release --no-default-features -F intel-mkl -F ilp64 --example mkl_64
cargo doc --no-deps --no-default-features -F intel-mkl -F ilp64
```

Do not use `cargo test -F intel-mkl` because the default `openblas` feature
would remain enabled and the crate rejects multiple backends.

## Backend linking

`build.rs` selects libraries according to the target and enabled feature:

- Linux uses pkg-config for `openblas`/`openblas64`,
  `flexiblas`/`flexiblas64`, and LP64 Netlib `blas`; MKL uses `MKLROOT`.
- Windows MSVC uses vcpkg or pkg-config for OpenBLAS and `MKLROOT` for MKL.
- Windows GNU uses pkg-config for OpenBLAS, FlexiBLAS, or Netlib and
  `MKLROOT` for MKL.
- macOS can use pkg-config OpenBLAS/FlexiBLAS, the Accelerate framework, or
  MKL. `accelerate` is LP64-only.
- `system-blas` probes only ABI-compatible system packages. It must not fall
  back from ILP64 to an LP64 `blas` package.

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
`CBlasVoid` pointers to match the C ABI.

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
