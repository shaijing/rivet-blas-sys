# AGENTS.md

Guide for coding agents working on `rivet-blas-sys`, a low-level CBLAS and
Intel oneMKL Inspector-Executor Sparse BLAS FFI crate. It provides unsafe
`extern "C"` declarations, not safe matrix abstractions.

## Repository map

- `Cargo.toml`, `build.rs`: features, configuration validation, native linking.
- `build_support/openblas.rs`: GEADD header/version ABI detection.
- `src/lib.rs`, `src/cblas/mod.rs`: public modules and prelude exports.
- `src/blas_types.rs`, `src/cblas/cblas_types.rs`: scalar aliases and CBLAS enums.
- `src/cblas/cblas_level_{one,two,three}.rs`: common BLAS declarations.
- `src/cblas/*_mkl.rs`, `*_mkl_64.rs`, `*_openblas.rs`: backend extensions.
- `src/mkl/{types,sparse,sparse_64}.rs`: MKL Sparse types and declarations.
- `tests/`: ABI and numerical tests; `tests/support/mkl_sparse_*` contains
  shared sparse cases and independent dense references for all precisions.
- `tests/test_mkl_sparse_header.rs`: opt-in C header/layout audit.
- `examples/`, `benches/cblas.rs`: runnable examples and Criterion benchmarks.
- `scripts/fedora-openblas-pc.py`: user-local Fedora OpenBLAS pkg-config files.

## Configuration and linking

The default is `openblas-dynamic-lp64`. Select exactly one complete native
configuration; use `--no-default-features -F <configuration>` for alternatives.
Do not use `--all-features` except for the documentation command below.
See `Cargo.toml` and README for the full feature matrix and setup commands.

- FlexiBLAS: Linux, dynamic linking only. MKL: Linux and Windows.
  Accelerate: macOS LP64 only. Netlib: LP64 only.
- OpenBLAS must never fall back to FlexiBLAS.
- Linux uses pkg-config; MKL uses the exact selected profile, without requiring
  `MKLROOT`. Dynamic MKL also needs its runtime libraries on the loader path.
- Windows MSVC uses vcpkg or pkg-config for OpenBLAS; Windows GNU uses
  pkg-config. macOS supports pkg-config OpenBLAS/Netlib and Accelerate.
- `mkl-sdl` ordinary symbols are LP64: keep `MKL_INTERFACE_LAYER` unset or
  `LP64`. Use an ILP64 configuration or explicit `*_64` symbols for 64-bit indices.
- Sparse `*_64` symbols were validated on oneMKL 2026.1; require a library
  version exporting them.
- OpenBLAS GEADD gained two transpose arguments in 0.3.34. Detect the ABI from
  all four header declarations, falling back to the pkg-config version only
  when headers are unavailable. Reject unknown/inconsistent signatures.
  Headers and libraries must match. FlexiBLAS retains the old ABI; docs.rs
  shows the new one. `rivet_blas_geadd_transpose` is not propagated to dependents.

When changing features or backends, update Cargo features, build validation and
linking, module exports, tests/examples/benchmarks, README, and this guide together.

## FFI conventions

- Keep common operations in common modules, MKL extensions in MKL modules,
  and OpenBLAS/FlexiBLAS extensions in OpenBLAS-named modules. A shared symbol
  name alone does not make an operation backend-independent.
- Ordinary declarations use `CBlasInt`. Actual MKL suffixed `*_64` declarations
  use `MklCBlasInt64`/`MklCBlasIndex64`, independent of the ordinary ABI.
- CBLAS complex arguments use `CBlasVoid` pointers. Sparse uses typed complex
  pointers and by-value `MklComplex8`/`MklComplex16` scalars; `MklInt`/`MklInt64`
  select ordinary/explicit-64 integers. Do not substitute CBLAS enums for sparse
  enums with different values.
- Keep `mkl_sparse_*` in `src/mkl/`. Creation borrows user arrays; export borrows
  storage without transferring ownership. Document lifetimes and mutations,
  and keep each handle within one ABI, including destruction.
- Sparse `update_values` currently supports full BSR updates only and may retain
  the replacement pointer. Keep its array alive. `copy` ignores its descriptor
  and makes an independent deep copy. Do not add deprecated legacy Sparse
  Level 2/3 routines or unsupported header-only APIs.
- Every new public FFI function needs Rustdoc covering the operation/output
  equation, dimensions/layout/strides/leading dimensions, backend/ABI limits,
  and a `# Safety` section with pointer and storage preconditions.
- Prefer small module-level `no_run` examples with asserted results.

## Validation

Before handing off code changes, run:

```bash
cargo fmt --all
git diff --check
cargo check
cargo test
cargo check --examples
```

For backend-specific changes, also run the relevant backend tests, examples,
and `cargo doc --no-deps` with that configuration. README documents setup,
static smoke tests, benchmarks, and platform commands. Test platforms are
Fedora 44, macOS 26 Tahoe, and Windows 11.

For module/cfg changes, verify all backend modules appear in:

```bash
DOCS_RS=1 RUSTDOCFLAGS="--cfg docsrs" cargo doc --no-deps --all-features
```

For sparse changes, run `test_mkl_sparse` under both LP64 and ILP64, plus SDL
and static configurations when installed. On Linux with GCC/Clang, also run
`test_mkl_sparse_header` under both ABIs with `MKL_INCLUDE_DIR` set and
`-- --ignored`. The audit is ignored by default so consumers need no C compiler
or headers for ordinary numerical tests.

Check `git status` for unintended changes; never add generated `target/` files.
