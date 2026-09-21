default: check

fmt:
    cargo fmt --all -- --check

check:
    cargo check

test:
    cargo test

examples:
    cargo check --examples

# Requires an OpenBLAS static archive and its ABI-matching pkg-config file.
static-openblas:
    cargo run --release --no-default-features --features openblas-static-ilp64 --example static_openblas

# Requires the Intel MKL static pkg-config profile.
static-mkl:
    cargo run --release --no-default-features --features mkl-static-ilp64-iomp --example static_mkl

# Requires a FlexiBLAS static archive and its ABI-matching pkg-config file.
static-flexiblas:
    cargo run --release --no-default-features --features flexiblas-static-ilp64 --example static_flexiblas

static: static-openblas static-flexiblas static-mkl
