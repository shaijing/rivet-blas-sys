use std::path::Path;

use std::env;

// Keep native-link configuration centralized in this build script.

fn feature_enabled(feature: &str) -> bool {
    let env_feature = feature.to_uppercase().replace('-', "_");
    env::var(format!("CARGO_FEATURE_{env_feature}")).is_ok()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Openblas,
    Flexiblas,
    IntelMkl,
    Netlib,
    Accelerate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Abi {
    Ilp64,
    Lp64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinkMode {
    Dynamic,
    Static,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MklThreading {
    Gomp,
    Iomp,
    Seq,
    Tbb,
    Sdl,
    None,
}

#[derive(Clone, Copy, Debug)]
struct BuildConfig {
    backend: Backend,
    abi: Abi,
    link: LinkMode,
    mkl_threading: MklThreading,
}

const CONFIGS: &[(&str, BuildConfig)] = &[
    (
        "openblas-dynamic-ilp64",
        BuildConfig {
            backend: Backend::Openblas,
            abi: Abi::Ilp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::None,
        },
    ),
    (
        "openblas-dynamic-lp64",
        BuildConfig {
            backend: Backend::Openblas,
            abi: Abi::Lp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::None,
        },
    ),
    (
        "openblas-static-ilp64",
        BuildConfig {
            backend: Backend::Openblas,
            abi: Abi::Ilp64,
            link: LinkMode::Static,
            mkl_threading: MklThreading::None,
        },
    ),
    (
        "openblas-static-lp64",
        BuildConfig {
            backend: Backend::Openblas,
            abi: Abi::Lp64,
            link: LinkMode::Static,
            mkl_threading: MklThreading::None,
        },
    ),
    (
        "flexiblas-dynamic-ilp64",
        BuildConfig {
            backend: Backend::Flexiblas,
            abi: Abi::Ilp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::None,
        },
    ),
    (
        "flexiblas-dynamic-lp64",
        BuildConfig {
            backend: Backend::Flexiblas,
            abi: Abi::Lp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::None,
        },
    ),
    (
        "flexiblas-static-ilp64",
        BuildConfig {
            backend: Backend::Flexiblas,
            abi: Abi::Ilp64,
            link: LinkMode::Static,
            mkl_threading: MklThreading::None,
        },
    ),
    (
        "flexiblas-static-lp64",
        BuildConfig {
            backend: Backend::Flexiblas,
            abi: Abi::Lp64,
            link: LinkMode::Static,
            mkl_threading: MklThreading::None,
        },
    ),
    (
        "mkl-dynamic-ilp64-gomp",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Ilp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::Gomp,
        },
    ),
    (
        "mkl-dynamic-ilp64-iomp",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Ilp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::Iomp,
        },
    ),
    (
        "mkl-dynamic-ilp64-seq",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Ilp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::Seq,
        },
    ),
    (
        "mkl-dynamic-ilp64-tbb",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Ilp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::Tbb,
        },
    ),
    (
        "mkl-dynamic-lp64-gomp",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Lp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::Gomp,
        },
    ),
    (
        "mkl-dynamic-lp64-iomp",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Lp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::Iomp,
        },
    ),
    (
        "mkl-dynamic-lp64-seq",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Lp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::Seq,
        },
    ),
    (
        "mkl-dynamic-lp64-tbb",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Lp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::Tbb,
        },
    ),
    (
        "mkl-sdl",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Ilp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::Sdl,
        },
    ),
    (
        "mkl-static-ilp64-gomp",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Ilp64,
            link: LinkMode::Static,
            mkl_threading: MklThreading::Gomp,
        },
    ),
    (
        "mkl-static-ilp64-iomp",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Ilp64,
            link: LinkMode::Static,
            mkl_threading: MklThreading::Iomp,
        },
    ),
    (
        "mkl-static-ilp64-seq",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Ilp64,
            link: LinkMode::Static,
            mkl_threading: MklThreading::Seq,
        },
    ),
    (
        "mkl-static-ilp64-tbb",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Ilp64,
            link: LinkMode::Static,
            mkl_threading: MklThreading::Tbb,
        },
    ),
    (
        "mkl-static-lp64-gomp",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Lp64,
            link: LinkMode::Static,
            mkl_threading: MklThreading::Gomp,
        },
    ),
    (
        "mkl-static-lp64-iomp",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Lp64,
            link: LinkMode::Static,
            mkl_threading: MklThreading::Iomp,
        },
    ),
    (
        "mkl-static-lp64-seq",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Lp64,
            link: LinkMode::Static,
            mkl_threading: MklThreading::Seq,
        },
    ),
    (
        "mkl-static-lp64-tbb",
        BuildConfig {
            backend: Backend::IntelMkl,
            abi: Abi::Lp64,
            link: LinkMode::Static,
            mkl_threading: MklThreading::Tbb,
        },
    ),
    (
        "netlib-dynamic-lp64",
        BuildConfig {
            backend: Backend::Netlib,
            abi: Abi::Lp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::None,
        },
    ),
    (
        "netlib-static-lp64",
        BuildConfig {
            backend: Backend::Netlib,
            abi: Abi::Lp64,
            link: LinkMode::Static,
            mkl_threading: MklThreading::None,
        },
    ),
    (
        "accelerate",
        BuildConfig {
            backend: Backend::Accelerate,
            abi: Abi::Lp64,
            link: LinkMode::Dynamic,
            mkl_threading: MklThreading::None,
        },
    ),
];

fn selected_config() -> BuildConfig {
    let mut selected = CONFIGS
        .iter()
        .filter(|(feature, _)| feature_enabled(feature))
        .map(|(feature, config)| (*feature, *config));

    let Some((feature, config)) = selected.next() else {
        panic!("enable exactly one complete BLAS configuration feature");
    };
    if let Some((other, _)) = selected.next() {
        panic!("BLAS configuration features `{feature}` and `{other}` are mutually exclusive");
    }
    config
}

fn emit_cfg_checks() {
    for cfg in [
        "rivet_blas_openblas",
        "rivet_blas_flexiblas",
        "rivet_blas_mkl",
        "rivet_blas_netlib",
        "rivet_blas_accelerate",
        "rivet_blas_static",
        "rivet_blas_mkl_sdl",
    ] {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
    }
}

fn emit_internal_cfg(config: BuildConfig) {
    emit_cfg_checks();

    let backend_cfg = match config.backend {
        Backend::Openblas => "rivet_blas_openblas",
        Backend::Flexiblas => "rivet_blas_flexiblas",
        Backend::IntelMkl => "rivet_blas_mkl",
        Backend::Netlib => "rivet_blas_netlib",
        Backend::Accelerate => "rivet_blas_accelerate",
    };
    println!("cargo:rustc-cfg={backend_cfg}");
    if config.link == LinkMode::Static {
        println!("cargo:rustc-cfg=rivet_blas_static");
    }
    if config.mkl_threading == MklThreading::Sdl {
        println!("cargo:rustc-cfg=rivet_blas_mkl_sdl");
    }
}

fn mkl_pkg_config_name(config: BuildConfig) -> &'static str {
    if config.mkl_threading == MklThreading::Sdl {
        return "mkl-sdl";
    }

    match (config.link, config.abi, config.mkl_threading) {
        (LinkMode::Dynamic, Abi::Ilp64, MklThreading::Gomp) => "mkl-dynamic-ilp64-gomp",
        (LinkMode::Dynamic, Abi::Ilp64, MklThreading::Iomp) => "mkl-dynamic-ilp64-iomp",
        (LinkMode::Dynamic, Abi::Ilp64, MklThreading::Seq) => "mkl-dynamic-ilp64-seq",
        (LinkMode::Dynamic, Abi::Ilp64, MklThreading::Tbb) => "mkl-dynamic-ilp64-tbb",
        (LinkMode::Dynamic, Abi::Lp64, MklThreading::Gomp) => "mkl-dynamic-lp64-gomp",
        (LinkMode::Dynamic, Abi::Lp64, MklThreading::Iomp) => "mkl-dynamic-lp64-iomp",
        (LinkMode::Dynamic, Abi::Lp64, MklThreading::Seq) => "mkl-dynamic-lp64-seq",
        (LinkMode::Dynamic, Abi::Lp64, MklThreading::Tbb) => "mkl-dynamic-lp64-tbb",
        (LinkMode::Static, Abi::Ilp64, MklThreading::Gomp) => "mkl-static-ilp64-gomp",
        (LinkMode::Static, Abi::Ilp64, MklThreading::Iomp) => "mkl-static-ilp64-iomp",
        (LinkMode::Static, Abi::Ilp64, MklThreading::Seq) => "mkl-static-ilp64-seq",
        (LinkMode::Static, Abi::Ilp64, MklThreading::Tbb) => "mkl-static-ilp64-tbb",
        (LinkMode::Static, Abi::Lp64, MklThreading::Gomp) => "mkl-static-lp64-gomp",
        (LinkMode::Static, Abi::Lp64, MklThreading::Iomp) => "mkl-static-lp64-iomp",
        (LinkMode::Static, Abi::Lp64, MklThreading::Seq) => "mkl-static-lp64-seq",
        (LinkMode::Static, Abi::Lp64, MklThreading::Tbb) => "mkl-static-lp64-tbb",
        _ => panic!("invalid Intel MKL configuration"),
    }
}

fn probe_mkl_pkg_config(config: BuildConfig) -> bool {
    let package = mkl_pkg_config_name(config);
    let mut pkg_config = pkg_config::Config::new();
    pkg_config.statik(config.link == LinkMode::Static);
    let result = if config.link == LinkMode::Static {
        // Intel's static .pc files contain absolute .a paths. The pkg-config
        // crate otherwise turns those paths back into unqualified `-l` flags,
        // allowing the linker to choose a shared object. Emit the metadata
        // ourselves so MKL archives remain explicitly static.
        pkg_config
            .cargo_metadata(false)
            .probe(package)
            .map(|library| emit_static_pkg_config_metadata(&library, mkl_static_runtime(config)))
    } else {
        pkg_config
            .cargo_metadata(false)
            .probe(package)
            .map(|library| emit_dynamic_mkl_pkg_config_metadata(&library))
    };

    if result.is_ok() {
        println!("cargo::warning=pkg_config {} used for Intel MKL", package);
        true
    } else {
        false
    }
}

fn mkl_static_runtime(config: BuildConfig) -> &'static [&'static str] {
    match config.mkl_threading {
        MklThreading::Gomp => &["gomp"],
        MklThreading::Iomp => &["iomp5"],
        MklThreading::Seq | MklThreading::Tbb | MklThreading::Sdl | MklThreading::None => &[],
    }
}

fn static_archive_exists(library: &pkg_config::Library, name: &str) -> bool {
    let archive_names = [format!("lib{name}.a"), format!("{name}.lib")];

    if library.link_files.iter().any(|path| {
        path.file_name()
            .and_then(|file| file.to_str())
            .is_some_and(|file| archive_names.iter().any(|name| name == file))
    }) {
        return true;
    }

    if library
        .link_paths
        .iter()
        .any(|path| archive_names.iter().any(|name| path.join(name).is_file()))
    {
        return true;
    }

    [
        Path::new("/usr/lib64"),
        Path::new("/lib64"),
        Path::new("/usr/lib"),
        Path::new("/lib"),
    ]
    .iter()
    .any(|path| archive_names.iter().any(|name| path.join(name).is_file()))
}

fn emit_static_pkg_config_metadata(library: &pkg_config::Library, static_libs: &[&str]) {
    for path in &library.link_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    for path in &library.link_files {
        if let Some(parent) = path.parent() {
            println!("cargo:rustc-link-search=native={}", parent.display());
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Some(name) = file_name
            .strip_prefix("lib")
            .and_then(|name| name.strip_suffix(".a"))
        {
            println!("cargo:rustc-link-lib=static={name}");
        }
    }

    // System libraries remain dynamic; iomp5 is static when the compiler
    // runtime archive is available in oneAPI's pkg-config search path.
    for name in &library.libs {
        let link_kind =
            if static_libs.contains(&name.as_str()) && static_archive_exists(library, name) {
                "static"
            } else {
                "dylib"
            };
        println!("cargo:rustc-link-lib={link_kind}={name}");
    }

    for args in &library.ld_args {
        println!("cargo:rustc-link-arg=-Wl,{}", args.join(","));
    }
}

fn emit_dynamic_mkl_pkg_config_metadata(library: &pkg_config::Library) {
    for path in &library.link_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    // MKL's interface, threading, and core shared objects reference one
    // another. Keep all layers in the final DT_NEEDED list when the consumer
    // toolchain enables --as-needed by default.
    let mut link_arg = String::from("-Wl,--no-as-needed");
    for name in &library.libs {
        link_arg.push_str(", -l");
        link_arg.push_str(name);
    }
    link_arg.push_str(",--as-needed");
    println!("cargo:rustc-link-arg={}", link_arg.replace(", ", ","));

    for args in &library.ld_args {
        println!("cargo:rustc-link-arg=-Wl,{}", args.join(","));
    }
}

fn probe_static_pkg_config(package: &str) {
    let library = pkg_config::Config::new()
        .statik(true)
        .cargo_metadata(false)
        .probe(package)
        .unwrap_or_else(|error| panic!("{package} static pkg-config probe failed: {error}"));

    if !static_archive_exists(&library, package) {
        panic!("{package} pkg-config profile does not provide a static archive");
    }

    emit_static_pkg_config_metadata(&library, &[package]);
}

// --- Windows Platform Logic ---
#[cfg(target_os = "windows")]
fn build_system(config: BuildConfig) {
    #[cfg(target_env = "msvc")]
    {
        if config.backend == Backend::IntelMkl {
            if !probe_mkl_pkg_config(config) {
                panic!(
                    "Intel MKL pkg-config profile `{}` not found; configure PKG_CONFIG_PATH",
                    mkl_pkg_config_name(config)
                );
            }
            println!("cargo::warning=intel-mkl used (msvc via pkg-config)");
        } else if config.backend == Backend::Openblas {
            if config.link == LinkMode::Static {
                vcpkg::Config::new()
                    .find_package("openblas")
                    .expect("openblas not found via vcpkg for static linking");
            } else {
                unsafe { env::set_var("VCPKGRS_DYNAMIC", "1") };
                if vcpkg::find_package("openblas").is_ok() {
                    println!("cargo::warning=vcpkg openblas used");
                } else if pkg_config::Config::new()
                    .statik(false)
                    .probe("openblas")
                    .is_ok()
                {
                    println!("cargo::warning=pkg_config openblas used");
                } else {
                    panic!("OpenBLAS not found via vcpkg or pkg-config");
                }
            }
        } else if config.backend == Backend::Netlib {
            if pkg_config::Config::new()
                .statik(config.link == LinkMode::Static)
                .probe("blas")
                .is_ok()
            {
                println!("cargo::warning=pkg_config netlib blas used");
            } else {
                panic!("Netlib BLAS not found via pkg-config");
            }
        }
    }

    #[cfg(target_env = "gnu")]
    {
        if config.backend == Backend::IntelMkl {
            if !probe_mkl_pkg_config(config) {
                panic!(
                    "Intel MKL pkg-config profile `{}` not found; configure PKG_CONFIG_PATH",
                    mkl_pkg_config_name(config)
                );
            }
            println!("cargo::warning=intel-mkl used (windows gnu via pkg-config)");
        } else if config.backend == Backend::Openblas {
            if pkg_config::Config::new()
                .statik(config.link == LinkMode::Static)
                .probe("openblas")
                .is_ok()
            {
                println!("cargo::warning=pkg_config openblas used (windows gnu)");
            } else {
                panic!("OpenBLAS not found via pkg-config (windows gnu)");
            }
        } else if config.backend == Backend::Netlib {
            if pkg_config::Config::new()
                .statik(config.link == LinkMode::Static)
                .probe("blas")
                .is_ok()
            {
                println!("cargo::warning=pkg_config netlib blas used (windows gnu)");
            } else {
                panic!("Netlib BLAS not found via pkg-config (windows gnu)");
            }
        }
    }
}

// --- Linux Platform Logic ---
#[cfg(target_os = "linux")]
fn build_system(config: BuildConfig) {
    if config.backend == Backend::IntelMkl {
        if !probe_mkl_pkg_config(config) {
            panic!(
                "Intel MKL pkg-config profile `{}` not found; configure PKG_CONFIG_PATH",
                mkl_pkg_config_name(config)
            );
        }
        println!("cargo::warning=intel-mkl used (linux)");
    } else if config.backend == Backend::Flexiblas {
        let flexiblas_pkg = if config.abi == Abi::Ilp64 {
            "flexiblas64"
        } else {
            "flexiblas"
        };
        if config.link == LinkMode::Static {
            probe_static_pkg_config(flexiblas_pkg);
        } else {
            pkg_config::Config::new()
                .statik(false)
                .probe(flexiblas_pkg)
                .expect("FlexiBLAS not found via pkg-config");
        }
        println!("cargo::warning=pkg_config {} used", flexiblas_pkg);
    } else if config.backend == Backend::Openblas {
        let blas_pkg = if config.abi == Abi::Ilp64 {
            "openblas64"
        } else {
            "openblas"
        };
        let flexiblas_pkg = if config.abi == Abi::Ilp64 {
            "flexiblas64"
        } else {
            "flexiblas"
        };
        if config.link == LinkMode::Static {
            probe_static_pkg_config(blas_pkg);
            println!("cargo::warning=pkg_config openblas static used");
        } else {
            // Try native openblas first
            if pkg_config::Config::new()
                .statik(false)
                .probe(blas_pkg)
                .is_ok()
            {
                println!("cargo::warning=pkg_config {} used", blas_pkg);
            }
            // Fallback to FlexiBLAS (common on Fedora/RHEL)
            else if pkg_config::Config::new()
                .statik(false)
                .probe(flexiblas_pkg)
                .is_ok()
            {
                println!(
                    "cargo::warning=pkg_config {} used as openblas fallback",
                    flexiblas_pkg
                );
            } else {
                panic!(
                    "Error: Could not find {} or {} via pkg-config.",
                    blas_pkg, flexiblas_pkg
                );
            }
        }
    } else if config.backend == Backend::Netlib {
        if pkg_config::Config::new()
            .statik(config.link == LinkMode::Static)
            .probe("blas")
            .is_ok()
        {
            println!("cargo::warning=pkg_config netlib blas used");
        } else {
            panic!("Error: Could not find netlib BLAS via pkg-config.");
        }
    }
}

// --- macOS Platform Logic ---
#[cfg(target_os = "macos")]
fn build_system(config: BuildConfig) {
    if config.backend == Backend::Openblas {
        if config.link == LinkMode::Static {
            pkg_config::Config::new()
                .statik(true)
                .probe("openblas")
                .expect("openblas not found via pkg-config for static linking on macOS");
            println!("cargo::warning=pkg_config openblas static used (macos)");
        } else if pkg_config::Config::new()
            .statik(false)
            .probe("openblas")
            .is_ok()
        {
            println!("cargo::warning=pkg_config openblas used (macos)");
        } else {
            panic!("Error: Could not find OpenBLAS via pkg-config on macOS.");
        }
    } else if config.backend == Backend::Accelerate {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo::warning=accelerate framework used (macos)");
    } else if config.backend == Backend::Netlib {
        if pkg_config::Config::new()
            .statik(config.link == LinkMode::Static)
            .probe("blas")
            .is_ok()
        {
            println!("cargo::warning=pkg_config netlib blas used (macos)");
        } else {
            panic!("Error: Could not find netlib BLAS via pkg-config on macOS.");
        }
    } else {
        // Default: link against macOS Accelerate framework (includes BLAS)
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo::warning=macos Accelerate framework used (default)");
    }
}

// --- Unsupported Platforms ---
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn build_system(config: BuildConfig) {
    let _ = config;
    println!("cargo:warning=unsupported platform for BLAS linking");
}

fn main() {
    // docs.rs renders all feature-gated APIs without linking a native backend.
    if env::var("DOCS_RS").is_ok() {
        emit_cfg_checks();
        return;
    }

    println!("cargo:rustc-check-cfg=cfg(docsrs)");
    emit_cfg_checks();

    let config = selected_config();
    if config.backend == Backend::Accelerate && !cfg!(target_os = "macos") {
        panic!("the `accelerate` configuration is only supported on macOS");
    }
    if config.backend == Backend::Flexiblas && !cfg!(target_os = "linux") {
        panic!("FlexiBLAS configurations are only supported on Linux");
    }
    if config.backend == Backend::IntelMkl && !cfg!(any(target_os = "windows", target_os = "linux"))
    {
        panic!("Intel MKL configurations are only supported on Windows and Linux");
    }
    emit_internal_cfg(config);

    // The compiler picks the correct build_system() version based on the target OS.
    build_system(config);
}
