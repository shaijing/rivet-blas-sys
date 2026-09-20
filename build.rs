#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
use std::path::Path;

#[allow(unused_imports)]
use std::env;

fn feature_enabled(feature: &str) -> bool {
    env::var(format!("CARGO_FEATURE_{}", feature.to_uppercase())).is_ok()
}

fn mkl_interface_lib() -> &'static str {
    if feature_enabled("ilp64") {
        "mkl_intel_ilp64"
    } else {
        "mkl_intel_lp64"
    }
}

// --- Windows Platform Logic ---
#[cfg(target_os = "windows")]
fn build_system() {
    #[cfg(target_env = "msvc")]
    {
        if feature_enabled("intel_mkl") {
            let mkl_root = env::var("MKLROOT").expect("MKLROOT should be set");
            let mkl_dir = Path::new(&mkl_root).join("lib");
            println!("cargo:rustc-link-search=native={}", mkl_dir.display());
            if feature_enabled("static") {
                println!("cargo:rustc-link-lib=static={}", mkl_interface_lib());
                println!("cargo:rustc-link-lib=static=mkl_intel_thread");
                println!("cargo:rustc-link-lib=static=mkl_core");
                println!("cargo:rustc-link-lib=static=mkl_def");
                println!("cargo:rustc-link-lib=dylib=libiomp5md");
            } else {
                println!("cargo:rustc-link-lib=mkl_rt");
            }
            println!("cargo::warning=intel-mkl used (msvc)");
        } else if feature_enabled("openblas") {
            if feature_enabled("static") {
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
                }
            }
        } else if feature_enabled("netlib") {
            if pkg_config::Config::new()
                .statik(feature_enabled("static"))
                .probe("blas")
                .is_ok()
            {
                println!("cargo::warning=pkg_config netlib blas used");
            }
        }
    }

    #[cfg(target_env = "gnu")]
    {
        if feature_enabled("intel_mkl") {
            let mkl_root = env::var("MKLROOT").expect("MKLROOT should be set");
            let mkl_dir = Path::new(&mkl_root).join("lib");
            println!("cargo:rustc-link-search=native={}", mkl_dir.display());
            if feature_enabled("static") {
                println!("cargo:rustc-link-lib=static={}", mkl_interface_lib());
                println!("cargo:rustc-link-lib=static=mkl_intel_thread");
                println!("cargo:rustc-link-lib=static=mkl_core");
                println!("cargo:rustc-link-lib=dylib=iomp5");
                println!("cargo:rustc-link-lib=dylib=pthread");
                println!("cargo:rustc-link-lib=dylib=m");
                println!("cargo:rustc-link-lib=dylib=dl");
            } else {
                println!("cargo:rustc-link-lib=mkl_rt");
            }
            println!("cargo::warning=intel-mkl used (windows gnu)");
        } else if feature_enabled("openblas") {
            if pkg_config::Config::new()
                .statik(feature_enabled("static"))
                .probe("openblas")
                .is_ok()
            {
                println!("cargo::warning=pkg_config openblas used (windows gnu)");
            }
        } else if feature_enabled("netlib") {
            if pkg_config::Config::new()
                .statik(feature_enabled("static"))
                .probe("blas")
                .is_ok()
            {
                println!("cargo::warning=pkg_config netlib blas used (windows gnu)");
            }
        }
    }
}

// --- Linux Platform Logic ---
#[cfg(target_os = "linux")]
fn build_system() {
    if feature_enabled("intel_mkl") {
        let mkl_root = env::var("MKLROOT").expect("MKLROOT should be set");
        let mkl_dir = Path::new(&mkl_root).join("lib/intel64");
        println!("cargo:rustc-link-search=native={}", mkl_dir.display());
        if feature_enabled("static") {
            println!("cargo:rustc-link-lib=static={}", mkl_interface_lib());
            println!("cargo:rustc-link-lib=static=mkl_intel_thread");
            println!("cargo:rustc-link-lib=static=mkl_core");
            println!("cargo:rustc-link-lib=dylib=iomp5");
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=dl");
        } else {
            println!("cargo:rustc-link-lib=mkl_rt");
        }
        println!("cargo::warning=intel-mkl used (linux)");
    } else if feature_enabled("openblas") {
        let blas_pkg = if feature_enabled("ilp64") {
            "openblas64"
        } else {
            "openblas"
        };
        let flexiblas_pkg = if feature_enabled("ilp64") {
            "flexiblas64"
        } else {
            "flexiblas"
        };
        if feature_enabled("static") {
            pkg_config::Config::new()
                .statik(true)
                .probe(blas_pkg)
                .expect("openblas not found via pkg-config for static linking");
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
    } else if feature_enabled("netlib") {
        if pkg_config::Config::new()
            .statik(feature_enabled("static"))
            .probe("blas")
            .is_ok()
        {
            println!("cargo::warning=pkg_config netlib blas used");
        } else {
            panic!("Error: Could not find netlib BLAS via pkg-config.");
        }
    } else if feature_enabled("system") {
        // Try system-provided BLAS: openblas, then flexiblas, then netlib blas
        if pkg_config::Config::new()
            .statik(false)
            .probe("openblas")
            .is_ok()
        {
            println!("cargo::warning=system openblas used");
        } else if pkg_config::Config::new()
            .statik(false)
            .probe("flexiblas")
            .is_ok()
        {
            println!("cargo::warning=system flexiblas used");
        } else if pkg_config::Config::new()
            .statik(false)
            .probe("blas")
            .is_ok()
        {
            println!("cargo::warning=system netlib blas used");
        } else {
            panic!("Error: No system BLAS implementation found via pkg-config.");
        }
    }
}

// --- macOS Platform Logic ---
#[cfg(target_os = "macos")]
fn build_system() {
    if feature_enabled("intel_mkl") {
        let mkl_root = env::var("MKLROOT").expect("MKLROOT should be set");
        let mkl_dir = Path::new(&mkl_root).join("lib");
        println!("cargo:rustc-link-search=native={}", mkl_dir.display());
        if feature_enabled("static") {
            println!("cargo:rustc-link-lib=static={}", mkl_interface_lib());
            println!("cargo:rustc-link-lib=static=mkl_intel_thread");
            println!("cargo:rustc-link-lib=static=mkl_core");
            println!("cargo:rustc-link-lib=dylib=iomp5");
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=dl");
        } else {
            println!("cargo:rustc-link-lib=mkl_rt");
        }
        println!("cargo::warning=intel-mkl used (macos)");
    } else if feature_enabled("openblas") {
        if feature_enabled("static") {
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
            println!(
                "cargo::warning=openblas feature requested but not found; falling back to Accelerate (macos)"
            );
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
    } else if feature_enabled("accelerate") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo::warning=accelerate framework used (macos)");
    } else if feature_enabled("netlib") {
        if pkg_config::Config::new()
            .statik(feature_enabled("static"))
            .probe("blas")
            .is_ok()
        {
            println!("cargo::warning=pkg_config netlib blas used (macos)");
        } else {
            panic!("Error: Could not find netlib BLAS via pkg-config on macOS.");
        }
    } else if feature_enabled("system") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo::warning=system accelerate framework used (macos)");
    } else {
        // Default: link against macOS Accelerate framework (includes BLAS)
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo::warning=macos Accelerate framework used (default)");
    }
}

// --- Unsupported Platforms ---
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn build_system() {
    if feature_enabled("system") {
        // Try pkg-config as a last resort on unknown platforms
        if pkg_config::Config::new()
            .statik(false)
            .probe("blas")
            .is_ok()
        {
            println!("cargo::warning=system blas used (unknown platform)");
        } else {
            panic!("Error: No BLAS implementation found on this platform.");
        }
    } else {
        println!("cargo:warning=unsupported platform for BLAS linking");
    }
}

fn main() {
    // Skip build logic when generating documentation on docs.rs
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    // The compiler picks the correct build_system() version based on the target OS
    build_system();
}
