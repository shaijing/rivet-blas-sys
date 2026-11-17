#[path = "../build_support/openblas.rs"]
mod detection;

use detection::{GeaddAbi, header_abi, version_abi};

fn header(transpose: bool) -> String {
    ["s", "d", "c", "z"].map(|precision| {
        let trans = if transpose { "enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tc," } else { "" };
        format!("void cblas_{precision}geadd \n (enum CBLAS_ORDER order, {trans} blasint m, blasint n, float alpha, const float *a, blasint lda, float beta, float *c, blasint ldc);\n")
    }).join("")
}

#[test]
fn detects_both_header_abis_and_ignores_comments() {
    for (transpose, expected) in [(false, GeaddAbi::Legacy), (true, GeaddAbi::Transpose)] {
        let source = format!(
            "/* {} */\n// cblas_sgeadd(bogus)\n{}",
            header(!transpose),
            header(transpose)
        );
        assert_eq!(header_abi(&source), Ok(expected));
    }
}

#[test]
fn rejects_incomplete_unknown_and_mixed_headers() {
    assert!(header_abi("void cblas_sgeadd(void);").is_err());
    assert!(header_abi(&header(false).replace("blasint ldc", "blasint ldc, int extra")).is_err());
    let mixed = header(false).replace("cblas_sgeadd", "old_sgeadd") + &header(true);
    assert!(header_abi(&mixed).is_err());
}

#[test]
fn version_fallback_tracks_the_abi_boundary() {
    for version in ["0.3.29", "0.3.33", "0.3.33.dev"] {
        assert_eq!(version_abi(version), Ok(GeaddAbi::Legacy));
    }
    for version in ["0.3.34", "0.3.34.dev", "0.4.0", "1.0.0"] {
        assert_eq!(version_abi(version), Ok(GeaddAbi::Transpose));
    }
    for version in [
        "",
        "unknown",
        "0.3",
        "0junk.3.34",
        "0.3junk.34",
        "0.3.34junk",
    ] {
        assert!(version_abi(version).is_err());
    }
}
