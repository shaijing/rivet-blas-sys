/// Integer type used by the selected backend's ordinary CBLAS entry points.
///
/// The selected complete configuration's `ilp64` ABI maps to `long long` and
/// its `lp64` ABI maps to `int`. For Intel MKL this corresponds to `MKL_INT`
/// with and without `MKL_ILP64`, respectively.
#[cfg(any(
    docsrs,
    all(
        not(docsrs),
        any(
            feature = "openblas-dynamic-ilp64",
            feature = "openblas-static-ilp64",
            feature = "flexiblas-dynamic-ilp64",
            feature = "flexiblas-static-ilp64",
            feature = "mkl-dynamic-ilp64-gomp",
            feature = "mkl-dynamic-ilp64-iomp",
            feature = "mkl-dynamic-ilp64-seq",
            feature = "mkl-dynamic-ilp64-tbb",
            feature = "mkl-static-ilp64-gomp",
            feature = "mkl-static-ilp64-iomp",
            feature = "mkl-static-ilp64-seq",
            feature = "mkl-static-ilp64-tbb",
            feature = "mkl-sdl",
        )
    )
))]
pub type CBlasInt = ::std::os::raw::c_longlong;
#[cfg(any(all(
    not(docsrs),
    any(
        feature = "openblas-dynamic-lp64",
        feature = "openblas-static-lp64",
        feature = "flexiblas-dynamic-lp64",
        feature = "flexiblas-static-lp64",
        feature = "mkl-dynamic-lp64-gomp",
        feature = "mkl-dynamic-lp64-iomp",
        feature = "mkl-dynamic-lp64-seq",
        feature = "mkl-dynamic-lp64-tbb",
        feature = "mkl-static-lp64-gomp",
        feature = "mkl-static-lp64-iomp",
        feature = "mkl-static-lp64-seq",
        feature = "mkl-static-lp64-tbb",
        feature = "netlib-dynamic-lp64",
        feature = "netlib-static-lp64",
        feature = "accelerate",
    )
)))]
pub type CBlasInt = ::std::os::raw::c_int;

/// Return/index type used by CBLAS (`size_t` for the ordinary MKL API).
pub type CBlasIndex = usize;

/// Integer types used by Intel MKL's explicit `*_64` CBLAS interface.
///
/// These aliases are intentionally independent of the selected configuration's
/// ABI: MKL's suffixed interface always uses `MKL_INT64` and `MKL_UINT64`.
pub type MklCBlasInt64 = ::std::os::raw::c_longlong;
pub type MklCBlasIndex64 = ::std::os::raw::c_ulonglong;

pub type CBlasFloat = ::std::os::raw::c_float;
pub type CBlasDouble = ::std::os::raw::c_double;
pub type CBlasVoid = ::std::os::raw::c_void;
pub type CBlasF16 = ::std::os::raw::c_ushort;

#[repr(C)]
#[derive(Copy, Clone)]
pub enum CBlasLayout {
    CBlasRowMajor = 101,
    CBlasColMajor = 102,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub enum CBlasTranspose {
    CBlasNoTrans = 111,
    CBlasTrans = 112,
    CBlasConjTrans = 113,
    /// FlexiBLAS/OpenBLAS extension for conjugating without transposing.
    CBlasConjNoTrans = 114,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum CBlasUplo {
    CblasUpper = 121,
    CblasLower = 122,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum CBlasDiag {
    CblasNonUnit = 131,
    CblasUnit = 132,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum CBlasSide {
    CblasLeft = 141,
    CblasRight = 142,
}
