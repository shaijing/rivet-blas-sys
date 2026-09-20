/// Integer type used by the selected backend's ordinary CBLAS entry points.
///
/// `ilp64` maps to `long long` and `lp64` maps to `int`. For Intel MKL this
/// corresponds to `MKL_INT` with and without `MKL_ILP64`, respectively.
#[cfg(all(feature = "ilp64", not(feature = "lp64")))]
pub type CBlasInt = ::std::os::raw::c_longlong;
#[cfg(all(feature = "lp64", not(feature = "ilp64")))]
pub type CBlasInt = ::std::os::raw::c_int;

/// Return/index type used by CBLAS (`size_t` for the ordinary MKL API).
pub type CBlasIndex = usize;

/// Integer types used by Intel MKL's explicit `*_64` CBLAS interface.
///
/// These aliases are intentionally independent of the `lp64`/`ilp64` feature:
/// MKL's suffixed interface always uses `MKL_INT64` and `MKL_UINT64`.
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
