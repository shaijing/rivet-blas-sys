use rivet_blas_sys::mkl::types::{MklComplex8, MklComplex16};

// Small independent dense reference calculations. No BLAS calls are used here.
pub trait Scalar: Copy + std::fmt::Debug {
    const COMPLEX: bool;
    fn new(real: f64, imag: f64) -> Self;
    fn parts(self) -> (f64, f64);
    fn tolerance() -> f64;
    fn add(self, rhs: Self) -> Self {
        let (a, b) = self.parts();
        let (c, d) = rhs.parts();
        Self::new(a + c, b + d)
    }
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = self.parts();
        let (c, d) = rhs.parts();
        Self::new(a * c - b * d, a * d + b * c)
    }
    fn conj(self) -> Self {
        let (a, b) = self.parts();
        Self::new(a, -b)
    }
}

macro_rules! real {
    ($ty:ty, $tol:expr) => {
        impl Scalar for $ty {
            const COMPLEX: bool = false;
            fn new(real: f64, _: f64) -> Self {
                real as Self
            }
            fn parts(self) -> (f64, f64) {
                (self as f64, 0.0)
            }
            fn tolerance() -> f64 {
                $tol
            }
        }
    };
}
real!(f32, 1e-5);
real!(f64, 1e-12);

macro_rules! complex {
    ($ty:ty, $component:ty, $tol:expr) => {
        impl Scalar for $ty {
            const COMPLEX: bool = true;
            fn new(real: f64, imag: f64) -> Self {
                Self {
                    real: real as $component,
                    imag: imag as $component,
                }
            }
            fn parts(self) -> (f64, f64) {
                (self.real as f64, self.imag as f64)
            }
            fn tolerance() -> f64 {
                $tol
            }
        }
    };
}
complex!(MklComplex8, f32, 1e-5);
complex!(MklComplex16, f64, 1e-12);

pub fn close<T: Scalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let (ar, ai) = actual.parts();
        let (er, ei) = expected.parts();
        let tol = T::tolerance() * (1.0 + er.abs() + ei.abs());
        assert!(
            (ar - er).abs() <= tol && (ai - ei).abs() <= tol,
            "element {i}: actual {actual:?}, expected {expected:?}"
        );
    }
}

pub fn transpose<T: Scalar>(a: &[T], rows: usize, cols: usize, conjugate: bool) -> Vec<T> {
    (0..cols)
        .flat_map(|col| {
            (0..rows).map(move |row| {
                let value = a[row * cols + col];
                if conjugate { value.conj() } else { value }
            })
        })
        .collect()
}

pub fn product<T: Scalar>(a: &[T], b: &[T], rows: usize, inner: usize, cols: usize) -> Vec<T> {
    (0..rows)
        .flat_map(|row| {
            (0..cols).map(move |col| {
                (0..inner).fold(T::new(0.0, 0.0), |sum, k| {
                    sum.add(a[row * inner + k].mul(b[k * cols + col]))
                })
            })
        })
        .collect()
}
