#![allow(dead_code)]
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex;

pub type Float = f64;

pub type RVector = Array1<Float>;
pub type RMatrix = Array2<Float>;
pub type RVectorView<'a> = ArrayView1<'a, Float>;
pub type RMatrixView<'a> = ArrayView2<'a, Float>;

pub type CVector = Array1<Complex<Float>>;
pub type CMatrix = Array2<Complex<Float>>;
pub type CVectorView<'a> = ArrayView1<'a, Complex<Float>>;
pub type CMatrixView<'a> = ArrayView2<'a, Complex<Float>>;
