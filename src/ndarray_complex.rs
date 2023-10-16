use crate::types::Float;
use ndarray::{Array, ArrayBase, Data, DataMut, DataOwned, Dimension, Zip};
use num_complex::{Complex, ComplexFloat};

pub trait Conj {
    type Output;
    fn conj(self) -> Self::Output;
}

impl<S, D> Conj for ArrayBase<S, D>
where
    S: DataOwned<Elem = Complex<Float>> + DataMut,
    D: Dimension,
{
    type Output = Self;

    fn conj(mut self) -> Self {
        self.map_inplace(|x| {
            *x = x.clone().conj();
        });
        self
    }
}

impl<'a, S, D> Conj for &'a ArrayBase<S, D>
where
    S: Data<Elem = Complex<Float>>,
    D: Dimension,
{
    type Output = Array<Complex<Float>, D>;

    fn conj(self) -> Array<Complex<Float>, D> {
        self.map(|&x| x.conj())
    }
}

#[allow(dead_code)]
pub fn carray_abs_diff_eq<'a, S, D>(
    a: &'a ArrayBase<S, D>,
    b: &'a ArrayBase<S, D>,
    epsilon: Float,
) -> bool
where
    S: Data<Elem = Complex<Float>>,
    D: Dimension,
{
    Zip::from(a).and(b).all(|&x, &y| (x - y).abs() < epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CMatrix, CVector};
    use ndarray::array;

    #[test]
    fn test_array_conj() {
        let a: CVector = array![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b: CVector = array![Complex::new(1.0, -2.0), Complex::new(3.0, -4.0)];
        assert_eq!(a.conj(), b);
    }

    #[test]
    fn test_arrayview_conj() {
        let a: CVector = array![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b: CVector = array![Complex::new(1.0, -2.0), Complex::new(3.0, -4.0)];
        assert_eq!(a.view().conj(), b);
    }

    #[test]
    fn test_abs_diff_eq_cvector() {
        let a: CVector = array![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b: CVector = array![Complex::new(1.0, 2.0005), Complex::new(3.0, 4.0)];
        assert!(carray_abs_diff_eq(&a, &b, 1e-3));
    }

    #[test]
    #[should_panic]
    fn test_abs_diff_eq_cvector_xfail() {
        let a: CVector = array![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b: CVector = array![Complex::new(1.0, 2.0005), Complex::new(3.0, 4.0)];
        assert!(carray_abs_diff_eq(&a, &b, 1e-4));
    }

    #[test]
    fn test_abs_diff_eq_cmatrix() {
        let a: CMatrix = array![
            [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
            [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)],
        ];
        let b: CMatrix = array![
            [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
            [Complex::new(5.0, 6.0005), Complex::new(7.0, 8.0)],
        ];
        assert!(carray_abs_diff_eq(&a, &b, 1e-3));
    }

    #[test]
    #[should_panic]
    fn test_abs_diff_eq_cmatrix_xfail() {
        let a: CMatrix = array![
            [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
            [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)],
        ];
        let b: CMatrix = array![
            [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
            [Complex::new(5.0, 6.0005), Complex::new(7.0, 8.0)],
        ];
        assert!(carray_abs_diff_eq(&a, &b, 1e-4));
    }
}
