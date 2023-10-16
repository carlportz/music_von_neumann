use crate::defaults::DEFAULT_BICGSTAB_TOL;
use crate::ndarray_complex::Conj;
use crate::types::{CVector, CVectorView, Float};
use ndarray_linalg::norm::Norm;
use num_complex::Complex;

pub fn bicgstab(
    linop: impl Fn(CVectorView) -> CVector,
    b: CVectorView,
    x0: Option<CVectorView>,
    tol: Option<Float>,
    maxiter: Option<usize>,
) -> Result<CVector, &'static str> {
    let mut x_i: CVector = if x0.is_some() {
        x0.unwrap().to_owned()
    } else {
        CVector::zeros(b.len())
    };
    let tol: Float = tol.unwrap_or(DEFAULT_BICGSTAB_TOL);
    let maxiter: usize = maxiter.unwrap_or(20 * b.len());

    let r0: CVector = b.to_owned() - linop(x_i.view());
    let r0_hat: CVector = r0.clone();
    let mut rho_i: Complex<Float> = r0.view().conj().dot(&r0_hat);
    let r0_hat_conj: CVector = r0_hat.conj();
    let mut r_i: CVector = r0.clone();
    let mut p_i: CVector = r0;

    for iter in 0..maxiter {
        let nu: CVector = linop(p_i.view());
        let alpha: Complex<Float> = rho_i / r0_hat_conj.dot(&nu);
        let h: CVector = &x_i + alpha * &p_i;
        let s: CVector = &r_i - alpha * &nu;
        println!("Iteration: {}, s_residual: {}", iter, s.norm());
        if s.norm() < tol {
            println!("Converged in {} iterations.", iter);
            return Ok(h);
        }
        let t: CVector = linop(s.view());
        let omega: Complex<Float> = t.view().conj().dot(&s) / t.view().conj().dot(&t);
        x_i = &h + omega * &s;
        r_i = &s - omega * &t;
        if r_i.norm() < tol {
            println!("Converged in {} iterations.", iter);
            return Ok(x_i);
        }
        let rho_ip: Complex<Float> = r0_hat_conj.dot(&r_i);
        let beta: Complex<Float> = (rho_ip / rho_i) * (alpha / omega);
        rho_i = rho_ip;
        p_i = &r_i + (&p_i - &nu * omega) * beta;
    }
    Err("Maximum number of iterations reached.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray_complex::carray_abs_diff_eq;
    use crate::types::{CMatrix, CMatrixView};
    use ndarray_linalg::Solve;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn random_complex<R: Rng + ?Sized>(rng: &mut R) -> Complex<Float> {
        Complex::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    }

    fn matvec(a: CMatrixView, x: CVectorView) -> CVector {
        let ndim = a.nrows();
        let mut out = CVector::zeros(ndim);
        for i in 0..ndim {
            out[i] = a.row(i).dot(&x);
        }
        out
    }

    fn test_bicgstab(ndim: usize, seed: Option<u64>) {
        let mut rng = if seed.is_some() {
            SmallRng::seed_from_u64(seed.unwrap())
        } else {
            SmallRng::from_entropy()
        };

        // Create a ndim x ndim random complex matrix
        let a: CMatrix = CMatrix::from_shape_simple_fn((ndim, ndim), || random_complex(&mut rng));
        println!("A:");
        println!("{:8.4}", a);

        // Create a ndim-dimensional random vector
        let b: CVector = CVector::from_shape_simple_fn(ndim, || random_complex(&mut rng));
        println!("b:");
        println!("{:8.4}", b);

        // Solve for ax = b using LU decomposition
        let x_ref = a.solve(&b).unwrap();
        println!("x_ref:");
        println!("{:12.8}", x_ref);

        // Solve for ax = b using BiCGSTAB
        let x_test: CVector =
            bicgstab(|x| matvec(a.view(), x), b.view(), None, None, Some(10000)).unwrap();
        println!("x_test:");
        println!("{:12.8}", x_test);

        // Compare the results
        assert!(carray_abs_diff_eq(&x_ref, &x_test, 1e-5));
    }

    #[test]
    fn test_bicgstab_10() {
        test_bicgstab(10, None);
    }

    #[test]
    fn test_bicgstab_10_seed() {
        test_bicgstab(10, Some(42));
    }

    /*
    #[test]
    fn test_bicgstab_100() {
        test_bicgstab(100, None);
    }

    #[test]
    fn test_bicgstab_100_seed() {
        test_bicgstab(100, Some(42));
    }
    */
}
