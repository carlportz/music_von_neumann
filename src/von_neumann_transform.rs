use crate::bicgstab::bicgstab;
use crate::ndarray_complex::Conj;
use crate::types::{CMatrix, CMatrixView, CVector, CVectorView, Float, RVector, RVectorView};
use ndarray::{s, Array, Array3, ArrayView3};
// use ndarray_linalg::{Solve, SolveH};
use ndarray_linalg::Solve;
use num_complex::{Complex, ComplexFloat};

const PI: Float = std::f64::consts::PI as Float;

pub struct VNTExtent {
    pub t_min: Float,
    pub t_max: Float,
    pub w_min: Float,
    pub w_max: Float,
}

fn get_alpha_nm_vec(w: RVectorView, alpha: Float, w_n: Float, t_m: Float) -> CVector {
    let diff: CVector = (w.to_owned() - w_n).mapv(|a| Complex::new(a, 0.0));
    Complex::new((2.0 * alpha / PI).powf(0.25), 0.0)
        * (Complex::new(-alpha, 0.0) * diff.mapv(|a| a.powi(2)) - Complex::i() * t_m * diff)
            .mapv(|a| a.exp())
}

fn get_grid(
    w_min: Float,
    w_max: Float,
    n_points: usize,
) -> (RVector, Float, RVector, RVector, Float) {
    // trimmed grid for signal in frequency domain
    let w_grid: RVector = RVector::linspace(w_min, w_max, n_points);
    let dw_grid = w_grid[1] - w_grid[0];

    // grid in von Neumann plane
    let w_span = w_grid[n_points - 1] - w_grid[0];
    let t_span = 2.0 * PI / dw_grid;
    let k = (n_points as Float).sqrt().round();
    assert_eq!((k as usize).pow(2), n_points);

    let dw = w_span / k;
    let dt = t_span / k;
    let w_n_arr: RVector = w_min + (RVector::range(0.0, k, 1.0) + 0.5) * dw;
    let t_n_arr: RVector = -t_span / 2.0 + (RVector::range(0.0, k, 1.0) + 0.5) * dt;

    // parameters for basis functions
    let alpha = t_span / (2.0 * w_span);

    (w_grid, t_span, w_n_arr, t_n_arr, alpha)
}

fn get_full_basis_functions(
    w_grid: RVectorView,
    w_n_arr: RVectorView,
    t_n_arr: RVectorView,
    alpha: Float,
) -> Array3<Complex<Float>> {
    let k = w_n_arr.shape()[0];
    let mut alpha_nmo: Array3<Complex<Float>> = Array::zeros((k, k, k * k));
    for (n, w_n) in w_n_arr.iter().enumerate() {
        for (m, t_m) in t_n_arr.iter().enumerate() {
            let alpha_nm_vec = get_alpha_nm_vec(w_grid.view(), alpha, *w_n, *t_m);
            alpha_nmo.slice_mut(s![n, m, ..]).assign(&alpha_nm_vec);
        }
    }
    alpha_nmo
}

fn get_contracted_basis_functions(
    signal: CVectorView,
    w_grid: RVectorView,
    w_n_arr: RVectorView,
    t_n_arr: RVectorView,
    alpha: Float,
) -> CMatrix {
    let k = w_n_arr.len();
    let mut alpha_nm: CMatrix = CMatrix::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            alpha_nm[[i, j]] = get_alpha_nm_vec(w_grid.view(), alpha, w_n_arr[i], t_n_arr[j])
                .conj()
                .dot(&signal);
        }
    }
    alpha_nm
}

fn get_ovlp_matrix(alpha: Float, w_n_arr: RVectorView, t_n_arr: RVectorView) -> CMatrix {
    let k = w_n_arr.len();
    let mut s: CMatrix = CMatrix::zeros((k * k, k * k));
    for (n, w_n) in w_n_arr.iter().enumerate() {
        for (i, w_i) in w_n_arr.iter().enumerate() {
            let w_diff = w_i - w_n;
            for (m, t_m) in t_n_arr.iter().enumerate() {
                for (j, t_j) in t_n_arr.iter().enumerate() {
                    let t_diff = t_j - t_m;
                    let t_sum = t_j + t_m;
                    let exponent = Complex::new(
                        -0.5 * alpha * w_diff.powi(2) - (1.0 / (8.0 * alpha)) * t_diff.powi(2),
                        0.5 * w_diff * t_sum,
                    );
                    s[[n * k + m, i * k + j]] = exponent.exp();
                }
            }
        }
    }
    // s *= Complex::new((2.0 * alpha / PI).sqrt(), 0.0);
    s
}

fn get_von_neumann_coefficients_direct(
    signal: CVectorView,
    alpha_nmo: ArrayView3<Complex<Float>>,
    alpha: Float,
    w_n_arr: RVectorView,
    t_n_arr: RVectorView,
) -> CMatrix {
    let k = alpha_nmo.shape()[0];
    let alpha_nm: CVector = alpha_nmo
        .into_shape((k * k, k * k))
        .unwrap()
        .conj()
        .dot(&signal);
    let s: CMatrix = get_ovlp_matrix(alpha, w_n_arr, t_n_arr);
    // let s: CMatrix = (s.view().t().conj() + &s).mapv(|a| a / 2.0);
    
    // // solveh appears to be buggy with accelerate AND intel-mkl
    // let q_nm: CMatrix = if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
    //     s.solve_into(alpha_nm).unwrap().into_shape((k, k)).unwrap()
    // } else {
    //     s.solveh_into(alpha_nm).unwrap().into_shape((k, k)).unwrap()
    // };
    let q_nm: CMatrix = s.solve_into(alpha_nm).unwrap().into_shape((k, k)).unwrap();

    q_nm
}

fn get_von_neumann_coefficients_iterative(
    alpha_nm: CMatrixView,
    alpha: Float,
    w_n_arr: RVectorView,
    t_n_arr: RVectorView,
) -> CMatrix {
    let k = alpha_nm.shape()[0];

    let q_nm: CMatrix = bicgstab(
        |v: CVectorView| -> CVector {
            let mut v_out: CVector = CVector::zeros(k * k);
            for nm in 0..(k * k) {
                let mut tmp: CVector = CVector::zeros(k * k);
                let w_n = w_n_arr[nm / k];
                let t_m = t_n_arr[nm % k];
                for (i, w_i) in w_n_arr.iter().enumerate() {
                    for (j, t_j) in t_n_arr.iter().enumerate() {
                        tmp[i * k + j] = Complex::new(
                            -0.5 * alpha * (w_i - w_n).powi(2)
                                - (1.0 / (8.0 * alpha)) * (t_j - t_m).powi(2),
                            0.5 * (w_i - w_n) * (t_j + t_m),
                        )
                        .exp()
                    }
                }
                v_out[nm] = tmp.dot(&v);
            }
            v_out
        },
        alpha_nm.into_shape(k * k).unwrap().view(),
        None,
        None,
        None,
    )
    .unwrap()
    .into_shape((k, k))
    .unwrap();
    q_nm
}

pub fn vnt_direct(
    signal: CVectorView,
    w_min: Float,
    w_max: Float,
) -> (CMatrix, VNTExtent, Option<Array3<Complex<Float>>>) {
    let nn = signal.len();
    let (w_grid, t_span, w_n_arr, t_n_arr, alpha) = get_grid(w_min, w_max, nn);
    let extent = VNTExtent {
        t_min: -0.5 * t_span,
        t_max: 0.5 * t_span,
        w_min,
        w_max,
    };
    let alpha_nmo = get_full_basis_functions(w_grid.view(), w_n_arr.view(), t_n_arr.view(), alpha);

    let q_nm = get_von_neumann_coefficients_direct(
        signal.view(),
        alpha_nmo.view(),
        alpha,
        w_n_arr.view(),
        t_n_arr.view(),
    );

    (q_nm, extent, Some(alpha_nmo))
}

pub fn vnt_iterative(
    signal: CVectorView,
    w_min: Float,
    w_max: Float,
) -> (CMatrix, VNTExtent, Option<Array3<Complex<Float>>>) {
    let nn = signal.len();
    let (w_grid, t_span, w_n_arr, t_n_arr, alpha) = get_grid(w_min, w_max, nn);
    let extent = VNTExtent {
        t_min: -0.5 * t_span,
        t_max: 0.5 * t_span,
        w_min,
        w_max,
    };
    let alpha_nm = get_contracted_basis_functions(
        signal.view(),
        w_grid.view(),
        w_n_arr.view(),
        t_n_arr.view(),
        alpha,
    );
    let q_nm = get_von_neumann_coefficients_iterative(
        alpha_nm.view(),
        alpha,
        w_n_arr.view(),
        t_n_arr.view(),
    );

    (q_nm, extent, None)
}

pub fn ivnt_direct(q_nm: CMatrixView, alpha_nmo: ArrayView3<Complex<Float>>) -> CVector {
    let k = q_nm.shape()[0];
    let signal: CVector = q_nm
        .into_shape(k * k)
        .unwrap()
        .dot(&alpha_nmo.into_shape((k * k, k * k)).unwrap());
    signal
}

pub fn ivnt_iterative(q_nm: CMatrixView, extent: &VNTExtent) -> CVector {
    let k = q_nm.shape()[0];
    let (w_grid, _t_span, w_n_arr, t_n_arr, alpha) = get_grid(extent.w_min, extent.w_max, k * k);

    let mut signal: CVector = CVector::zeros(k * k);
    for (i, w_n) in w_n_arr.iter().enumerate() {
        for (j, t_m) in t_n_arr.iter().enumerate() {
            let alpha_nm_vec = get_alpha_nm_vec(w_grid.view(), alpha, *w_n, *t_m);
            signal += &(q_nm[[i, j]] * alpha_nm_vec);
        }
    }
    signal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray_complex::carray_abs_diff_eq;
    use crate::test_refs::{
        get_ref_alpha_nm, get_ref_basis_functions, get_ref_grid, get_ref_ivnt, get_ref_ovlp_matrix,
        get_ref_pulse, get_ref_vnt, get_ref_von_neumann_coefficients, OMEGA_MAX, OMEGA_MIN,
    };
    use approx::RelativeEq;

    #[test]
    fn test_get_grid() {
        let signal = get_ref_pulse();
        let nn = signal.len();
        let (w_grid, t_span, w_n_arr, t_n_arr, alpha) = get_grid(OMEGA_MIN, OMEGA_MAX, nn);
        let (w_grid_ref, t_span_ref, w_n_arr_ref, t_n_arr_ref, alpha_ref) = get_ref_grid();

        assert!(w_grid.relative_eq(&w_grid_ref, 1e-14, 1e-12));
        assert!(t_span.relative_eq(&t_span_ref, 1e-14, 1e-12));
        assert!(w_n_arr.relative_eq(&w_n_arr_ref, 1e-14, 1e-12));
        assert!(t_n_arr.relative_eq(&t_n_arr_ref, 1e-14, 1e-12));
        assert!(alpha.relative_eq(&alpha_ref, 1e-14, 1e-12));
    }

    #[test]
    fn test_get_alpha_nm_vec() {
        let (w_grid, _t_span, w_n_arr, t_n_arr, alpha) = get_ref_grid();
        let k = w_n_arr.len();
        let alpha_nmo_ref = get_ref_basis_functions();
        for i in 0..k {
            for j in 0..k {
                let alpha_ij = get_alpha_nm_vec(w_grid.view(), alpha, w_n_arr[i], t_n_arr[j]);
                assert!(carray_abs_diff_eq(
                    &alpha_ij,
                    &alpha_nmo_ref.slice(s![i, j, ..]).to_owned(),
                    1e-14
                ));
            }
        }
    }

    #[test]
    fn test_get_full_basis_function() {
        let (w_grid, _t_span, w_n_arr, t_n_arr, alpha) = get_ref_grid();
        let alpha_nmo =
            get_full_basis_functions(w_grid.view(), w_n_arr.view(), t_n_arr.view(), alpha);
        let alpha_nmo_ref = get_ref_basis_functions();

        assert!(carray_abs_diff_eq(&alpha_nmo, &alpha_nmo_ref, 1e-14));
    }

    #[test]
    fn test_get_contracted_basis_functions() {
        let signal = get_ref_pulse();
        let (w_grid, _t_span, w_n_arr, t_n_arr, alpha) = get_ref_grid();

        let alpha_nm = get_contracted_basis_functions(
            signal.view(),
            w_grid.view(),
            w_n_arr.view(),
            t_n_arr.view(),
            alpha,
        );
        let alpha_nm_ref = get_ref_alpha_nm();

        assert!(carray_abs_diff_eq(&alpha_nm, &alpha_nm_ref, 1e-14));
    }

    #[test]
    fn test_get_ovlp_matrix() {
        let (_w_grid, _t_span, w_n_arr, t_n_arr, alpha) = get_ref_grid();
        let s = get_ovlp_matrix(alpha, w_n_arr.view(), t_n_arr.view());
        let s_ref = get_ref_ovlp_matrix();

        assert!(carray_abs_diff_eq(&s, &s_ref, 1e-12));
    }

    #[test]
    fn test_get_von_neumann_coefficients_direct() {
        let signal = get_ref_pulse();
        let (_w_grid, _t_span, w_n_arr, t_n_arr, alpha) = get_ref_grid();
        let alpha_nmo = get_ref_basis_functions();
        let q_nm = get_von_neumann_coefficients_direct(
            signal.view(),
            alpha_nmo.view(),
            alpha,
            w_n_arr.view(),
            t_n_arr.view(),
        );
        let q_nm_ref = get_ref_von_neumann_coefficients();
        
        println!("{:?}", q_nm);
        println!("{:?}", q_nm_ref);
        assert!(carray_abs_diff_eq(&q_nm, &q_nm_ref, 1e-10));
    }

    #[test]
    fn test_get_von_neumann_coefficients_iterative() {
        let (_w_grid, _t_span, w_n_arr, t_n_arr, alpha) = get_ref_grid();
        let alpha_nm = get_ref_alpha_nm();
        let q_nm = get_von_neumann_coefficients_iterative(
            alpha_nm.view(),
            alpha,
            w_n_arr.view(),
            t_n_arr.view(),
        );
        let q_nm_ref = get_ref_von_neumann_coefficients();

        assert!(carray_abs_diff_eq(&q_nm, &q_nm_ref, 1e-7));
    }

    fn test_vnt(
        vnt_function: fn(
            CVectorView,
            Float,
            Float,
        ) -> (CMatrix, VNTExtent, Option<Array3<Complex<Float>>>),
        epsilon: Float,
    ) {
        let signal = get_ref_pulse();
        let (q_nm, extent, alpha_nmo) = vnt_function(signal.view(), OMEGA_MIN, OMEGA_MAX);
        let (q_nm_ref, extent_ref, alpha_nm_ref) = get_ref_vnt();

        assert!(carray_abs_diff_eq(&q_nm, &q_nm_ref, epsilon));
        if alpha_nmo.is_some() {
            assert!(carray_abs_diff_eq(
                &alpha_nmo.unwrap(),
                &alpha_nm_ref,
                epsilon
            ));
        }

        assert!(extent.t_min.relative_eq(&extent_ref.t_min, 1e-14, 1e-12));
        assert!(extent.t_max.relative_eq(&extent_ref.t_max, 1e-14, 1e-12));
        assert!(extent.w_min.relative_eq(&extent_ref.w_min, 1e-14, 1e-12));
        assert!(extent.w_max.relative_eq(&extent_ref.w_max, 1e-14, 1e-12));
    }

    #[test]
    fn test_vnt_direct() {
        test_vnt(vnt_direct, 1e-10);
    }

    #[test]
    fn test_vnt_iterative() {
        test_vnt(vnt_iterative, 1e-7);
    }

    #[test]
    fn test_ivnt_direct() {
        let (q_nm, _extent, _alpha_nm) = get_ref_vnt();
        let alpha_nmo = get_ref_basis_functions();
        let signal_recon = ivnt_direct(q_nm.view(), alpha_nmo.view());
        let signal_recon_ref = get_ref_ivnt();

        assert!(carray_abs_diff_eq(&signal_recon, &signal_recon_ref, 1e-10));
    }

    #[test]
    fn test_ivnt_iterative() {
        let (q_nm, extent, _alpha_nm) = get_ref_vnt();
        let signal_recon = ivnt_iterative(q_nm.view(), &extent);
        let signal_recon_ref = get_ref_ivnt();

        println!("signal_recon: {:?}", signal_recon);
        println!("signal_recon_ref: {:?}", signal_recon_ref);
        println!("{}", &signal_recon / &signal_recon_ref);

        assert!(carray_abs_diff_eq(&signal_recon, &signal_recon_ref, 1e-7));
    }
}
