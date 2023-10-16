#![allow(dead_code)]

use crate::ndarray_complex::Conj;
use crate::types::{CMatrix, CVector, Float, RVector, RVectorView};
use crate::von_neumann_transform::VNTExtent;
use ndarray::{Array1, Array3};
use ndarray_npy::NpzReader;
use num_complex::Complex;
use std::fs::File;

const PI: Float = std::f64::consts::PI as Float;

pub const OMEGA0: Float = 1e-15 * (2.0 * PI * 299792458.0) / (800.0e-9);
pub const SIGMA_OMEGA: Float = 0.025;
pub const CHIRP: Float = 30000.0;
pub const OMEGA_MIN: Float = 2.27;
pub const OMEGA_MAX: Float = 2.47;
pub const NPOINTS: usize = 121;

pub fn pulse(w: RVectorView, w0: Float, sigma: Float, chirp: Float) -> CVector {
    let amplitude: CVector =
        w.mapv(|w_i| Complex::new((-0.5 * ((w_i - w0) / sigma).powi(2)).exp(), 0.0));
    let phase_factor: CVector = w.mapv(|w_i| {
        Complex::from_polar(
            1.0,
            (0.5 * chirp * (w_i - w0).powi(2) + PI) % (2.0 * PI) - PI,
        )
    });
    amplitude * phase_factor
}

pub fn get_ref_pulse() -> CVector {
    let mut npz = NpzReader::new(File::open("tests/test_refs.npz").unwrap()).unwrap();
    let signal = npz.by_name("signal.npy").unwrap();
    signal
}

pub fn get_ref_grid() -> (RVector, Float, RVector, RVector, Float) {
    let mut npz = NpzReader::new(File::open("tests/test_refs.npz").unwrap()).unwrap();

    let w_grid = npz.by_name("w_grid.npy").unwrap();
    let t_span_arr: Array1<Float> = npz.by_name("t_span.npy").unwrap();
    let w_n_arr = npz.by_name("w_n_arr.npy").unwrap();
    let t_n_arr = npz.by_name("t_n_arr.npy").unwrap();
    let alpha_arr: Array1<Float> = npz.by_name("alpha.npy").unwrap();

    let t_span = t_span_arr[0];
    let alpha = alpha_arr[0];

    (w_grid, t_span, w_n_arr, t_n_arr, alpha)
}

pub fn get_ref_basis_functions() -> Array3<Complex<Float>> {
    let mut npz = NpzReader::new(File::open("tests/test_refs.npz").unwrap()).unwrap();
    let alpha_nmo = npz.by_name("alpha_nmo.npy").unwrap();
    alpha_nmo
}

pub fn get_ref_alpha_nm() -> CMatrix {
    let mut npz = NpzReader::new(File::open("tests/test_refs.npz").unwrap()).unwrap();
    let signal: CVector = npz.by_name("signal.npy").unwrap();
    let alpha_nmo: Array3<Complex<Float>> = npz.by_name("alpha_nmo.npy").unwrap();
    let alpha_nm: CMatrix = alpha_nmo.map_axis(ndarray::Axis(2), |row| row.conj().dot(&signal));
    alpha_nm
}

pub fn get_ref_ovlp_matrix() -> CMatrix {
    let mut npz = NpzReader::new(File::open("tests/test_refs.npz").unwrap()).unwrap();
    let s = npz.by_name("s.npy").unwrap();
    s
}

pub fn get_ref_von_neumann_coefficients() -> CMatrix {
    let mut npz = NpzReader::new(File::open("tests/test_refs.npz").unwrap()).unwrap();
    let q_nm = npz.by_name("q_nm.npy").unwrap();
    q_nm
}

pub fn get_ref_vnt() -> (CMatrix, VNTExtent, Array3<Complex<Float>>) {
    let mut npz = NpzReader::new(File::open("tests/test_refs.npz").unwrap()).unwrap();
    // let signal: CVector = npz.by_name("signal.npy").unwrap();
    let q_nm = npz.by_name("vnt_q_nm.npy").unwrap();
    let extent_arr: Array1<Float> = npz.by_name("vnt_extent.npy").unwrap();
    let alpha_nmo: Array3<Complex<Float>> = npz.by_name("vnt_alpha_nmo.npy").unwrap();
    // let alpha_nm: CMatrix = alpha_nmo.map_axis(ndarray::Axis(2), |row| row.conj().dot(&signal));

    let extent = VNTExtent {
        t_min: extent_arr[0],
        t_max: extent_arr[1],
        w_min: extent_arr[2],
        w_max: extent_arr[3],
    };

    (q_nm, extent, alpha_nmo)
}

pub fn get_ref_ivnt() -> CVector {
    let mut npz = NpzReader::new(File::open("tests/test_refs.npz").unwrap()).unwrap();
    let signal_recon = npz.by_name("signal_recon.npy").unwrap();
    signal_recon
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray_complex::carray_abs_diff_eq;
    use crate::types::RVector;

    #[test]
    fn test_pulse() {
        let w_grid = RVector::linspace(OMEGA_MIN, OMEGA_MAX, NPOINTS);
        let signal = pulse(w_grid.view(), OMEGA0, SIGMA_OMEGA, CHIRP);
        let signal_ref = get_ref_pulse();

        assert!(carray_abs_diff_eq(&signal, &signal_ref, 1e-14));
    }
}
