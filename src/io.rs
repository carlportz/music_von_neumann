use crate::types::{CMatrixView, Float, RMatrixView, RVector};
use crate::von_neumann_transform::VNTExtent;
use ndarray::{array, s};
use ndarray_npy::NpzWriter;
use png::{BitDepth, ColorType};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const PI: Float = std::f64::consts::PI as Float;

pub fn write_npz(npz_path_str: &str, q_nm: CMatrixView, extent: &VNTExtent) {
    let mut npz = NpzWriter::new(File::create(npz_path_str).unwrap());
    npz.add_array("q_nm", &q_nm).unwrap();
    npz.add_array(
        "extent",
        &array![extent.t_min, extent.t_max, extent.w_min, extent.w_max,],
    )
    .unwrap();
    npz.finish().unwrap();
}

pub fn write_png(
    png_path_str: &str,
    amplitude: RMatrixView,
    phase: RMatrixView,
    extent: &VNTExtent,
) {
    let png_path = Path::new(png_path_str);
    let png_file = File::create(png_path).unwrap();
    let ref mut w = BufWriter::new(png_file);

    let mut encoder = png::Encoder::new(w, amplitude.ncols() as u32, amplitude.nrows() as u32);
    encoder.set_color(ColorType::Rgb);
    encoder.set_depth(BitDepth::Eight);
    encoder
        .add_text_chunk(
            "extent".to_string(),
            format!(
                "extent: {{ t_min: {:20.12}, t_max: {:20.12}, w_min: {:20.12}, w_max: {:20.12} }}",
                extent.t_min, extent.t_max, extent.w_min, extent.w_max,
            ),
        )
        .unwrap();

    let mut writer = encoder.write_header().unwrap();

    let mut quantised_amplitude =
        RVector::from_iter(amplitude.slice(s![..;-1, ..]).iter().cloned());
    quantised_amplitude /= quantised_amplitude
        .iter()
        .fold(0.0, |acc: Float, &x| acc.max(x));
    quantised_amplitude *= 255.0;
    let quantised_amplitude = quantised_amplitude.mapv(|x| x.round() as u8);

    let mut quantised_phase = RVector::from_iter(phase.slice(s![..;-1, ..]).iter().cloned());
    quantised_phase = (quantised_phase + PI) / (2.0 * PI);
    quantised_phase *= 255.0;
    let quantised_phase = quantised_phase.mapv(|x| x.round() as u8);

    let mut png_data = Vec::new();
    for (qa, qp) in quantised_amplitude.iter().zip(quantised_phase.iter()) {
        png_data.push(*qa);
        png_data.push(*qp);
        png_data.push(0);
    }

    writer.write_image_data(&png_data).unwrap();
}
