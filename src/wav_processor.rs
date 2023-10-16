use crate::types::{Float, RVector};
use hound::{SampleFormat, WavReader};
use ndarray::{Array, Axis};

pub(crate) fn read_wav(wav_path: &str) -> (Float, RVector) {
    let reader = WavReader::open(wav_path).unwrap();
    let sampling_rate = reader.spec().sample_rate as Float;
    let channel_count = reader.spec().channels as usize;
    let sample_count = reader.len() as usize;
    let data: RVector = match reader.spec().sample_format {
        SampleFormat::Float => {
            Array::from_iter(reader.into_samples::<f32>().map(|v| v.unwrap() as Float))
        }
        SampleFormat::Int => {
            let bit_depth = reader.spec().bits_per_sample;
            Array::from_iter(
                reader
                    .into_samples::<i32>()
                    .map(|v| (v.unwrap() as Float) / Float::from(2.0).powi(bit_depth as i32 - 1)),
            )
        }
    };
    let data: RVector = data
        .into_shape((sample_count / channel_count, channel_count))
        .unwrap()
        .mean_axis(Axis(1))
        .unwrap();
    (sampling_rate, data)
}
