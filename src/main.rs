use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::configs::OperationMode;
use crate::types::{CVector, Float, RVector};
use crate::von_neumann_transform::{ivnt_direct, ivnt_iterative, vnt_direct, vnt_iterative};
use env_logger::{Builder, Target};
use log::{debug, info, LevelFilter};
use ndarray::s;
use ndrustfft::{ndfft_r2c, ndifft_r2c, R2cFftHandler};
use plotly::common::{Anchor, ColorBar, ColorScale, ColorScalePalette, ThicknessMode, Title};
use plotly::layout::Axis;
use plotly::{HeatMap, Layout, Plot, Scatter};

use crate::defaults::{
    DEFAULT_DIRECT_CUTOFF_NPOINT, DEFAULT_FREQ_MAX, DEFAULT_NPOINT, DEFAULT_OPERATION_MODE,
};
use crate::io::{write_npz, write_png};
use clap::{value_parser, Arg, ArgAction, Command};

mod bicgstab;
mod configs;
mod defaults;
mod io;
mod linear_operator;
mod ndarray_complex;
mod test_refs;
mod types;
mod von_neumann_transform;
mod wav_processor;

const PI: Float = std::f64::consts::PI as Float;

/*
fn filter_by<I1, I2>(bs: I1, ys: I2) -> impl Iterator<Item=<I2 as Iterator>::Item>
    where I1: Iterator<Item=bool>,
          I2: Iterator {
    bs.zip(ys).filter(|x| x.0).map(|x| x.1)
}
*/

fn main() {
    // read CLI arguments
    let matches = Command::new("Music von Neumann")
        .version("0.1")
        .author("Xincheng Miao <xincheng.miao@uni-wuerzburg.de>")
        .about("Converts wave files to its von Neumann time-frequency representation")
        .arg(
            Arg::new("input")
                .value_name("wav_path")
                .help("Sets the path to input wav file")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .value_name("output")
                .short('o')
                .long("output")
                .help("Sets the prefix of output files [default: same as input file]")
                .required(false),
        )
        .arg(
            Arg::new("npoint")
                .value_name("npoint")
                .short('n')
                .long("npoint")
                .help(format!(
                    "Sets the number of points in the time domain [default: {}]",
                    DEFAULT_NPOINT
                ))
                .required(false)
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("max_freq")
                .value_name("max_freq")
                .short('f')
                .long("max_freq")
                .help(format!(
                    "Sets the maximum frequency in the frequency domain [default: {:.1}]",
                    DEFAULT_FREQ_MAX
                ))
                .required(false)
                .value_parser(value_parser!(Float)),
        )
        .arg(
            Arg::new("mode")
                .value_name("mode")
                .short('m')
                .long("mode")
                .help(format!(
                    "Sets the operation mode [default: {}]",
                    DEFAULT_OPERATION_MODE.to_string()
                ))
                .required(false)
                .value_parser(["direct", "iterative", "auto"]),
        )
        .arg(
            Arg::new("debug")
                .short('d')
                .long("debug")
                .action(ArgAction::SetTrue)
                .help("Use debug mode"),
        )
        .get_matches();

    let wav_path: PathBuf = Path::new(matches.get_one::<String>("input").unwrap()).to_path_buf();
    let png_path: PathBuf = Path::new(
        matches
            .get_one::<String>("output")
            .unwrap_or(&wav_path.to_str().unwrap().to_owned()),
    )
    .with_extension("png");
    let npz_path: PathBuf = Path::new(
        matches
            .get_one::<String>("output")
            .unwrap_or(&wav_path.to_str().unwrap().to_owned()),
    )
    .with_extension("npz");
    let npoint = *matches
        .get_one::<usize>("npoint")
        .unwrap_or(&DEFAULT_NPOINT);
    let freq_max: Float = *matches
        .get_one::<Float>("max_freq")
        .unwrap_or(&DEFAULT_FREQ_MAX);
    let mut operation_mode: OperationMode = match matches
        .get_one::<String>("mode")
        .unwrap_or(&DEFAULT_OPERATION_MODE.to_string())
        .as_str()
    {
        "direct" => OperationMode::Direct,
        "iterative" => OperationMode::Iterative,
        "auto" => OperationMode::Auto,
        _ => panic!("Invalid operation mode!"),
    };
    let debug: bool = matches.get_flag("debug");

    // initialise logger
    let mut builder = Builder::new();
    if debug {
        builder.filter_level(LevelFilter::Debug);
    } else {
        builder.filter_level(LevelFilter::Info);
    }
    builder.target(Target::Stdout);
    builder.init();

    info!("Input wav file: {}", wav_path.to_str().unwrap());
    info!("Output png file: {}", png_path.to_str().unwrap());
    info!("Output npz file: {}", npz_path.to_str().unwrap());
    info!("Number of points in the time domain: {}", npoint);
    info!("Maximum frequency in the frequency domain: {:.1}", freq_max);
    info!("Operation mode: {}", operation_mode.to_string());

    // choose operation mode if "auto" is selected
    if operation_mode == OperationMode::Auto {
        if npoint > DEFAULT_DIRECT_CUTOFF_NPOINT {
            info!("Using iterative mode due to large number of points in the time domain.");
            operation_mode = OperationMode::Iterative;
        } else {
            info!("Using direct mode due to small number of points in the time domain.");
            operation_mode = OperationMode::Direct;
        }
    }

    let start = Instant::now();
    // read wav file
    let (sampling_rate, data) = wav_processor::read_wav(wav_path.to_str().unwrap());
    let duration = start.elapsed();
    debug!("Time elapsed in reading wav file is: {:?}", duration);

    // trim signal
    let start = Instant::now();
    let ntime = (((npoint - 1) as Float) * sampling_rate / freq_max).ceil() as usize;
    let data = data.slice(s![0..ntime]).to_owned();
    let duration = start.elapsed();
    debug!("Time elapsed in signal processing is: {:?}", duration);

    // perform FFT on audio data
    let start = Instant::now();
    let dt = 1.0 / sampling_rate;
    let mut signal = CVector::zeros(ntime / 2 + 1);
    let mut handler = R2cFftHandler::<Float>::new(ntime);
    ndfft_r2c(&data, &mut signal, &mut handler, 0);

    let isig_max = (dt * (ntime as Float) * freq_max).floor() as usize;
    let freq: RVector = RVector::linspace(
        0.0,
        (isig_max as Float) * sampling_rate / (ntime as Float),
        isig_max + 1,
    );

    let signal = signal.slice(s![..isig_max + 1]).to_owned();
    assert_eq!(signal.len(), freq.len());
    assert_eq!(signal.len(), npoint);
    let duration = start.elapsed();
    debug!("Time elapsed in FFT is: {:?}", duration);

    let w_min = 0.0 * 2.0 * PI;
    let w_max = freq[freq.len() - 1] * 2.0 * PI;

    // perform von Neumann transform
    let start = Instant::now();
    let (q_nm, extent, alpha_nmo) = if operation_mode == OperationMode::Direct {
        vnt_direct(signal.view(), w_min, w_max)
    } else if operation_mode == OperationMode::Iterative {
        vnt_iterative(signal.view(), w_min, w_max)
    } else {
        panic!("Invalid operation mode!")
    };
    let duration = start.elapsed();
    debug!("Time elapsed in VNT is: {:?}", duration);

    // calculate amplitude and phase
    let start = Instant::now();
    let amplitude = q_nm.mapv(|x| x.norm());
    let phase = q_nm.mapv(|x| x.arg());
    let duration = start.elapsed();
    debug!("Time elapsed in amp/phase creation is: {:?}", duration);

    // write output files
    write_npz(npz_path.to_str().unwrap(), q_nm.view(), &extent);
    write_png(
        png_path.to_str().unwrap(),
        amplitude.view(),
        phase.view(),
        &extent,
    );

    // perform inverse von Neumann transform
    // and visualise the results for debugging
    if debug {
        let time: RVector = RVector::linspace(0.0, ((ntime - 1) as Float) * dt, ntime);

        let signal_recon = if operation_mode == OperationMode::Direct {
            ivnt_direct(q_nm.view(), alpha_nmo.unwrap().view())
        } else if operation_mode == OperationMode::Iterative {
            ivnt_iterative(q_nm.view(), &extent)
        } else {
            panic!("Invalid operation mode!")
        };

        let wave_recon_len = (signal_recon.len() - 1) * 2;
        let mut wave_recon = RVector::zeros(wave_recon_len);
        let mut handler = R2cFftHandler::<Float>::new(wave_recon_len);
        ndifft_r2c(&signal_recon, &mut wave_recon, &mut handler, 0);
        let time_recon = RVector::linspace(
            0.0,
            ((wave_recon_len - 1) as Float) / (2.0 * freq[freq.len() - 1]),
            wave_recon_len,
        );

        // constants for plot layout
        const FIGURE_HEIGHT: usize = 900;
        const FIGURE_WIDTH: usize = 1360;
        const SIGNAL_TRACE_HEIGHT_FRACTION: Float = 0.28;
        const HEATMAP_TRACE_HEIGHT_FRACTION: Float = 0.6;
        const HEATMAP_TRACE_WSPACE_FRACTION: Float = 0.1;
        const COLORBAR_WIDTH_FRACTION: Float = 0.05;

        let audio_trace = Scatter::from_array(
            time.clone(),
            data.clone() / data.iter().fold(0.0, |acc: Float, x| acc.max(x.abs())),
        )
        .name("original audio");
        let audio_recon_trace = Scatter::from_array(
            time_recon.clone(),
            wave_recon.clone()
                / wave_recon
                    .iter()
                    .fold(0.0, |acc: Float, x| acc.max(x.abs())),
        )
        .name("reconstructed audio");

        // time dimension in the VNT plane in s
        let vnt_time = RVector::linspace(extent.t_min, extent.t_max, amplitude.ncols());
        // frequency dimension in the VNT plane in Hz
        let vnt_freq =
            RVector::linspace(extent.w_min, extent.w_max, amplitude.nrows()) / (2.0 * PI);

        let amplitude_trace = HeatMap::new_z(amplitude.outer_iter().map(|x| x.to_vec()).collect())
            .x(vnt_time.to_vec())
            .y(vnt_freq.to_vec())
            .x_axis("x2")
            .y_axis("y2")
            .color_bar(
                ColorBar::new()
                    .title(Title::new("amplitude"))
                    .x_anchor(Anchor::Left)
                    .x(0.5 - 1.5 * COLORBAR_WIDTH_FRACTION)
                    .x_pad(0.0)
                    .y_anchor(Anchor::Bottom)
                    .y(0.0)
                    .y_pad(0.0)
                    .len_mode(ThicknessMode::Pixels)
                    .len(
                        ((FIGURE_HEIGHT as Float) * (HEATMAP_TRACE_HEIGHT_FRACTION - 0.11))
                            as usize,
                    ),
            )
            .color_scale(ColorScale::Palette(ColorScalePalette::Viridis));
        let phase_trace = HeatMap::new_z(phase.outer_iter().map(|x| x.to_vec()).collect())
            .x(vnt_time.to_vec())
            .y(vnt_freq.to_vec())
            .x_axis("x3")
            .y_axis("y3")
            .color_bar(
                ColorBar::new()
                    .title(Title::new("phase"))
                    .x_anchor(Anchor::Left)
                    .x(1.0 - 0.5 * COLORBAR_WIDTH_FRACTION)
                    .x_pad(0.0)
                    .y_anchor(Anchor::Bottom)
                    .y(0.0)
                    .y_pad(0.0)
                    .len_mode(ThicknessMode::Pixels)
                    .len(
                        ((FIGURE_HEIGHT as Float) * (HEATMAP_TRACE_HEIGHT_FRACTION - 0.11))
                            as usize,
                    ),
            )
            .color_scale(ColorScale::Palette(ColorScalePalette::Rainbow));

        // Set up plot layout
        let layout = Layout::new()
            .x_axis(
                Axis::new()
                    .title(Title::new("time / s"))
                    .domain(&[0.0, 1.0])
                    .anchor("y1"), //.range(vec![extent.0, extent.1]),
            )
            .y_axis(
                Axis::new()
                    .title(Title::new("amplitude"))
                    .domain(&[1.0 - SIGNAL_TRACE_HEIGHT_FRACTION, 1.0])
                    .anchor("x1"), //.range(vec![extent.2, extent.3]),
            )
            .x_axis2(
                Axis::new()
                    .title(Title::new("time / s"))
                    .domain(&[
                        0.0,
                        0.5 - 0.5 * HEATMAP_TRACE_WSPACE_FRACTION - COLORBAR_WIDTH_FRACTION,
                    ])
                    .anchor("y2"),
            )
            .y_axis2(
                Axis::new()
                    .title(Title::new("frequency / Hz"))
                    .domain(&[0.0, HEATMAP_TRACE_HEIGHT_FRACTION])
                    .anchor("x2"),
            )
            .x_axis3(
                Axis::new()
                    .title(Title::new("time / s"))
                    .domain(&[
                        0.5 + 0.5 * HEATMAP_TRACE_WSPACE_FRACTION,
                        1.0 - COLORBAR_WIDTH_FRACTION,
                    ])
                    .anchor("y3"),
            )
            .y_axis3(
                Axis::new()
                    .title(Title::new("frequency / Hz"))
                    .domain(&[0.0, HEATMAP_TRACE_HEIGHT_FRACTION])
                    .anchor("x3"),
            )
            .auto_size(true)
            .height(FIGURE_HEIGHT)
            .width(FIGURE_WIDTH);

        let mut plot = Plot::new();
        plot.add_trace(audio_trace);
        plot.add_trace(audio_recon_trace);
        plot.add_trace(amplitude_trace);
        plot.add_trace(phase_trace);
        plot.set_layout(layout);

        plot.show();
    }
}
