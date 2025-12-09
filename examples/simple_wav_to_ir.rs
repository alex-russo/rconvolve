//! # Simple WAV to IR Converter
//!
//! A minimal CLI tool that converts WAV recordings to impulse response files.
//! Uses the same default settings as the HTML measurement app.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example simple_wav_to_ir -- input.wav [output.wav]
//! ```
//!
//! ## Default Settings
//!
//! - Sample rate: 48 kHz
//! - Sweep: 2.0 seconds, 20 Hz to 20 kHz
//! - IR length: 2.0 seconds
//! - Regularization: 0.001

use anyhow::{Context, Result};
use hound::{WavReader, WavSpec, WavWriter};
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input.wav> [output.wav]", args[0]);
        eprintln!("Converts recorded sweep WAV to impulse response");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = if args.len() > 2 {
        &args[2]
    } else {
        "output_ir.wav"
    };

    const SAMPLE_RATE: f32 = 48000.0;
    const DURATION: f32 = 2.0;
    const START_FREQ: f32 = 20.0;
    const END_FREQ: f32 = 20000.0;
    const IR_LENGTH: f32 = 2.0; // Match sweep duration for optimal results
    const REGULARIZATION: f32 = 0.001; // Proper regularization value

    println!("Simple WAV to IR Converter");
    println!("Input: {}", input_path);
    println!("Output: {}", output_path);

    // Read WAV file
    let mut reader =
        WavReader::open(input_path).with_context(|| format!("Failed to open {}", input_path))?;

    let spec = reader.spec();
    println!(
        "Input: {} Hz, {} channels, {} samples",
        spec.sample_rate,
        spec.channels,
        reader.len()
    );

    // Convert samples to f32 (interleaved for stereo)
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to read samples")?
        .into_iter()
        .map(|s| (s as f32) / 32768.0)
        .collect();

    // Generate reference sweep
    let sweep = rconvolve::sweep::exponential(SAMPLE_RATE, DURATION, START_FREQ, END_FREQ)?;

    println!("Processing...");
    println!("Sweep: {} samples ({:.1}s)", sweep.len(), DURATION);
    println!(
        "Recording: {} samples ({:.1}s)",
        samples.len(),
        (samples.len() as f32) / (SAMPLE_RATE * (spec.channels as f32))
    );

    // Set up deconvolution
    let ir_length_samples = (IR_LENGTH * SAMPLE_RATE) as usize;
    let mut config = rconvolve::deconvolve::DeconvolutionConfig::default();
    config.ir_length = Some(ir_length_samples);
    config.method = rconvolve::deconvolve::DeconvolutionMethod::RegularizedFrequencyDomain {
        regularization: REGULARIZATION,
    };

    // Extract impulse response using the new stereo-aware function
    let ir = rconvolve::deconvolve::extract_ir_from_interleaved_with_config(
        &sweep,
        &samples,
        spec.channels as usize,
        &config,
    )?;

    // Always output stereo
    let output_channels = 2;

    println!(
        "Debug: Input channels: {}, Output channels: {}, IR length: {}",
        spec.channels,
        output_channels,
        ir.len()
    );

    // Write output WAV
    let output_spec = WavSpec {
        channels: output_channels,
        sample_rate: SAMPLE_RATE as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(output_path, output_spec)
        .with_context(|| format!("Failed to create {}", output_path))?;

    for &sample in &ir {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;

    println!("✅ Success!");
    println!(
        "IR: {} samples ({:.2}s, {} channels)",
        ir.len(),
        (ir.len() as f32) / (SAMPLE_RATE * (output_channels as f32)),
        output_channels
    );
    println!("Saved: {}", output_path);

    Ok(())
}
