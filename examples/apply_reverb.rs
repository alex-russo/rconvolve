//! # Apply Convolution Reverb
//!
//! A CLI tool that applies an impulse response to an audio file using convolution.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example apply_reverb -- <input_audio.wav> <impulse_response.wav> [output.wav]
//! ```
//!
//! ## Example
//!
//! ```bash
//! # Apply reverb to vocals using a hall IR
//! cargo run --example apply_reverb -- vocals.wav hall_ir.wav vocals_reverb.wav
//! ```

use anyhow::{Context, Result};
use hound::{WavReader, WavSpec, WavWriter};
use std::env;

/// Read WAV samples and convert to f32, supporting both 16-bit and 24-bit audio
fn read_wav_samples(reader: &mut WavReader<std::io::BufReader<std::fs::File>>) -> Result<Vec<f32>> {
    let spec = reader.spec();

    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => Ok(reader
            .samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to read 16-bit samples")?
            .into_iter()
            .map(|s| (s as f32) / 32768.0)
            .collect()),
        (hound::SampleFormat::Int, 24) => {
            Ok(reader
                .samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .context("Failed to read 24-bit samples")?
                .into_iter()
                .map(|s| (s as f32) / 8388608.0) // 2^23
                .collect())
        }
        (hound::SampleFormat::Int, 32) => {
            Ok(reader
                .samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .context("Failed to read 32-bit samples")?
                .into_iter()
                .map(|s| (s as f32) / 2147483648.0) // 2^31
                .collect())
        }
        (hound::SampleFormat::Float, 32) => Ok(reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to read 32-bit float samples")?),
        _ => {
            anyhow::bail!(
                "Unsupported audio format: {} bits, {:?}",
                spec.bits_per_sample,
                spec.sample_format
            )
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <input_audio.wav> <impulse_response.wav> [output.wav]",
            args[0]
        );
        eprintln!("Applies impulse response to audio file using convolution reverb");
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let ir_path = &args[2];
    let output_path = if args.len() > 3 {
        &args[3]
    } else {
        "output_reverb.wav"
    };

    println!("Convolution Reverb Tool");
    println!("Audio: {}", audio_path);
    println!("IR: {}", ir_path);
    println!("Output: {}", output_path);
    println!();

    // Read audio file
    let mut audio_reader =
        WavReader::open(audio_path).with_context(|| format!("Failed to open {}", audio_path))?;
    let audio_spec = audio_reader.spec();
    let audio_samples = read_wav_samples(&mut audio_reader)?;

    println!(
        "Audio: {} Hz, {} channels, {} samples ({:.2}s)",
        audio_spec.sample_rate,
        audio_spec.channels,
        audio_samples.len(),
        (audio_samples.len() as f32) / (audio_spec.sample_rate as f32 * audio_spec.channels as f32)
    );

    // Read impulse response file
    let mut ir_reader =
        WavReader::open(ir_path).with_context(|| format!("Failed to open {}", ir_path))?;
    let ir_spec = ir_reader.spec();
    let ir_samples = read_wav_samples(&mut ir_reader)?;

    println!(
        "IR: {} Hz, {} channels, {} samples ({:.2}s)",
        ir_spec.sample_rate,
        ir_spec.channels,
        ir_samples.len(),
        (ir_samples.len() as f32) / (ir_spec.sample_rate as f32 * ir_spec.channels as f32)
    );

    // Verify sample rates match
    if audio_spec.sample_rate != ir_spec.sample_rate {
        anyhow::bail!(
            "Sample rate mismatch: audio is {} Hz but IR is {} Hz",
            audio_spec.sample_rate,
            ir_spec.sample_rate
        );
    }

    println!("\nProcessing...");

    // Handle channel configurations
    let output = if audio_spec.channels == 1 && ir_spec.channels == 1 {
        // Mono to mono
        println!("Mode: Mono audio + Mono IR = Mono output");
        rconvolve::convolve::fft_convolve(&audio_samples, &ir_samples)?
    } else if audio_spec.channels == 2 && ir_spec.channels == 2 {
        // Stereo to stereo (process channels independently)
        println!("Mode: Stereo audio + Stereo IR = Stereo output");

        // Deinterleave audio
        let audio_left: Vec<f32> = audio_samples.iter().step_by(2).copied().collect();
        let audio_right: Vec<f32> = audio_samples.iter().skip(1).step_by(2).copied().collect();

        // Deinterleave IR
        let ir_left: Vec<f32> = ir_samples.iter().step_by(2).copied().collect();
        let ir_right: Vec<f32> = ir_samples.iter().skip(1).step_by(2).copied().collect();

        // Convolve each channel
        let left = rconvolve::convolve::fft_convolve(&audio_left, &ir_left)?;
        let right = rconvolve::convolve::fft_convolve(&audio_right, &ir_right)?;

        // Interleave output
        let mut output = Vec::with_capacity(left.len() + right.len());
        for (l, r) in left.iter().zip(right.iter()) {
            output.push(*l);
            output.push(*r);
        }
        output
    } else if audio_spec.channels == 1 && ir_spec.channels == 2 {
        // Mono to stereo (apply stereo IR to mono audio)
        println!("Mode: Mono audio + Stereo IR = Stereo output");

        // Deinterleave IR
        let ir_left: Vec<f32> = ir_samples.iter().step_by(2).copied().collect();
        let ir_right: Vec<f32> = ir_samples.iter().skip(1).step_by(2).copied().collect();

        // Convolve mono audio with each IR channel
        let left = rconvolve::convolve::fft_convolve(&audio_samples, &ir_left)?;
        let right = rconvolve::convolve::fft_convolve(&audio_samples, &ir_right)?;

        // Interleave output
        let mut output = Vec::with_capacity(left.len() + right.len());
        for (l, r) in left.iter().zip(right.iter()) {
            output.push(*l);
            output.push(*r);
        }
        output
    } else if audio_spec.channels == 2 && ir_spec.channels == 1 {
        // Stereo to stereo (apply mono IR to both channels)
        println!("Mode: Stereo audio + Mono IR = Stereo output");

        // Deinterleave audio
        let audio_left: Vec<f32> = audio_samples.iter().step_by(2).copied().collect();
        let audio_right: Vec<f32> = audio_samples.iter().skip(1).step_by(2).copied().collect();

        // Convolve each channel with mono IR
        let left = rconvolve::convolve::fft_convolve(&audio_left, &ir_samples)?;
        let right = rconvolve::convolve::fft_convolve(&audio_right, &ir_samples)?;

        // Interleave output
        let mut output = Vec::with_capacity(left.len() + right.len());
        for (l, r) in left.iter().zip(right.iter()) {
            output.push(*l);
            output.push(*r);
        }
        output
    } else {
        anyhow::bail!(
            "Unsupported channel configuration: {} audio channels, {} IR channels",
            audio_spec.channels,
            ir_spec.channels
        );
    };

    // Determine output channels
    let output_channels = if audio_spec.channels == 1 && ir_spec.channels == 1 {
        1
    } else {
        2
    };

    // Normalize output to prevent clipping
    let max_amplitude = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let normalized: Vec<f32> = if max_amplitude > 1.0 {
        println!("Normalizing output (peak was {:.2})", max_amplitude);
        output.iter().map(|s| s / max_amplitude).collect()
    } else {
        output
    };

    // Write output WAV
    let output_spec = WavSpec {
        channels: output_channels,
        sample_rate: audio_spec.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(output_path, output_spec)
        .with_context(|| format!("Failed to create {}", output_path))?;

    for &sample in &normalized {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;

    println!("\n✅ Success!");
    println!(
        "Output: {} samples ({:.2}s, {} channels)",
        normalized.len(),
        (normalized.len() as f32) / (audio_spec.sample_rate as f32 * output_channels as f32),
        output_channels
    );
    println!("Saved: {}", output_path);

    Ok(())
}
