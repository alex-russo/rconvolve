//! # Real-time Convolution Reverb
//!
//! Demonstrates real-time audio processing using PartitionedConvolution.
//! Captures audio from the default input device (microphone), applies
//! convolution reverb using an impulse response, and outputs to speakers.
//!
//! ## Usage
//!
//! ```bash
//! # Use a WAV file as the impulse response
//! cargo run --example realtime_reverb -- impulse_response.wav
//!
//! # Or generate a simple test IR
//! cargo run --example realtime_reverb -- --test-ir
//! ```
//!
//! ## Controls
//!
//! Press Ctrl+C to stop.
//!
//! ## Requirements
//!
//! - Working audio input device (microphone)
//! - Working audio output device (speakers/headphones)
//! - Impulse response WAV file (or use --test-ir flag)

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use hound::WavReader;
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::HeapRb;
use std::env;
use std::sync::{Arc, Mutex};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    println!("Real-time Convolution Reverb");
    println!("============================");
    println!();

    // Load or generate impulse response
    let ir = if args.len() > 1 {
        if args[1] == "--test-ir" {
            println!("Generating test impulse response (short decay)...");
            generate_test_ir(48000.0)
        } else {
            println!("Loading impulse response: {}", args[1]);
            load_ir_from_wav(&args[1])?
        }
    } else {
        println!("No IR provided, generating test impulse response...");
        println!("Usage: {} [impulse_response.wav | --test-ir]", args[0]);
        println!();
        generate_test_ir(48000.0)
    };

    println!(
        "IR length: {} samples ({:.3}s)",
        ir.len(),
        ir.len() as f32 / 48000.0
    );
    println!();

    // Set up audio
    let host = cpal::default_host();

    let input_device = host
        .default_input_device()
        .context("No input device available")?;
    let output_device = host
        .default_output_device()
        .context("No output device available")?;

    println!("Input device: {}", input_device.name()?);
    println!("Output device: {}", output_device.name()?);

    // Verify the devices support this config
    let supported_input_config = input_device
        .default_input_config()
        .context("Failed to get default input config")?;
    let supported_output_config = output_device
        .default_output_config()
        .context("Failed to get default output config")?;

    println!(
        "Actual input config: {} Hz, {} channels",
        supported_input_config.sample_rate().0,
        supported_input_config.channels()
    );
    println!(
        "Actual output config: {} Hz, {} channels",
        supported_output_config.sample_rate().0,
        supported_output_config.channels()
    );
    println!();

    let input_config = StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(48000),
        buffer_size: cpal::BufferSize::Fixed(512),
    };

    let output_config = StreamConfig {
        channels: supported_output_config.channels(),
        sample_rate: cpal::SampleRate(48000),
        buffer_size: cpal::BufferSize::Fixed(512),
    };

    println!("Sample rate: {} Hz", input_config.sample_rate.0);
    println!("Input channels: {}", input_config.channels);
    println!("Output channels: {}", output_config.channels);
    println!("Block size: 512 samples");
    println!();

    // Create convolution processor using uniform partitioned convolution
    let bypass_mode = false; // Set to true to bypass convolution for testing
    let wet_only = false; // Set to true to hear only the reverb (wet signal)

    println!("Using uniform partitioned convolution (UPC)");
    if wet_only {
        println!("Mode: WET ONLY (reverb only, no dry signal)");
    }

    // Use partition size of 2048 samples for good efficiency/latency tradeoff
    let processor = rconvolve::convolve::PartitionedConvolution::new(ir, 512, Some(512))?;

    let latency = 512; // Block size latency
    println!(
        "Processing latency: {} samples ({:.1}ms)",
        latency,
        latency as f32 / 48000.0 * 1000.0
    );
    println!();

    let processor = Arc::new(Mutex::new(processor));

    // Create ring buffer for passing audio between input and output streams
    let ring_buffer = HeapRb::<f32>::new(48000); // 1 second buffer
    let (mut producer, mut consumer) = ring_buffer.split();

    // Pre-fill with silence to account for processing latency
    for _ in 0..1024 {
        let _ = producer.try_push(0.0);
    }

    // Input stream (capture from microphone)
    let processor_input = processor.clone();
    let input_stream = input_device.build_input_stream(
        &input_config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if bypass_mode {
                // Bypass mode - just pass through the audio
                for &sample in data {
                    let _ = producer.try_push(sample);
                }
            } else {
                let mut proc = processor_input.lock().unwrap();

                // Handle variable buffer sizes by processing in chunks
                for chunk in data.chunks(512) {
                    if chunk.len() == 512 {
                        if let Ok(output) = proc.process_block(chunk) {
                            // Apply moderate gain reduction
                            let gain = 0.5;
                            for (i, sample) in output.iter().enumerate() {
                                let wet = sample * gain;
                                let dry = chunk[i];
                                let final_sample = if wet_only { wet } else { dry + wet };
                                let _ = producer.try_push(final_sample);
                            }
                        }
                    } else {
                        // If not a full block, pad with zeros
                        let mut padded = chunk.to_vec();
                        padded.resize(512, 0.0);
                        if let Ok(output) = proc.process_block(&padded) {
                            // Only push the valid samples (matching input length)
                            let gain = 0.5;
                            for (i, sample) in output.iter().take(chunk.len()).enumerate() {
                                let wet = sample * gain;
                                let dry = chunk[i];
                                let final_sample = if wet_only { wet } else { dry + wet };
                                let _ = producer.try_push(final_sample);
                            }
                        }
                    }
                }
            }
        },
        |err| eprintln!("Input stream error: {}", err),
        None,
    )?; // Output stream (play to speakers)
    let output_channels = output_config.channels as usize;
    let output_stream = output_device.build_output_stream(
        &output_config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for frame in data.chunks_mut(output_channels) {
                let sample = consumer.try_pop().unwrap_or(0.0);
                // Duplicate mono sample to all output channels
                for channel_sample in frame.iter_mut() {
                    *channel_sample = sample;
                }
            }
        },
        |err| eprintln!("Output stream error: {}", err),
        None,
    )?;

    // Start streaming
    input_stream.play()?;
    output_stream.play()?;

    println!("🎤 Real-time reverb active!");
    println!("Speak into your microphone to hear the reverb effect.");
    println!("Press Ctrl+C to stop...");
    println!();

    // Keep running until Ctrl+C
    std::thread::park();

    Ok(())
}

/// Load impulse response from WAV file
fn load_ir_from_wav(path: &str) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path).with_context(|| format!("Failed to open {}", path))?;

    let spec = reader.spec();

    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|s| (s as f32) / 32768.0)
            .collect(),
        (hound::SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|s| (s as f32) / 8388608.0)
            .collect(),
        (hound::SampleFormat::Float, 32) => {
            reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?
        }
        _ => anyhow::bail!(
            "Unsupported WAV format: {} bits, {:?}",
            spec.bits_per_sample,
            spec.sample_format
        ),
    };

    // Convert stereo to mono if needed
    let mono_samples = if spec.channels == 2 {
        samples
            .chunks_exact(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok(mono_samples)
}

/// Generate a simple test impulse response (exponential decay)
fn generate_test_ir(sample_rate: f32) -> Vec<f32> {
    let duration = 2.0;
    let length = (sample_rate * duration) as usize;
    let mut ir = vec![0.0; length];

    // initial impulse
    ir[0] = 1.0;

    // early reflections (random small peaks)
    for i in 1..500 {
        ir[i] += (rand::random::<f32>() - 0.5) * 0.2;
    }

    // dense reverb tail
    for i in 500..length {
        let t = i as f32 / sample_rate;
        let decay = (-1.2 * t).exp(); // approx RT60 ~ 1.5s
        let noise = rand::random::<f32>() * 2.0 - 1.0;
        ir[i] += noise * decay * 0.3;
    }

    // normalize
    let max = ir.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    if max > 0.0 {
        for v in &mut ir {
            *v /= max;
        }
    }

    ir
}
