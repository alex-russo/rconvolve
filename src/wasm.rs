//! WASM bindings for browser integration
//!
//! This module provides JavaScript-friendly functions for processing audio data
//! from Float32Array inputs in web browsers.

use crate::{convolve, deconvolve, AudioBuffer};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use js_sys::Float32Array;

#[cfg(not(feature = "std"))]
use alloc::format; // Import `format!` macro for no_std builds

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// Ensure all wasm_bindgen-related code is conditionally compiled

/// WebAssembly bindings for exponential sweep generation.
///
/// Use this to generate test signals for impulse response measurement.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmSweepGenerator;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmSweepGenerator {
    /// Generate exponential sweep and return as Float32Array
    #[wasm_bindgen]
    pub fn exponential(
        sample_rate: f32,
        duration: f32,
        start_freq: f32,
        end_freq: f32,
    ) -> Result<Float32Array, JsValue> {
        let sweep = crate::sweep::exponential(sample_rate, duration, start_freq, end_freq)
            .map_err(|e| JsValue::from_str(&format!("Sweep generation error: {:?}", e)))?;

        Ok(Float32Array::from(&sweep[..]))
    }
}

/// WebAssembly bindings for impulse response extraction.
///
/// Extracts impulse responses from recorded sweep responses using deconvolution.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmIRProcessor {
    original_sweep: AudioBuffer,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmIRProcessor {
    /// Create a new IR processor with the original sweep
    #[wasm_bindgen(constructor)]
    pub fn new(original_sweep: &Float32Array) -> WasmIRProcessor {
        let sweep_vec: Vec<f32> = original_sweep.to_vec();
        WasmIRProcessor {
            original_sweep: sweep_vec,
        }
    }

    /// Extract impulse response with custom settings
    #[wasm_bindgen]
    pub fn extract_impulse_response_advanced(
        &self,
        recorded_response: &Float32Array,
        ir_length: usize,
        regularization: f32,
    ) -> Result<Float32Array, JsValue> {
        let response_vec: Vec<f32> = recorded_response.to_vec();

        let mut config = deconvolve::DeconvolutionConfig::default();
        config.ir_length = Some(ir_length);
        config.method =
            deconvolve::DeconvolutionMethod::RegularizedFrequencyDomain { regularization };

        let ir = deconvolve::extract_ir_with_config(&self.original_sweep, &response_vec, &config)
            .map_err(|e| JsValue::from_str(&format!("IR extraction error: {:?}", e)))?;

        Ok(Float32Array::from(&ir[..]))
    }
}

/// WebAssembly bindings for real-time partitioned convolution.
///
/// Provides block-based convolution processing suitable for real-time audio.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmConvolutionProcessor {
    processor: convolve::PartitionedConvolution,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmConvolutionProcessor {
    /// Create a new convolution processor with an impulse response
    /// Uses multi-stage partitioned convolution for optimal performance with long IRs
    #[wasm_bindgen(constructor)]
    pub fn new(
        impulse_response: &Float32Array,
        block_size: usize,
    ) -> Result<WasmConvolutionProcessor, JsValue> {
        let ir_vec: Vec<f32> = impulse_response.to_vec();

        let processor = convolve::PartitionedConvolution::new(ir_vec, block_size, None)
            .map_err(|e| JsValue::from_str(&format!("Convolution processor error: {:?}", e)))?;

        Ok(WasmConvolutionProcessor { processor })
    }

    /// Process an audio block and return convolved result
    #[wasm_bindgen]
    pub fn process_block(&mut self, audio_block: &Float32Array) -> Result<Float32Array, JsValue> {
        let input_vec: Vec<f32> = audio_block.to_vec();

        let output = self
            .processor
            .process_block(&input_vec)
            .map_err(|e| JsValue::from_str(&format!("Processing error: {:?}", e)))?;

        Ok(Float32Array::from(&output[..]))
    }

    /// Reset the processor state (clears all internal buffers)
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.processor.reset();
    }
}

/// Apply convolution reverb to audio data.
///
/// Convenience function for one-shot (non-realtime) convolution.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn convolve_audio(
    audio: &Float32Array,
    impulse_response: &Float32Array,
) -> Result<Float32Array, JsValue> {
    let audio_vec: Vec<f32> = audio.to_vec();
    let ir_vec: Vec<f32> = impulse_response.to_vec();

    let result = convolve::apply_ir(&audio_vec, &ir_vec)
        .map_err(|e| JsValue::from_str(&format!("Convolution error: {:?}", e)))?;

    Ok(Float32Array::from(&result[..]))
}

// ============================================================================
// Stereo Convolution WASM Bindings
// ============================================================================

/// Stereo convolution processor for WASM
/// Processes left and right channels independently with separate IRs
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmStereoConvolutionProcessor {
    processor: convolve::StereoConvolution,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmStereoConvolutionProcessor {
    /// Create a new stereo convolution processor with separate L/R impulse responses
    #[wasm_bindgen(constructor)]
    pub fn new(
        ir_left: &Float32Array,
        ir_right: &Float32Array,
        block_size: usize,
    ) -> Result<WasmStereoConvolutionProcessor, JsValue> {
        let ir_left_vec: Vec<f32> = ir_left.to_vec();
        let ir_right_vec: Vec<f32> = ir_right.to_vec();

        let processor = convolve::StereoConvolution::new(ir_left_vec, ir_right_vec, block_size)
            .map_err(|e| JsValue::from_str(&format!("Stereo convolution error: {:?}", e)))?;

        Ok(WasmStereoConvolutionProcessor { processor })
    }

    /// Create a stereo processor from a mono IR (applies same IR to both channels)
    #[wasm_bindgen]
    pub fn from_mono(
        ir: &Float32Array,
        block_size: usize,
    ) -> Result<WasmStereoConvolutionProcessor, JsValue> {
        let ir_vec: Vec<f32> = ir.to_vec();

        let processor = convolve::StereoConvolution::from_mono(ir_vec, block_size)
            .map_err(|e| JsValue::from_str(&format!("Stereo convolution error: {:?}", e)))?;

        Ok(WasmStereoConvolutionProcessor { processor })
    }

    /// Process stereo audio blocks
    /// Returns interleaved output (L, R, L, R, ...)
    #[wasm_bindgen]
    pub fn process_block(
        &mut self,
        input_left: &Float32Array,
        input_right: &Float32Array,
    ) -> Result<Float32Array, JsValue> {
        let left_vec: Vec<f32> = input_left.to_vec();
        let right_vec: Vec<f32> = input_right.to_vec();

        let (out_left, out_right) = self
            .processor
            .process_block(&left_vec, &right_vec)
            .map_err(|e| JsValue::from_str(&format!("Processing error: {:?}", e)))?;

        // Interleave output
        let mut output = Vec::with_capacity(out_left.len() * 2);
        for i in 0..out_left.len() {
            output.push(out_left[i]);
            output.push(out_right[i]);
        }

        Ok(Float32Array::from(&output[..]))
    }

    /// Process interleaved stereo audio (L, R, L, R, ...)
    #[wasm_bindgen]
    pub fn process_interleaved(
        &mut self,
        interleaved: &Float32Array,
    ) -> Result<Float32Array, JsValue> {
        let input_vec: Vec<f32> = interleaved.to_vec();

        let output = self
            .processor
            .process_interleaved(&input_vec)
            .map_err(|e| JsValue::from_str(&format!("Processing error: {:?}", e)))?;

        Ok(Float32Array::from(&output[..]))
    }

    /// Reset the processor state
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.processor.reset();
    }
}

/// True stereo convolution processor for WASM
/// Uses a 4-channel matrix (LL, LR, RL, RR) for full spatial reverb
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmTrueStereoConvolutionProcessor {
    processor: convolve::TrueStereoConvolution,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmTrueStereoConvolutionProcessor {
    /// Create a new true stereo convolution processor with 4 IRs
    ///
    /// - ir_ll: Left input → Left output
    /// - ir_lr: Left input → Right output
    /// - ir_rl: Right input → Left output
    /// - ir_rr: Right input → Right output
    #[wasm_bindgen(constructor)]
    pub fn new(
        ir_ll: &Float32Array,
        ir_lr: &Float32Array,
        ir_rl: &Float32Array,
        ir_rr: &Float32Array,
        block_size: usize,
    ) -> Result<WasmTrueStereoConvolutionProcessor, JsValue> {
        let ir_ll_vec: Vec<f32> = ir_ll.to_vec();
        let ir_lr_vec: Vec<f32> = ir_lr.to_vec();
        let ir_rl_vec: Vec<f32> = ir_rl.to_vec();
        let ir_rr_vec: Vec<f32> = ir_rr.to_vec();

        let processor = convolve::TrueStereoConvolution::new(
            ir_ll_vec, ir_lr_vec, ir_rl_vec, ir_rr_vec, block_size,
        )
        .map_err(|e| JsValue::from_str(&format!("True stereo convolution error: {:?}", e)))?;

        Ok(WasmTrueStereoConvolutionProcessor { processor })
    }

    /// Create true stereo from a stereo IR pair with synthetic cross-feed
    ///
    /// This creates LR and RL channels by attenuating and delaying the opposite channel's IR.
    ///
    /// - cross_feed_gain: 0.0 to 1.0, typical: 0.3-0.5
    /// - cross_feed_delay_samples: typical: 10-50 at 48kHz
    #[wasm_bindgen]
    pub fn from_stereo_with_crossfeed(
        ir_left: &Float32Array,
        ir_right: &Float32Array,
        block_size: usize,
        cross_feed_gain: f32,
        cross_feed_delay_samples: usize,
    ) -> Result<WasmTrueStereoConvolutionProcessor, JsValue> {
        let ir_left_vec: Vec<f32> = ir_left.to_vec();
        let ir_right_vec: Vec<f32> = ir_right.to_vec();

        let processor = convolve::TrueStereoConvolution::from_stereo_with_crossfeed(
            ir_left_vec,
            ir_right_vec,
            block_size,
            cross_feed_gain,
            cross_feed_delay_samples,
        )
        .map_err(|e| JsValue::from_str(&format!("True stereo convolution error: {:?}", e)))?;

        Ok(WasmTrueStereoConvolutionProcessor { processor })
    }

    /// Create true stereo from a mono IR with synthetic stereo spread
    ///
    /// - spread: 0.0 = mono, 1.0 = full stereo spread
    #[wasm_bindgen]
    pub fn from_mono_with_spread(
        ir: &Float32Array,
        block_size: usize,
        spread: f32,
    ) -> Result<WasmTrueStereoConvolutionProcessor, JsValue> {
        let ir_vec: Vec<f32> = ir.to_vec();

        let processor =
            convolve::TrueStereoConvolution::from_mono_with_spread(ir_vec, block_size, spread)
                .map_err(|e| {
                    JsValue::from_str(&format!("True stereo convolution error: {:?}", e))
                })?;

        Ok(WasmTrueStereoConvolutionProcessor { processor })
    }

    /// Process stereo audio blocks with full matrix convolution
    /// Returns interleaved output (L, R, L, R, ...)
    #[wasm_bindgen]
    pub fn process_block(
        &mut self,
        input_left: &Float32Array,
        input_right: &Float32Array,
    ) -> Result<Float32Array, JsValue> {
        let left_vec: Vec<f32> = input_left.to_vec();
        let right_vec: Vec<f32> = input_right.to_vec();

        let (out_left, out_right) = self
            .processor
            .process_block(&left_vec, &right_vec)
            .map_err(|e| JsValue::from_str(&format!("Processing error: {:?}", e)))?;

        // Interleave output
        let mut output = Vec::with_capacity(out_left.len() * 2);
        for i in 0..out_left.len() {
            output.push(out_left[i]);
            output.push(out_right[i]);
        }

        Ok(Float32Array::from(&output[..]))
    }

    /// Process interleaved stereo audio (L, R, L, R, ...)
    #[wasm_bindgen]
    pub fn process_interleaved(
        &mut self,
        interleaved: &Float32Array,
    ) -> Result<Float32Array, JsValue> {
        let input_vec: Vec<f32> = interleaved.to_vec();

        let output = self
            .processor
            .process_interleaved(&input_vec)
            .map_err(|e| JsValue::from_str(&format!("Processing error: {:?}", e)))?;

        Ok(Float32Array::from(&output[..]))
    }

    /// Reset the processor state
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.processor.reset();
    }
}

/// Batch stereo convolution (non-real-time)
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stereo_convolve_audio(
    left: &Float32Array,
    right: &Float32Array,
    ir_left: &Float32Array,
    ir_right: &Float32Array,
) -> Result<Float32Array, JsValue> {
    let left_vec: Vec<f32> = left.to_vec();
    let right_vec: Vec<f32> = right.to_vec();
    let ir_left_vec: Vec<f32> = ir_left.to_vec();
    let ir_right_vec: Vec<f32> = ir_right.to_vec();

    let (out_left, out_right) =
        convolve::stereo_fft_convolve(&left_vec, &right_vec, &ir_left_vec, &ir_right_vec)
            .map_err(|e| JsValue::from_str(&format!("Stereo convolution error: {:?}", e)))?;

    // Interleave output
    let mut output = Vec::with_capacity(out_left.len() * 2);
    for i in 0..out_left.len().max(out_right.len()) {
        output.push(out_left.get(i).copied().unwrap_or(0.0));
        output.push(out_right.get(i).copied().unwrap_or(0.0));
    }

    Ok(Float32Array::from(&output[..]))
}
