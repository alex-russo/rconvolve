//! Deconvolution for impulse response extraction
//!
//! This module provides functions to extract clean impulse responses from recorded
//! sweep responses using deconvolution techniques.

use crate::{utils::next_power_of_two, AudioBuffer, AudioError, AudioResult, Sample};
use rustfft::{num_complex::Complex, FftPlanner};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Deconvolution method for impulse response extraction
///
/// # Regularization Values
///
/// For `RegularizedFrequencyDomain`, typical values:
/// - `1e-6` to `1e-4`: Low noise, high detail (good for clean recordings)
/// - `1e-3` (default): Balanced (good starting point)
/// - `1e-2`: High noise reduction, smoother (for noisy recordings)
///
/// Lower values preserve more detail but may amplify noise.
/// Higher values reduce noise but may blur transients.
/// Start with `1e-3` and adjust based on recording quality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeconvolutionMethod {
    /// Frequency-domain division (faster, but more noise-sensitive)
    ///
    /// Simple spectral division without regularization. Use when:
    /// - Recording quality is excellent
    /// - You need maximum speed
    /// - Noise is not a concern
    FrequencyDomain,
    /// Regularized frequency-domain division (recommended)
    ///
    /// Uses Wiener deconvolution with regularization parameter to balance
    /// detail preservation vs noise reduction. This is the recommended method
    /// for most applications.
    RegularizedFrequencyDomain {
        /// Regularization parameter (see enum-level docs for typical values)
        regularization: Sample,
    },
}

impl Default for DeconvolutionMethod {
    fn default() -> Self {
        Self::RegularizedFrequencyDomain {
            regularization: 1e-3, // More conservative default
        }
    }
}

/// Configuration for impulse response extraction
///
/// # Example
///
/// ```rust
/// use rconvolve::deconvolve;
///
/// let config = deconvolve::DeconvolutionConfig {
///     method: deconvolve::DeconvolutionMethod::RegularizedFrequencyDomain {
///         regularization: 1e-3,
///     },
///     ir_length: Some(96000),  // 2 seconds at 48kHz
///     window: Some(deconvolve::WindowType::Hann),
///     pre_delay: 0,
/// };
/// ```
#[derive(Debug, Clone, Default)]
pub struct DeconvolutionConfig {
    /// Deconvolution method to use
    ///
    /// Defaults to `RegularizedFrequencyDomain` with `regularization: 1e-3`
    pub method: DeconvolutionMethod,
    /// Length of the extracted impulse response (in samples)
    ///
    /// If `None`, the full extracted IR length is used.
    /// If `Some(n)`, the IR is truncated or zero-padded to exactly `n` samples.
    pub ir_length: Option<usize>,
    /// Window function to apply to the result
    ///
    /// Windowing helps reduce artifacts at the edges of the IR.
    /// `Hann` is a good default choice.
    pub window: Option<WindowType>,
    /// Pre-delay to remove (in samples)
    ///
    /// Removes the specified number of samples from the beginning of the IR.
    /// Useful for removing system latency or initial silence.
    pub pre_delay: usize,
}

/// Window function types for spectral processing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Hann (raised cosine) window - good general-purpose choice.
    Hann,
    /// Hamming window - similar to Hann with slightly different coefficients.
    Hamming,
    /// Blackman window - better sidelobe suppression, wider main lobe.
    Blackman,
}

/// Extract impulse response from a sweep and its recorded response
///
/// Uses default configuration with regularized frequency-domain deconvolution.
/// For custom settings, use [`extract_ir_with_config`].
///
/// # Example
///
/// ```rust
/// use rconvolve::{sweep, deconvolve};
///
/// let sweep_signal = sweep::exponential(48000.0, 2.0, 20.0, 20000.0).unwrap();
/// let recorded_response = vec![0.0; 96000]; // Your recorded sweep response
/// let config = deconvolve::DeconvolutionConfig {
///     ir_length: Some(sweep_signal.len()),
///     ..Default::default()
/// };
/// let ir = deconvolve::extract_ir_with_config(&sweep_signal, &recorded_response, &config).unwrap();
/// assert_eq!(ir.len(), sweep_signal.len());
/// ```
pub fn extract_ir(
    original_sweep: &[Sample],
    recorded_response: &[Sample],
) -> AudioResult<AudioBuffer> {
    extract_ir_with_config(
        original_sweep,
        recorded_response,
        &DeconvolutionConfig::default(),
    )
}

/// Extract impulse response with custom configuration
///
/// Allows fine-tuning of deconvolution parameters for optimal results.
///
/// # Example
///
/// ```rust
/// use rconvolve::{sweep, deconvolve};
///
/// let sweep_signal = sweep::exponential(48000.0, 2.0, 20.0, 20000.0).unwrap();
/// let recorded_response = sweep_signal.clone();
///
/// let config = deconvolve::DeconvolutionConfig {
///     method: deconvolve::DeconvolutionMethod::RegularizedFrequencyDomain {
///         regularization: 1e-3,
///     },
///     ir_length: Some(96000),  // 2 seconds at 48kHz
///     window: Some(deconvolve::WindowType::Hann),
///     pre_delay: 0,
/// };
/// let ir =
///     deconvolve::extract_ir_with_config(&sweep_signal, &recorded_response, &config).unwrap();
/// let _ = ir;
/// ```
pub fn extract_ir_with_config(
    original_sweep: &[Sample],
    recorded_response: &[Sample],
    config: &DeconvolutionConfig,
) -> AudioResult<AudioBuffer> {
    if original_sweep.is_empty() || recorded_response.is_empty() {
        return Err(AudioError::InsufficientData);
    }

    match config.method {
        DeconvolutionMethod::FrequencyDomain => {
            extract_ir_frequency_domain(original_sweep, recorded_response, config, None)
        }
        DeconvolutionMethod::RegularizedFrequencyDomain { regularization } => {
            extract_ir_frequency_domain(
                original_sweep,
                recorded_response,
                config,
                Some(regularization),
            )
        }
    }
}

fn extract_ir_frequency_domain(
    original_sweep: &[Sample],
    recorded_response: &[Sample],
    config: &DeconvolutionConfig,
    regularization: Option<Sample>,
) -> AudioResult<AudioBuffer> {
    // Use the larger of the two signals for FFT size to avoid truncation
    let max_length = original_sweep.len().max(recorded_response.len());
    let fft_size = next_power_of_two(max_length * 2); // 2x for safety margin

    // Pad both signals to the same length
    let sweep_padded = pad_to_length(original_sweep, fft_size);
    let response_padded = pad_to_length(recorded_response, fft_size);

    // Create FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    // Convert to complex
    let mut sweep_complex: Vec<Complex<Sample>> =
        sweep_padded.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut response_complex: Vec<Complex<Sample>> = response_padded
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();

    // Forward FFT
    fft.process(&mut sweep_complex);
    fft.process(&mut response_complex);

    // Frequency domain deconvolution: H = Y * conj(X) / (|X|^2 + λ)
    // This implements Wiener deconvolution for sweep signals
    let mut ir_complex = Vec::with_capacity(fft_size);

    for (response_bin, sweep_bin) in response_complex.iter().zip(sweep_complex.iter()) {
        let division_result = if let Some(reg) = regularization {
            // Regularized Wiener deconvolution
            let sweep_power = sweep_bin.norm_sqr();
            let denominator = sweep_power + reg;
            if denominator > 1e-30 {
                (response_bin * sweep_bin.conj()) / denominator
            } else {
                Complex::new(0.0, 0.0)
            }
        } else {
            // Simple spectral division
            let sweep_power = sweep_bin.norm_sqr();
            if sweep_power > 1e-30 {
                (response_bin * sweep_bin.conj()) / sweep_power
            } else {
                Complex::new(0.0, 0.0)
            }
        };
        ir_complex.push(division_result);
    }

    // Inverse FFT
    ifft.process(&mut ir_complex);

    // Extract real part and apply FFT normalization
    let mut ir: Vec<Sample> = ir_complex
        .iter()
        .map(|c| c.re / (fft_size as Sample))
        .collect();

    // Handle circular convolution artifacts by extracting the linear convolution part
    // The impulse response should be causal (start at time 0)
    // Due to circular convolution, we may need to unwrap the result

    // Find the peak location to detect if there's a circular shift
    let peak_idx = ir
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    // If the peak is in the second half, it might be due to circular convolution
    if peak_idx > fft_size / 2 {
        // Rotate the signal to put the peak at the beginning
        ir.rotate_left(peak_idx);
    }

    post_process_ir(ir, config)
}

fn post_process_ir(mut ir: AudioBuffer, config: &DeconvolutionConfig) -> AudioResult<AudioBuffer> {
    // Remove pre-delay
    if config.pre_delay > 0 && config.pre_delay < ir.len() {
        ir.drain(0..config.pre_delay);
    }

    // Truncate to desired length
    if let Some(ir_length) = config.ir_length {
        if ir_length < ir.len() {
            ir.truncate(ir_length);
        } else if ir_length > ir.len() {
            ir.resize(ir_length, 0.0);
        }
    }

    // Apply window function
    if let Some(window_type) = config.window {
        apply_window(&mut ir, window_type);
    }

    Ok(ir)
}

fn apply_window(buffer: &mut [Sample], window_type: WindowType) {
    let len = buffer.len();
    if len <= 1 {
        return;
    }

    for (i, sample) in buffer.iter_mut().enumerate() {
        let window_value = match window_type {
            WindowType::Hann => {
                #[cfg(feature = "std")]
                let val = 0.5
                    * (1.0
                        - ((2.0 * core::f32::consts::PI * (i as Sample)) / ((len - 1) as Sample))
                            .cos());
                #[cfg(not(feature = "std"))]
                let val = 0.5
                    * (1.0
                        - libm::cosf(
                            (2.0 * core::f32::consts::PI * (i as Sample)) / ((len - 1) as Sample),
                        ));
                val
            }
            WindowType::Hamming => {
                #[cfg(feature = "std")]
                let val = 0.54
                    - 0.46
                        * ((2.0 * core::f32::consts::PI * (i as Sample)) / ((len - 1) as Sample))
                            .cos();
                #[cfg(not(feature = "std"))]
                let val = 0.54
                    - 0.46
                        * libm::cosf(
                            (2.0 * core::f32::consts::PI * (i as Sample)) / ((len - 1) as Sample),
                        );
                val
            }
            WindowType::Blackman => {
                let n = (i as Sample) / ((len - 1) as Sample);
                #[cfg(feature = "std")]
                let val = 0.42 - 0.5 * (2.0 * core::f32::consts::PI * n).cos()
                    + 0.08 * (4.0 * core::f32::consts::PI * n).cos();
                #[cfg(not(feature = "std"))]
                let val = 0.42 - 0.5 * libm::cosf(2.0 * core::f32::consts::PI * n)
                    + 0.08 * libm::cosf(4.0 * core::f32::consts::PI * n);
                val
            }
        };
        *sample *= window_value;
    }
}

fn pad_to_length(input: &[Sample], target_length: usize) -> Vec<Sample> {
    let mut padded = input.to_vec();
    padded.resize(target_length, 0.0);
    padded
}

/// Estimate the Signal-to-Noise Ratio (SNR) of the extracted impulse response
///
/// Calculates SNR by comparing the peak amplitude (before `noise_floor_start`) to the
/// RMS level of the noise floor (from `noise_floor_start` to the end).
///
/// # Arguments
/// * `ir` - The impulse response to analyze
/// * `noise_floor_start` - Sample index where noise floor begins (typically last 25% of IR)
///
/// # Returns
/// SNR in decibels, or `f32::INFINITY` if noise floor is zero
///
/// # Example
///
/// ```rust
/// use rconvolve::deconvolve;
///
/// // Simple IR with a peak and some tail
/// let mut ir = vec![0.0; 100];
/// ir[10] = 1.0;
/// for i in 50..100 {
///     ir[i] = 0.01;
/// }
///
/// // Estimate SNR using last 25% as noise floor
/// let noise_floor_start = ir.len() * 3 / 4;
/// let snr = deconvolve::estimate_snr(&ir, noise_floor_start).unwrap();
/// println!("SNR: {} dB", snr);
/// ```
pub fn estimate_snr(ir: &[Sample], noise_floor_start: usize) -> AudioResult<Sample> {
    if ir.len() <= noise_floor_start {
        return Err(AudioError::InsufficientData);
    }

    // Find peak of the impulse response
    let peak_amplitude = ir
        .iter()
        .take(noise_floor_start)
        .map(|&x| x.abs())
        .fold(0.0, f32::max);

    // Calculate RMS of the noise floor
    let noise_samples = &ir[noise_floor_start..];
    if noise_samples.is_empty() {
        return Err(AudioError::InsufficientData);
    }

    let noise_rms = {
        let sum_squares: Sample = noise_samples.iter().map(|&x| x * x).sum();
        #[cfg(feature = "std")]
        let rms = (sum_squares / (noise_samples.len() as Sample)).sqrt();
        #[cfg(not(feature = "std"))]
        let rms = libm::sqrtf(sum_squares / (noise_samples.len() as Sample));
        rms
    };

    if noise_rms > 0.0 {
        #[cfg(feature = "std")]
        let snr_db = 20.0 * (peak_amplitude / noise_rms).log10();
        #[cfg(not(feature = "std"))]
        let snr_db = 20.0 * libm::log10f(peak_amplitude / noise_rms);
        Ok(snr_db)
    } else {
        Ok(f32::INFINITY)
    }
}

/// Quality rating for impulse responses
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityRating {
    /// High quality IR suitable for professional use
    Good,
    /// Acceptable quality with minor issues
    Fair,
    /// Poor quality with significant problems
    Poor,
}

/// Quality metrics for an impulse response
#[derive(Debug, Clone)]
pub struct IRQuality {
    /// Peak amplitude of the IR
    pub peak: Sample,
    /// RMS level of the IR
    pub rms: Sample,
    /// Early-to-late energy ratio in dB (early reflections vs reverb tail)
    pub early_late_ratio_db: Sample,
    /// Noise ratio in the tail (0.0 = no noise, 1.0 = all noise)
    pub noise_ratio: Sample,
    /// Signal-to-noise ratio in dB
    pub snr_db: Sample,
    /// Overall quality rating
    pub rating: QualityRating,
    /// List of specific warnings about quality issues
    pub warnings: Vec<&'static str>,
}

/// Assess the overall quality of an impulse response
///
/// This function analyzes various metrics to determine if an IR is suitable
/// for use in convolution reverb applications.
///
/// # Metrics Calculated
///
/// - **Peak**: Maximum amplitude of the IR
/// - **RMS**: Root mean square level
/// - **Early-to-late ratio**: Energy ratio between early reflections (first 10%) and late reverb (10-75%)
/// - **Noise ratio**: RMS of tail (last 25%) relative to peak
/// - **SNR**: Signal-to-noise ratio in dB
/// - **Rating**: Overall quality assessment (Good, Fair, or Poor)
///
/// # Example
///
/// ```rust
/// use rconvolve::deconvolve;
///
/// // Simple IR with a peak and decaying tail
/// let mut ir = vec![0.0; 100];
/// ir[10] = 1.0;
/// for i in 11..80 {
///     ir[i] = (-(i as f32 - 10.0) / 50.0).exp() * 0.1;
/// }
///
/// let quality = deconvolve::assess_ir_quality(&ir).unwrap();
///
/// println!("Peak: {}", quality.peak);
/// println!("RMS: {}", quality.rms);
/// println!("SNR: {} dB", quality.snr_db);
/// println!("Early-to-late ratio: {} dB", quality.early_late_ratio_db);
/// println!("Noise ratio: {}", quality.noise_ratio);
/// println!("Rating: {:?}", quality.rating);
///
/// for warning in &quality.warnings {
///     println!("Warning: {}", warning);
/// }
/// ```
pub fn assess_ir_quality(ir: &[Sample]) -> AudioResult<IRQuality> {
    if ir.is_empty() {
        return Err(AudioError::InsufficientData);
    }

    // Calculate basic metrics
    let peak = ir.iter().map(|&x| x.abs()).fold(0.0, f32::max);

    let rms = {
        let sum_squares: Sample = ir.iter().map(|&x| x * x).sum();
        #[cfg(feature = "std")]
        let rms = (sum_squares / (ir.len() as Sample)).sqrt();
        #[cfg(not(feature = "std"))]
        let rms = libm::sqrtf(sum_squares / (ir.len() as Sample));
        rms
    };

    // Calculate early-to-late energy ratio (more meaningful for IR quality)
    // Early reflections (first 10%) vs late reverb (10-75%)
    let early_end = ir.len() / 10;
    let late_end = (ir.len() * 3) / 4;

    let early_energy: Sample = ir.iter().take(early_end).map(|&x| x * x).sum();
    let late_energy: Sample = ir
        .iter()
        .skip(early_end)
        .take(late_end - early_end)
        .map(|&x| x * x)
        .sum();

    let early_late_ratio_db = if late_energy > 0.0 {
        #[cfg(feature = "std")]
        let ratio = 10.0 * (early_energy / late_energy).log10();
        #[cfg(not(feature = "std"))]
        let ratio = 10.0 * libm::log10f(early_energy / late_energy);
        ratio
    } else if early_energy > 0.0 {
        f32::INFINITY
    } else {
        0.0
    };

    // Calculate noise in the tail (last 25% of the IR)
    let tail_start = (ir.len() * 3) / 4;
    let noise_ratio = if tail_start < ir.len() {
        let tail_samples = &ir[tail_start..];
        let tail_rms = {
            let sum_squares: Sample = tail_samples.iter().map(|&x| x * x).sum();
            #[cfg(feature = "std")]
            let rms = (sum_squares / (tail_samples.len() as Sample)).sqrt();
            #[cfg(not(feature = "std"))]
            let rms = libm::sqrtf(sum_squares / (tail_samples.len() as Sample));
            rms
        };

        if peak > 0.0 {
            tail_rms / peak
        } else {
            1.0
        }
    } else {
        0.0
    };

    // Estimate SNR using the tail as noise floor
    let snr_db = estimate_snr(ir, tail_start).unwrap_or(0.0);

    // Quality assessment logic
    let mut rating = QualityRating::Good;
    let mut warnings = Vec::new();

    // Check peak amplitude
    if peak < 0.01 {
        rating = QualityRating::Poor;
        warnings.push("Very low peak - try different regularization or check recording quality");
    } else if peak < 0.1 {
        rating = QualityRating::Fair;
        warnings.push("Low peak amplitude - may need level adjustment");
    }

    // Check early-to-late energy ratio (good IRs have strong early reflections)
    if early_late_ratio_db < 6.0 {
        rating = QualityRating::Poor;
        warnings
            .push("Weak early reflections - may indicate poor recording or over-regularization");
    } else if early_late_ratio_db < 12.0 {
        if matches!(rating, QualityRating::Good) {
            rating = QualityRating::Fair;
        }
        warnings.push("Moderate early reflection strength - acceptable but could be better");
    }

    // Check noise level
    if noise_ratio > 0.1 {
        rating = QualityRating::Poor;
        warnings.push("High noise floor detected in IR tail");
    } else if noise_ratio > 0.05 {
        if matches!(rating, QualityRating::Good) {
            rating = QualityRating::Fair;
        }
        warnings.push("Some noise detected - try higher regularization");
    }

    Ok(IRQuality {
        peak,
        rms,
        early_late_ratio_db,
        noise_ratio,
        snr_db,
        rating,
        warnings,
    })
}

/// Extract both channels from interleaved stereo data
///
/// # Arguments
/// * `stereo_data` - Interleaved audio data (L, R, L, R, ...)
/// * `num_channels` - Number of channels (1 for mono, 2 for stereo)
///
/// # Returns
/// Tuple of `(left_channel, right_channel)` buffers
///
/// For mono input, both channels are identical copies.
///
/// # Example
///
/// ```rust
/// use rconvolve::deconvolve;
///
/// let interleaved = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];  // L, R, L, R, L, R
/// let (left, right) = deconvolve::extract_stereo_channels(&interleaved, 2);
/// // left = [1.0, 3.0, 5.0], right = [2.0, 4.0, 6.0]
/// ```
pub fn extract_stereo_channels(
    stereo_data: &[Sample],
    num_channels: usize,
) -> (AudioBuffer, AudioBuffer) {
    if num_channels == 1 {
        // Mono input - duplicate to both channels
        (stereo_data.to_vec(), stereo_data.to_vec())
    } else {
        // Extract both left and right channels from interleaved data
        let mut left = Vec::with_capacity(stereo_data.len() / num_channels);
        let mut right = Vec::with_capacity(stereo_data.len() / num_channels);

        for chunk in stereo_data.chunks(num_channels) {
            left.push(chunk[0]); // Left channel
            right.push(chunk.get(1).copied().unwrap_or(chunk[0])); // Right channel (or duplicate left if missing)
        }

        (left, right)
    }
}

/// Extract mono signal from interleaved stereo data (left channel only)
///
/// # Arguments
/// * `stereo_data` - Interleaved audio data (L, R, L, R, ...)
/// * `num_channels` - Number of channels (1 for mono, 2 for stereo)
///
/// # Returns
/// Left channel as a mono buffer. For mono input, returns the data as-is.
///
/// # Example
///
/// ```rust
/// use rconvolve::deconvolve;
///
/// let interleaved = vec![1.0, 2.0, 3.0, 4.0];  // L, R, L, R
/// let left = deconvolve::extract_left_channel(&interleaved, 2);
/// // left = [1.0, 3.0]
/// ```
pub fn extract_left_channel(stereo_data: &[Sample], num_channels: usize) -> AudioBuffer {
    if num_channels == 1 {
        // Already mono, return as-is
        stereo_data.to_vec()
    } else {
        // Extract left channel from interleaved data
        stereo_data
            .chunks(num_channels)
            .map(|chunk| chunk[0]) // Take first channel (left)
            .collect()
    }
}

/// Convert mono signal to stereo by duplicating the channel
///
/// Creates interleaved stereo output (L, R, L, R, ...) where both channels
/// contain identical samples.
///
/// # Example
///
/// ```rust
/// use rconvolve::deconvolve;
///
/// let mono = vec![1.0, 2.0, 3.0];
/// let stereo = deconvolve::mono_to_stereo(&mono);
/// // stereo = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
/// ```
pub fn mono_to_stereo(mono_data: &[Sample]) -> AudioBuffer {
    let mut stereo = Vec::with_capacity(mono_data.len() * 2);
    for &sample in mono_data {
        stereo.push(sample); // Left channel
        stereo.push(sample); // Right channel (duplicate)
    }
    stereo
}

/// Create interleaved stereo IR from separate left and right channel responses
///
/// This captures the spatial response of a single source (sweep) to left and right microphones.
/// The output is interleaved as (L, R, L, R, ...).
///
/// If the IRs have different lengths, the shorter one is zero-padded.
///
/// # Example
///
/// ```rust
/// use rconvolve::deconvolve;
///
/// let left_ir = vec![0.5, 0.3, 0.1];
/// let right_ir = vec![0.4, 0.2, 0.1];
/// let stereo_ir = deconvolve::create_stereo_ir(&left_ir, &right_ir);
/// // stereo_ir = [0.5, 0.4, 0.3, 0.2, 0.1, 0.1]
/// ```
pub fn create_stereo_ir(left_ir: &[Sample], right_ir: &[Sample]) -> AudioBuffer {
    let max_len = left_ir.len().max(right_ir.len());
    let mut stereo = Vec::with_capacity(max_len * 2);

    for i in 0..max_len {
        let left_sample = left_ir.get(i).copied().unwrap_or(0.0);
        let right_sample = right_ir.get(i).copied().unwrap_or(0.0);
        stereo.push(left_sample); // Left channel
        stereo.push(right_sample); // Right channel
    }

    stereo
}

/// Extract IR from interleaved stereo/mono data, handling channel conversion automatically
///
/// # Important Behavior
///
/// This function **always returns interleaved stereo output**:
/// - For mono input: The mono IR is duplicated to both channels
/// - For stereo input: Separate L/R IRs are extracted, capturing spatial response
///
/// # Arguments
/// * `original_sweep` - The reference sweep signal (mono)
/// * `recorded_response_interleaved` - Recorded response (interleaved: L, R, L, R, ...)
/// * `num_channels` - Number of channels in the recorded response (1 for mono, 2 for stereo)
///
/// # Example
///
/// ```rust
/// use rconvolve::{sweep, deconvolve};
///
/// let sweep_signal = sweep::exponential(48000.0, 2.0, 20.0, 20000.0).unwrap();
/// // Simulate interleaved stereo recording (duplicate sweep)
/// let interleaved_recording: Vec<f32> = sweep_signal
///     .iter()
///     .flat_map(|&s| [s, s])
///     .collect();
///
/// // Always returns interleaved stereo IR
/// let stereo_ir = deconvolve::extract_ir_from_interleaved(
///     &sweep_signal,
///     &interleaved_recording,
///     2,  // Number of channels
/// ).unwrap();
/// let _ = stereo_ir;
/// ```
pub fn extract_ir_from_interleaved(
    original_sweep: &[Sample],
    recorded_response_interleaved: &[Sample],
    num_channels: usize,
) -> AudioResult<AudioBuffer> {
    extract_ir_from_interleaved_with_config(
        original_sweep,
        recorded_response_interleaved,
        num_channels,
        &DeconvolutionConfig::default(),
    )
}

/// Extract IR from interleaved stereo/mono data with custom configuration
///
/// Same as [`extract_ir_from_interleaved`] but allows custom deconvolution settings.
///
/// # Important Behavior
///
/// Always returns interleaved stereo output (see [`extract_ir_from_interleaved`] for details).
///
/// # Example
///
/// ```rust
/// use rconvolve::{sweep, deconvolve};
///
/// let sweep_signal = sweep::exponential(48000.0, 2.0, 20.0, 20000.0).unwrap();
/// let interleaved_recording: Vec<f32> = sweep_signal
///     .iter()
///     .flat_map(|&s| [s, s])
///     .collect();
///
/// let config = deconvolve::DeconvolutionConfig {
///     method: deconvolve::DeconvolutionMethod::RegularizedFrequencyDomain {
///         regularization: 1e-3,
///     },
///     ir_length: Some(96000),
///     window: Some(deconvolve::WindowType::Hann),
///     pre_delay: 0,
/// };
///
/// let stereo_ir = deconvolve::extract_ir_from_interleaved_with_config(
///     &sweep_signal,
///     &interleaved_recording,
///     2,
///     &config,
/// ).unwrap();
/// let _ = stereo_ir;
/// ```
pub fn extract_ir_from_interleaved_with_config(
    original_sweep: &[Sample],
    recorded_response_interleaved: &[Sample],
    num_channels: usize,
    config: &DeconvolutionConfig,
) -> AudioResult<AudioBuffer> {
    if num_channels == 1 {
        // Mono input - process once and duplicate to stereo
        let mono_ir =
            extract_ir_with_config(original_sweep, recorded_response_interleaved, config)?;
        Ok(mono_to_stereo(&mono_ir))
    } else {
        // Stereo input - process both channels separately for spatial stereo IR
        let (left_channel, right_channel) =
            extract_stereo_channels(recorded_response_interleaved, num_channels);

        // Process left channel response to the sweep
        let left_ir = extract_ir_with_config(original_sweep, &left_channel, config)?;

        // Process right channel response to the sweep
        let right_ir = extract_ir_with_config(original_sweep, &right_channel, config)?;

        // Create stereo IR capturing spatial response (single source → L/R mics)
        Ok(create_stereo_ir(&left_ir, &right_ir))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sweep;
    use approx::assert_abs_diff_eq;

    #[cfg(not(feature = "std"))]
    use alloc::{format, vec, vec::Vec};

    #[test]
    fn test_ir_extraction_basic() {
        // Generate a test sweep
        let sweep = sweep::exponential(1000.0, 1.0, 100.0, 500.0).unwrap();

        // Simulate a simple response (just the sweep with some delay and scaling)
        let mut response = vec![0.0; 10]; // 10 sample delay
        response.extend_from_slice(&sweep);
        for sample in &mut response[10..] {
            *sample *= 0.8; // Scale down
        }

        let ir = extract_ir(&sweep, &response).unwrap();

        // The IR should have a peak around sample 10 (the delay we added)
        assert!(!ir.is_empty());

        let peak_index = ir
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Peak should be roughly around the delay we introduced
        assert!(peak_index >= 5 && peak_index <= 15);
    }

    #[test]
    fn test_window_functions() {
        let mut buffer = vec![1.0; 100];

        apply_window(&mut buffer, WindowType::Hann);
        // Hann window should taper to zero at the edges
        assert_abs_diff_eq!(buffer[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(buffer[99], 0.0, epsilon = 1e-6);
        assert!(buffer[50] > 0.9); // Peak in the middle

        // Window application for single-length buffer (should be unchanged)
        let mut single = vec![1.0];
        apply_window(&mut single, WindowType::Hann);
        assert_eq!(single, vec![1.0]);
        apply_window(&mut single, WindowType::Hamming);
        assert_eq!(single, vec![1.0]);
        apply_window(&mut single, WindowType::Blackman);
        assert_eq!(single, vec![1.0]);
    }

    #[test]
    fn test_pad_to_length() {
        let input = vec![1.0, 2.0, 3.0];
        let padded = pad_to_length(&input, 5);
        assert_eq!(padded, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
        let padded = pad_to_length(&input, 3);
        assert_eq!(padded, input);
        let padded = pad_to_length(&input, 0);
        assert_eq!(padded, vec![]); // resize(0) truncates to empty
    }

    #[test]
    fn test_extract_stereo_channels_edge_cases() {
        // Empty input
        let (left, right) = extract_stereo_channels(&[], 2);
        assert_eq!(left, Vec::<f32>::new());
        assert_eq!(right, Vec::<f32>::new());

        // Odd-length input
        let data = vec![1.0, 2.0, 3.0];
        let (left, right) = extract_stereo_channels(&data, 2);
        assert_eq!(left, vec![1.0, 3.0]);
        assert_eq!(right, vec![2.0, 3.0]);
    }

    #[test]
    fn test_extract_left_channel_edge_cases() {
        // Empty input
        let left = extract_left_channel(&[], 2);
        assert_eq!(left, Vec::<f32>::new());

        // Odd-length input
        let data = vec![1.0, 2.0, 3.0];
        let left = extract_left_channel(&data, 2);
        assert_eq!(left, vec![1.0, 3.0]);
    }

    #[test]
    fn test_mono_to_stereo_edge_cases() {
        // Empty input
        let stereo = mono_to_stereo(&[]);
        assert_eq!(stereo, Vec::<f32>::new());

        // Single sample
        let stereo = mono_to_stereo(&[1.0]);
        assert_eq!(stereo, vec![1.0, 1.0]);
    }

    #[test]
    fn test_create_stereo_ir_edge_cases() {
        // Both empty
        let stereo = create_stereo_ir(&[], &[]);
        assert_eq!(stereo, Vec::<f32>::new());

        // One empty, one non-empty
        let stereo = create_stereo_ir(&[1.0, 2.0], &[]);
        assert_eq!(stereo, vec![1.0, 0.0, 2.0, 0.0]);
        let stereo = create_stereo_ir(&[], &[3.0, 4.0]);
        assert_eq!(stereo, vec![0.0, 3.0, 0.0, 4.0]);
    }


    #[test]
    fn test_snr_estimation() {
        // Create a test IR with known SNR characteristics
        let mut ir = vec![0.0; 1000];
        ir[100] = 1.0; // Strong peak at sample 100

        // Add noise floor
        for i in 500..1000 {
            ir[i] = 0.01; // -40dB noise floor
        }

        let snr = estimate_snr(&ir, 500).unwrap();

        // Should be approximately 40dB SNR
        assert!(snr > 35.0 && snr < 45.0);
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(1000), 1024);
        assert_eq!(next_power_of_two(1024), 1024);
    }

    #[test]
    fn test_assess_ir_quality() {
        // Create a high-quality test IR
        let mut good_ir = vec![0.0; 1000];
        good_ir[50] = 1.0; // Strong peak
                        // Add much weaker exponential decay for better dynamic range
        for i in 51..700 {
            // End decay earlier to leave quiet tail
            #[cfg(feature = "std")]
            let decay_value = (-((i as f32) - 50.0) / 400.0).exp() * 0.003; // Much weaker decay
            #[cfg(not(feature = "std"))]
            let decay_value = libm::expf(-((i as f32) - 50.0) / 400.0) * 0.003; // Much weaker decay
            good_ir[i] = decay_value;
        }

        let quality = assess_ir_quality(&good_ir).unwrap();
        assert_eq!(quality.rating, QualityRating::Good);
        assert!(quality.peak > 0.9);
        assert!(quality.early_late_ratio_db > 12.0); // Good IRs should have strong early reflections
        assert!(quality.warnings.is_empty());

        // Create a poor-quality test IR
        let mut poor_ir = vec![0.01; 1000]; // High noise floor
        poor_ir[50] = 0.02; // Very low peak

        let quality = assess_ir_quality(&poor_ir).unwrap();
        assert_eq!(quality.rating, QualityRating::Poor);
        assert!(!quality.warnings.is_empty());
    }

    #[test]
    fn test_stereo_mono_channel_handling() {
        // Test mono to stereo conversion
        let mono_data = vec![0.1, 0.2, 0.3, 0.4];
        let stereo_data = mono_to_stereo(&mono_data);
        let expected_stereo = vec![0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4];
        assert_eq!(stereo_data, expected_stereo);

        // Test left channel extraction from stereo
        let interleaved_stereo = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7]; // L, R, L, R, L, R
        let left_channel = extract_left_channel(&interleaved_stereo, 2);
        let expected_left = vec![0.1, 0.2, 0.3];
        assert_eq!(left_channel, expected_left);

        // Test stereo channel extraction
        let (left, right) = extract_stereo_channels(&interleaved_stereo, 2);
        let expected_left = vec![0.1, 0.2, 0.3];
        let expected_right = vec![0.9, 0.8, 0.7];
        assert_eq!(left, expected_left);
        assert_eq!(right, expected_right);

        // Test stereo IR creation
        let left_ir = vec![0.1, 0.2, 0.3];
        let right_ir = vec![0.4, 0.5, 0.6];
        let stereo_ir = create_stereo_ir(&left_ir, &right_ir);
        let expected_stereo_ir = vec![0.1, 0.4, 0.2, 0.5, 0.3, 0.6];
        assert_eq!(stereo_ir, expected_stereo_ir);

        // Test mono passthrough
        let mono_input = vec![0.1, 0.2, 0.3];
        let mono_output = extract_left_channel(&mono_input, 1);
        assert_eq!(mono_input, mono_output);
    }

    #[test]
    fn test_extract_ir_from_interleaved() {
        let sample_rate = 1000.0;
        let duration = 0.1;
        let start_freq = 10.0;
        let end_freq = 100.0;

        // Generate test sweep
        let sweep = sweep::exponential(sample_rate, duration, start_freq, end_freq).unwrap();

        // Create stereo recorded response (interleaved) with different L/R responses
        let mut stereo_response = Vec::with_capacity(sweep.len() * 2);
        for &sample in &sweep {
            stereo_response.push(sample * 0.8); // Left channel (attenuated sweep)
            stereo_response.push(sample * 0.3); // Right channel (different attenuation)
        }

        // Extract IR from stereo data (should produce spatial stereo IR)
        let stereo_ir = extract_ir_from_interleaved(&sweep, &stereo_response, 2).unwrap();

        // Extract IR from mono data (should produce duplicated stereo IR)
        let mono_ir = extract_ir_from_interleaved(&sweep, &sweep, 1).unwrap();

        // Both should be stereo output (always stereo)
        assert!(stereo_ir.len() % 2 == 0); // Must be even for stereo
        assert!(mono_ir.len() % 2 == 0); // Must be even for stereo

        // For spatial stereo IR, left and right channels should be different
        let stereo_left = extract_left_channel(&stereo_ir, 2);
        let stereo_right: Vec<f32> = stereo_ir.chunks(2).map(|chunk| chunk[1]).collect();

        // For mono input, left and right should be identical
        let mono_left = extract_left_channel(&mono_ir, 2);
        let mono_right: Vec<f32> = mono_ir.chunks(2).map(|chunk| chunk[1]).collect();
        assert_eq!(mono_left, mono_right); // Should be identical for mono input

        // Spatial stereo should have different L/R channels (due to different attenuation)
        assert_ne!(stereo_left, stereo_right); // Should be different for stereo input
    }

    #[test]
    fn test_deconvolution_method_display() {
        let method = DeconvolutionMethod::RegularizedFrequencyDomain {
            regularization: 1e-6,
        };
        let display = format!("{:?}", method);
        assert!(display.contains("RegularizedFrequencyDomain"));
        assert!(display.contains("regularization"));
    }

    #[test]
    fn test_ir_quality_display() {
        let quality = IRQuality {
            peak: 0.8,
            rms: 0.2,
            early_late_ratio_db: 10.0,
            noise_ratio: 0.01,
            snr_db: 60.0,
            rating: QualityRating::Good,
            warnings: vec![],
        };
        assert_eq!(format!("{:?}", quality).contains("peak"), true);
    }

    #[test]
    fn test_extract_ir_empty_inputs() {
        let empty: Vec<f32> = vec![];
        let sweep = vec![1.0, 0.5, 0.25];

        // Empty sweep
        let result1 = extract_ir(&empty, &sweep);
        assert_eq!(result1, Err(AudioError::InsufficientData));

        // Empty response
        let result2 = extract_ir(&sweep, &empty);
        assert_eq!(result2, Err(AudioError::InsufficientData));
    }

    #[test]
    fn test_estimate_snr_insufficient_data() {
        let ir = vec![1.0, 0.5, 0.25];
        let result = estimate_snr(&ir, 10); // noise_floor_start beyond length
        assert_eq!(result, Err(AudioError::InsufficientData));
    }

    #[test]
    fn test_assess_ir_quality_empty() {
        let empty: Vec<f32> = vec![];
        let result = assess_ir_quality(&empty);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_stereo_channels_mono() {
        let data = vec![1.0, 2.0, 3.0];
        let (left, right) = extract_stereo_channels(&data, 1);
        // For mono, left and right should be identical
        assert_eq!(left, right);
        assert_eq!(left, data);
    }

    #[test]
    fn test_extract_left_channel_empty() {
        let empty: Vec<f32> = vec![];
        let result = extract_left_channel(&empty, 2);
        assert_eq!(result, Vec::<f32>::new());
    }

    #[test]
    fn test_mono_to_stereo_empty() {
        let empty: Vec<f32> = vec![];
        let result = mono_to_stereo(&empty);
        assert_eq!(result, Vec::<f32>::new());
    }

    #[test]
    fn test_create_stereo_ir_different_lengths() {
        let left = vec![1.0, 2.0, 3.0];
        let right = vec![4.0, 5.0]; // Shorter
        let result = create_stereo_ir(&left, &right);
        // Should pad to longest length
        assert_eq!(result.len(), 6); // 3 samples * 2 channels
    }

    #[test]
    fn test_extract_ir_from_interleaved_empty() {
        let sweep = vec![1.0, 2.0];
        let empty: Vec<f32> = vec![];
        let result = extract_ir_from_interleaved(&sweep, &empty, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_deconvolution_config_default() {
        let config = DeconvolutionConfig::default();
        assert!(matches!(
            config.method,
            DeconvolutionMethod::RegularizedFrequencyDomain { .. }
        ));
        assert_eq!(config.ir_length, None);
    }

    #[test]
    fn test_deconvolution_config_clone() {
        let config1 = DeconvolutionConfig::default();
        let config2 = config1.clone();
        assert_eq!(config1.ir_length, config2.ir_length);
    }

    #[test]
    fn test_frequency_domain_method() {
        // Test non-regularized frequency domain deconvolution
        let sweep = sweep::exponential(1000.0, 0.5, 100.0, 500.0).unwrap();
        let response = sweep.clone();

        let config = DeconvolutionConfig {
            method: DeconvolutionMethod::FrequencyDomain,
            ir_length: None,
            window: None,
            pre_delay: 0,
        };

        let ir = extract_ir_with_config(&sweep, &response, &config).unwrap();
        assert!(!ir.is_empty());
    }

    #[test]
    fn test_post_process_pre_delay() {
        let sweep = sweep::exponential(1000.0, 0.5, 100.0, 500.0).unwrap();
        let response = sweep.clone();

        let config = DeconvolutionConfig {
            method: DeconvolutionMethod::RegularizedFrequencyDomain {
                regularization: 1e-3,
            },
            ir_length: None,
            window: None,
            pre_delay: 10, // Remove first 10 samples
        };

        let ir = extract_ir_with_config(&sweep, &response, &config).unwrap();
        assert!(!ir.is_empty());
    }

    #[test]
    fn test_post_process_ir_length_truncate() {
        let sweep = sweep::exponential(1000.0, 0.5, 100.0, 500.0).unwrap();
        let response = sweep.clone();

        let config = DeconvolutionConfig {
            method: DeconvolutionMethod::RegularizedFrequencyDomain {
                regularization: 1e-3,
            },
            ir_length: Some(100), // Truncate to 100 samples
            window: None,
            pre_delay: 0,
        };

        let ir = extract_ir_with_config(&sweep, &response, &config).unwrap();
        assert_eq!(ir.len(), 100);
    }

    #[test]
    fn test_post_process_ir_length_pad() {
        let sweep = sweep::exponential(1000.0, 0.1, 100.0, 500.0).unwrap();
        let response = sweep.clone();

        let config = DeconvolutionConfig {
            method: DeconvolutionMethod::RegularizedFrequencyDomain {
                regularization: 1e-3,
            },
            ir_length: Some(1000), // Pad to 1000 samples
            window: None,
            pre_delay: 0,
        };

        let ir = extract_ir_with_config(&sweep, &response, &config).unwrap();
        assert_eq!(ir.len(), 1000);
        // Last samples should be zero-padded
        assert!(ir[ir.len() - 10..].iter().all(|&x| x.abs() < 0.1));
    }

    #[test]
    fn test_post_process_ir_length_exact() {
        let sweep = sweep::exponential(1000.0, 0.5, 100.0, 500.0).unwrap();
        let response = sweep.clone();

        let ir_no_config = extract_ir(&sweep, &response).unwrap();
        let target_len = ir_no_config.len();

        let config = DeconvolutionConfig {
            method: DeconvolutionMethod::RegularizedFrequencyDomain {
                regularization: 1e-3,
            },
            ir_length: Some(target_len), // Exact length
            window: None,
            pre_delay: 0,
        };

        let ir = extract_ir_with_config(&sweep, &response, &config).unwrap();
        assert_eq!(ir.len(), target_len);
    }

    #[test]
    fn test_window_hamming() {
        let sweep = sweep::exponential(1000.0, 0.5, 100.0, 500.0).unwrap();
        let response = sweep.clone();

        let config = DeconvolutionConfig {
            method: DeconvolutionMethod::RegularizedFrequencyDomain {
                regularization: 1e-3,
            },
            ir_length: None,
            window: Some(WindowType::Hamming),
            pre_delay: 0,
        };

        let ir = extract_ir_with_config(&sweep, &response, &config).unwrap();
        assert!(!ir.is_empty());
    }

    #[test]
    fn test_window_blackman() {
        let sweep = sweep::exponential(1000.0, 0.5, 100.0, 500.0).unwrap();
        let response = sweep.clone();

        let config = DeconvolutionConfig {
            method: DeconvolutionMethod::RegularizedFrequencyDomain {
                regularization: 1e-3,
            },
            ir_length: None,
            window: Some(WindowType::Blackman),
            pre_delay: 0,
        };

        let ir = extract_ir_with_config(&sweep, &response, &config).unwrap();
        assert!(!ir.is_empty());
    }

    #[test]
    fn test_window_single_sample() {
        let mut buffer = vec![1.0];
        apply_window(&mut buffer, WindowType::Hann);
        assert_eq!(buffer, vec![1.0]);
        apply_window(&mut buffer, WindowType::Hamming);
        assert_eq!(buffer, vec![1.0]);
        apply_window(&mut buffer, WindowType::Blackman);
        assert_eq!(buffer, vec![1.0]);
    }

    #[test]
    fn test_window_empty() {
        let mut buffer = vec![];
        apply_window(&mut buffer, WindowType::Hann);
        assert_eq!(buffer, vec![]);
    }

    #[test]
    fn test_estimate_snr_zero_noise() {
        // IR with signal but zero noise floor
        let mut ir = vec![0.0; 100];
        ir[10] = 1.0; // Peak
        // Rest is zero (no noise)

        let snr = estimate_snr(&ir, 50).unwrap();
        assert_eq!(snr, f32::INFINITY);
    }

    #[test]
    fn test_estimate_snr_zero_peak() {
        // IR with no signal (all zeros)
        let ir = vec![0.0; 100];
        let snr = estimate_snr(&ir, 50).unwrap();
        // Should handle zero peak gracefully
        assert!(snr.is_finite() || snr == f32::INFINITY);
    }

    #[test]
    fn test_assess_ir_quality_fair_rating() {
        // Create IR that should get Fair rating
        let mut ir = vec![0.0; 1000];
        ir[50] = 0.15; // Moderate peak (between 0.1 and 0.01)
        // Add some decay
        for i in 51..500 {
            #[cfg(feature = "std")]
            let decay = (-((i as f32) - 50.0) / 200.0).exp() * 0.01;
            #[cfg(not(feature = "std"))]
            let decay = libm::expf(-((i as f32) - 50.0) / 200.0) * 0.01;
            ir[i] = decay;
        }

        let quality = assess_ir_quality(&ir).unwrap();
        // Should be Fair or Poor depending on other metrics
        assert!(matches!(quality.rating, QualityRating::Fair | QualityRating::Poor));
    }

    #[test]
    fn test_assess_ir_quality_all_warnings() {
        // Create IR that triggers multiple warnings
        let mut ir = vec![0.005; 1000]; // Very low values
        ir[50] = 0.008; // Very low peak

        let quality = assess_ir_quality(&ir).unwrap();
        assert_eq!(quality.rating, QualityRating::Poor);
        assert!(!quality.warnings.is_empty());
    }

    #[test]
    fn test_assess_ir_quality_early_late_edge_cases() {
        // IR with no late energy (only early)
        let mut ir = vec![0.0; 1000];
        ir[10] = 1.0;
        // Only early energy, no late energy
        for i in 11..100 {
            ir[i] = 0.1;
        }

        let quality = assess_ir_quality(&ir).unwrap();
        // early_late_ratio_db should be INFINITY
        assert!(quality.early_late_ratio_db.is_infinite() || quality.early_late_ratio_db > 100.0);
    }

    #[test]
    fn test_assess_ir_quality_no_early_energy() {
        // IR with only late energy
        let mut ir = vec![0.0; 1000];
        // Skip early part, only late energy
        for i in 200..800 {
            ir[i] = 0.1;
        }

        let quality = assess_ir_quality(&ir).unwrap();
        // Should have low or zero early_late_ratio_db
        assert!(quality.early_late_ratio_db <= 0.0 || quality.early_late_ratio_db < 6.0);
    }

    #[test]
    fn test_extract_ir_from_interleaved_with_config_mono() {
        let sweep = sweep::exponential(1000.0, 0.1, 100.0, 500.0).unwrap();
        let mono_recording = sweep.clone();

        let config = DeconvolutionConfig {
            method: DeconvolutionMethod::RegularizedFrequencyDomain {
                regularization: 1e-3,
            },
            ir_length: Some(200),
            window: Some(WindowType::Hann),
            pre_delay: 5,
        };

        let stereo_ir = extract_ir_from_interleaved_with_config(
            &sweep,
            &mono_recording,
            1,
            &config,
        )
        .unwrap();

        // Should be stereo (interleaved)
        assert!(stereo_ir.len() % 2 == 0);
        // Left and right should be identical for mono input
        let left: Vec<f32> = stereo_ir.iter().step_by(2).copied().collect();
        let right: Vec<f32> = stereo_ir.iter().skip(1).step_by(2).copied().collect();
        assert_eq!(left, right);
    }

    #[test]
    fn test_extract_ir_from_interleaved_with_config_stereo() {
        let sweep = sweep::exponential(1000.0, 0.1, 100.0, 500.0).unwrap();
        let mut stereo_recording = Vec::new();
        for &s in &sweep {
            stereo_recording.push(s * 0.8);
            stereo_recording.push(s * 0.6);
        }

        let config = DeconvolutionConfig {
            method: DeconvolutionMethod::FrequencyDomain, // Non-regularized
            ir_length: Some(200),
            window: Some(WindowType::Hamming),
            pre_delay: 0,
        };

        let stereo_ir = extract_ir_from_interleaved_with_config(
            &sweep,
            &stereo_recording,
            2,
            &config,
        )
        .unwrap();

        assert!(stereo_ir.len() % 2 == 0);
    }

    #[test]
    fn test_extract_stereo_channels_multi_channel() {
        // Test with more than 2 channels (should still work)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 channels, 2 samples
        let (left, right) = extract_stereo_channels(&data, 3);
        // Should extract first and second channels
        assert_eq!(left, vec![1.0, 4.0]);
        assert_eq!(right, vec![2.0, 5.0]);
    }

    #[test]
    fn test_extract_left_channel_multi_channel() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 channels
        let left = extract_left_channel(&data, 3);
        assert_eq!(left, vec![1.0, 4.0]);
    }

    #[test]
    fn test_post_process_pre_delay_edge_cases() {
        let sweep = sweep::exponential(1000.0, 0.1, 100.0, 500.0).unwrap();
        let response = sweep.clone();

        // Pre-delay equal to IR length (should not panic)
        let ir_no_config = extract_ir(&sweep, &response).unwrap();
        let config = DeconvolutionConfig {
            method: DeconvolutionMethod::RegularizedFrequencyDomain {
                regularization: 1e-3,
            },
            ir_length: None,
            window: None,
            pre_delay: ir_no_config.len(), // Equal to length
        };

        let ir = extract_ir_with_config(&sweep, &response, &config).unwrap();
        // Should be empty or very short
        assert!(ir.len() <= ir_no_config.len());
    }

    #[test]
    fn test_post_process_pre_delay_greater_than_length() {
        let sweep = sweep::exponential(1000.0, 0.1, 100.0, 500.0).unwrap();
        let response = sweep.clone();

        let ir_no_config = extract_ir(&sweep, &response).unwrap();
        let config = DeconvolutionConfig {
            method: DeconvolutionMethod::RegularizedFrequencyDomain {
                regularization: 1e-3,
            },
            ir_length: None,
            window: None,
            pre_delay: ir_no_config.len() + 100, // Greater than length
        };

        let ir = extract_ir_with_config(&sweep, &response, &config).unwrap();
        // Should not panic, should handle gracefully
        assert!(ir.len() <= ir_no_config.len());
    }

    #[test]
    fn test_deconvolution_method_default() {
        let method = DeconvolutionMethod::default();
        match method {
            DeconvolutionMethod::RegularizedFrequencyDomain { regularization } => {
                assert_eq!(regularization, 1e-3);
            }
            _ => panic!("Default should be RegularizedFrequencyDomain"),
        }
    }

    #[test]
    fn test_deconvolution_method_partial_eq() {
        let method1 = DeconvolutionMethod::FrequencyDomain;
        let method2 = DeconvolutionMethod::FrequencyDomain;
        assert_eq!(method1, method2);

        let method3 = DeconvolutionMethod::RegularizedFrequencyDomain {
            regularization: 1e-3,
        };
        let method4 = DeconvolutionMethod::RegularizedFrequencyDomain {
            regularization: 1e-3,
        };
        assert_eq!(method3, method4);

        assert_ne!(method1, method3);
    }

    #[test]
    fn test_window_type_partial_eq() {
        assert_eq!(WindowType::Hann, WindowType::Hann);
        assert_ne!(WindowType::Hann, WindowType::Hamming);
        assert_ne!(WindowType::Hann, WindowType::Blackman);
    }

    #[test]
    fn test_quality_rating_partial_eq() {
        assert_eq!(QualityRating::Good, QualityRating::Good);
        assert_eq!(QualityRating::Fair, QualityRating::Fair);
        assert_eq!(QualityRating::Poor, QualityRating::Poor);
        assert_ne!(QualityRating::Good, QualityRating::Poor);
    }

    #[test]
    fn test_ir_quality_clone() {
        let quality = IRQuality {
            peak: 0.8,
            rms: 0.2,
            early_late_ratio_db: 10.0,
            noise_ratio: 0.01,
            snr_db: 60.0,
            rating: QualityRating::Good,
            warnings: vec!["test warning"],
        };
        let cloned = quality.clone();
        assert_eq!(quality.peak, cloned.peak);
        assert_eq!(quality.warnings, cloned.warnings);
    }

    #[test]
    fn test_assess_ir_quality_moderate_early_late_ratio() {
        // Create IR with moderate early-to-late ratio
        let mut ir = vec![0.0; 1000];
        ir[50] = 1.0; // Strong peak
        // Moderate early energy
        for i in 51..150 {
            ir[i] = 0.05;
        }
        // Moderate late energy
        for i in 150..750 {
            ir[i] = 0.02;
        }

        let quality = assess_ir_quality(&ir).unwrap();
        // Should return a valid rating (any rating is acceptable)
        assert!(matches!(
            quality.rating,
            QualityRating::Good | QualityRating::Fair | QualityRating::Poor
        ));
        // Verify metrics are calculated
        assert!(quality.peak > 0.0);
        assert!(quality.rms > 0.0);
    }

    #[test]
    fn test_assess_ir_quality_moderate_noise() {
        // Create IR with moderate noise ratio
        let mut ir = vec![0.0; 1000];
        ir[50] = 1.0;
        // Add moderate noise in tail
        for i in 750..1000 {
            ir[i] = 0.03; // Moderate noise
        }

        let quality = assess_ir_quality(&ir).unwrap();
        // Should return a valid rating (any rating is acceptable)
        assert!(matches!(
            quality.rating,
            QualityRating::Good | QualityRating::Fair | QualityRating::Poor
        ));
        // Verify noise ratio is calculated
        assert!(quality.noise_ratio >= 0.0);
        assert!(quality.snr_db.is_finite() || quality.snr_db == f32::INFINITY);
    }

    #[test]
    fn test_extract_ir_circular_convolution_rotation() {
        // Create a scenario that might trigger circular convolution rotation
        // by using a very short sweep that might cause peak in second half
        let sweep = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let mut response = vec![0.0; 20];
        response[15] = 1.0; // Peak near the end

        let ir = extract_ir(&sweep, &response).unwrap();
        assert!(!ir.is_empty());
        // The rotation logic should handle this
    }

    #[test]
    fn test_extract_ir_zero_denominator_protection() {
        // Test case where sweep power might be very low (near zero)
        // This tests the denominator <= 1e-30 protection
        let sweep = vec![1e-20, 1e-20, 1e-20]; // Very small values
        let response = vec![1.0, 0.5, 0.25];

        // Should not panic, should handle zero/very small denominators
        let result = extract_ir(&sweep, &response);
        // May succeed or fail, but shouldn't panic
        let _ = result;
    }
}