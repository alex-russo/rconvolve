//! Fast FFT-based convolution for real-time audio processing
//!
//! This module implements overlap-save convolution using rustfft for efficient
//! convolution reverb and cabinet simulation applications.

use crate::{utils::next_power_of_two, AudioBuffer, AudioError, AudioResult, Sample};
use realfft::RealFftPlanner;
use rustfft::{num_complex::Complex, FftPlanner};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec;

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;

#[cfg(feature = "std")]
use std::sync::Arc;

/// Configuration for batch convolution
///
/// # Example
///
/// ```rust
/// use rconvolve::convolve;
///
/// let dry_audio = vec![1.0, 0.0, 0.0, 0.0];
/// let impulse_response = vec![0.5, 0.3, 0.1];
/// let wet_audio = convolve::apply_ir(&dry_audio, &impulse_response).unwrap();
/// assert_eq!(wet_audio.len(), dry_audio.len() + impulse_response.len() - 1);
/// ```
#[derive(Debug, Clone)]
pub struct ConvolutionConfig {
    /// Whether to normalize the output to peak level 1.0
    pub normalize: bool,
    /// Trim the output to input length (removes reverb tail)
    ///
    /// If `true`, output length equals input length.
    /// If `false`, output includes the full reverb tail (input_len + ir_len - 1).
    pub trim_to_input: bool,
    /// Dry/wet mix (0.0 = all dry, 1.0 = all wet)
    ///
    /// Controls the blend between original signal and convolved signal.
    pub wet_level: Sample,
    /// Output gain multiplier
    ///
    /// Applied after dry/wet mixing. Use to compensate for level changes.
    pub gain: Sample,
}

impl Default for ConvolutionConfig {
    fn default() -> Self {
        Self {
            normalize: false,
            trim_to_input: false,
            wet_level: 1.0,
            gain: 1.0,
        }
    }
}

/// Apply impulse response to audio using FFT convolution
///
/// Uses default configuration (full wet, no normalization, includes tail).
/// For custom settings, use [`apply_ir_with_config`].
///
/// # Example
///
/// ```rust
/// use rconvolve::convolve;
///
/// let dry_audio = vec![1.0, 0.0, 0.0, 0.0];
/// let impulse_response = vec![0.5, 0.3, 0.1];
/// let wet_audio = convolve::apply_ir(&dry_audio, &impulse_response).unwrap();
/// ```
pub fn apply_ir(input: &[Sample], impulse_response: &[Sample]) -> AudioResult<AudioBuffer> {
    apply_ir_with_config(input, impulse_response, &ConvolutionConfig::default())
}

/// Apply impulse response with custom configuration
///
/// Allows fine-tuning of convolution parameters including dry/wet mix,
/// normalization, and output trimming.
///
/// # Example
///
/// ```rust
/// use rconvolve::convolve;
///
/// let input = vec![1.0, 0.0, 0.0, 0.0];
/// let ir = vec![0.5, 0.3, 0.1];
///
/// let config = convolve::ConvolutionConfig {
///     normalize: false,
///     trim_to_input: true,
///     wet_level: 0.5,  // 50% wet
///     gain: 1.0,
/// };
/// let mixed = convolve::apply_ir_with_config(&input, &ir, &config).unwrap();
/// ```
pub fn apply_ir_with_config(
    input: &[Sample],
    impulse_response: &[Sample],
    config: &ConvolutionConfig,
) -> AudioResult<AudioBuffer> {
    if input.is_empty() || impulse_response.is_empty() {
        return Err(AudioError::InsufficientData);
    }

    let convolved = fft_convolve(input, impulse_response)?;

    let mut output = if config.trim_to_input {
        convolved.into_iter().take(input.len()).collect()
    } else {
        convolved
    };

    // Apply dry/wet mix
    if config.wet_level < 1.0 {
        let dry_level = 1.0 - config.wet_level;
        for (i, wet_sample) in output.iter_mut().enumerate() {
            if i < input.len() {
                *wet_sample = dry_level * input[i] + config.wet_level * *wet_sample;
            } else {
                *wet_sample *= config.wet_level;
            }
        }
    }

    // Apply gain
    if config.gain != 1.0 {
        for sample in &mut output {
            *sample *= config.gain;
        }
    }

    // Normalize if requested
    if config.normalize {
        normalize_buffer(&mut output);
    }

    Ok(output)
}

/// Perform FFT-based convolution
///
/// Lower-level function that performs FFT convolution without any post-processing.
/// Output length is `signal.len() + kernel.len() - 1`.
///
/// # Example
///
/// ```rust
/// use rconvolve::convolve;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
/// let kernel = vec![0.5, 0.3, 0.1];
/// let result = convolve::fft_convolve(&signal, &kernel).unwrap();
/// ```
pub fn fft_convolve(signal: &[Sample], kernel: &[Sample]) -> AudioResult<AudioBuffer> {
    if signal.is_empty() || kernel.is_empty() {
        return Err(AudioError::InsufficientData);
    }

    let output_length = signal.len() + kernel.len() - 1;
    let fft_size = next_power_of_two(output_length);

    // Create FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    // Prepare padded signals
    let mut signal_complex: Vec<Complex<Sample>> = signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(core::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    let mut kernel_complex: Vec<Complex<Sample>> = kernel
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(core::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    // Forward FFT
    fft.process(&mut signal_complex);
    fft.process(&mut kernel_complex);

    // Multiply in frequency domain
    for (sig, ker) in signal_complex.iter_mut().zip(kernel_complex.iter()) {
        *sig *= ker;
    }

    // Inverse FFT
    ifft.process(&mut signal_complex);

    // Extract real part, normalize, and truncate to correct length
    let output: Vec<Sample> = signal_complex
        .iter()
        .take(output_length)
        .map(|c| c.re / (fft_size as Sample))
        .collect();

    Ok(output)
}

/// Perform time-domain convolution (for comparison/validation)
///
/// Slower O(n²) implementation that produces identical results to FFT convolution.
/// Useful for validation and testing. For production use, prefer [`fft_convolve`].
///
/// # Example
///
/// ```rust
/// use rconvolve::convolve;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
/// let kernel = vec![0.5, 0.3, 0.1];
/// let result = convolve::time_convolve(&signal, &kernel).unwrap();
/// ```
pub fn time_convolve(signal: &[Sample], kernel: &[Sample]) -> AudioResult<AudioBuffer> {
    if signal.is_empty() || kernel.is_empty() {
        return Err(AudioError::InsufficientData);
    }

    let output_length = signal.len() + kernel.len() - 1;
    let mut output = vec![0.0; output_length];

    for (i, &sig_sample) in signal.iter().enumerate() {
        for (j, &kernel_sample) in kernel.iter().enumerate() {
            output[i + j] += sig_sample * kernel_sample;
        }
    }

    Ok(output)
}

/// Partitioned convolution for very long impulse responses
///
/// Uses uniform partitioned convolution with real FFT for efficient real-time processing.
/// This allows processing of very long impulse responses (several seconds) with low latency.
///
/// # Partition Size
///
/// The `partition_size` parameter in [`new`](Self::new) is currently ignored.
/// Partitions always equal the block size for optimal performance. Future versions
/// may support non-uniform partitioning for different latency/CPU tradeoffs.
///
/// # Example
///
/// ```rust
/// use rconvolve::convolve::PartitionedConvolution;
///
/// let impulse_response = vec![0.5, 0.3, 0.1];
/// let input_block = vec![1.0; 512];
/// let mut processor = PartitionedConvolution::new(impulse_response, 512, None).unwrap();
/// let output = processor.process_block(&input_block).unwrap();
/// assert_eq!(output.len(), input_block.len());
/// ```
pub struct PartitionedConvolution {
    /// FFT'd partitions of the impulse response (hermitian symmetric format from real FFT)
    ir_partitions: Vec<Vec<Complex<Sample>>>,
    /// Size of the FFT (power of 2, at least 2 * partition_size)
    fft_size: usize,
    /// Block size for processing
    block_size: usize,
    /// Real FFT planner (real to complex)
    r2c: Arc<dyn realfft::RealToComplex<Sample>>,
    /// Real FFT planner (complex to real)
    c2r: Arc<dyn realfft::ComplexToReal<Sample>>,
    /// Frequency-domain delay line storing recent input spectra (hermitian symmetric)
    fdl: Vec<Vec<Complex<Sample>>>,
    /// Current position in the FDL
    fdl_position: usize,
    /// Temporary buffers
    input_buffer: AudioBuffer,
    spectrum_buffer: Vec<Complex<Sample>>,
    output_buffer: AudioBuffer,
    overlap_buffer: AudioBuffer,
}

impl PartitionedConvolution {
    /// Create a new partitioned convolution processor
    ///
    /// Uses uniform partitioned convolution with real FFT for efficient processing.
    ///
    /// # Arguments
    /// * `impulse_response` - The impulse response to convolve with
    /// * `block_size` - Processing block size (must match input block size)
    /// * `_partition_size` - Currently ignored; partitions always equal `block_size`
    ///
    /// # Returns
    /// Processor ready for real-time audio processing
    ///
    /// # Note
    /// The `partition_size` parameter is currently ignored. Partitions always equal
    /// `block_size` for optimal performance. Future versions may support non-uniform
    /// partitioning for different latency/CPU tradeoffs.
    pub fn new(
        impulse_response: AudioBuffer,
        block_size: usize,
        _partition_size: Option<usize>,
    ) -> AudioResult<Self> {
        if impulse_response.is_empty() {
            return Err(AudioError::InsufficientData);
        }

        // For now, partition_size must equal block_size for correct operation
        // Supporting partition_size != block_size requires a more complex algorithm
        let partition_size = block_size;

        // FFT size must be at least 2 * partition_size for linear convolution
        let fft_size = next_power_of_two(partition_size * 2);
        let spectrum_size = fft_size / 2 + 1; // Hermitian symmetry means we only store half

        // Create Real FFT planners
        let mut planner = RealFftPlanner::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let c2r = planner.plan_fft_inverse(fft_size);

        // Partition and FFT the impulse response
        let num_partitions = impulse_response.len().div_ceil(partition_size);
        let mut ir_partitions = Vec::with_capacity(num_partitions);

        let mut scratch_buffer = vec![0.0; fft_size];
        let mut scratch_spectrum = vec![Complex::new(0.0, 0.0); spectrum_size];

        for i in 0..num_partitions {
            let start = i * partition_size;
            let end = (start + partition_size).min(impulse_response.len());

            // Zero-pad the partition to FFT size
            scratch_buffer.fill(0.0);
            scratch_buffer[..end - start].copy_from_slice(&impulse_response[start..end]);

            // Real FFT the partition
            r2c.process(&mut scratch_buffer, &mut scratch_spectrum)
                .map_err(|_| AudioError::InsufficientData)?;

            ir_partitions.push(scratch_spectrum.clone());
        }

        // Frequency-domain delay line (circular buffer of spectra)
        let fdl = vec![vec![Complex::new(0.0, 0.0); spectrum_size]; num_partitions];

        let input_buffer = vec![0.0; fft_size];
        let spectrum_buffer = vec![Complex::new(0.0, 0.0); spectrum_size];
        let output_buffer = vec![0.0; fft_size];
        let overlap_buffer = vec![0.0; fft_size - block_size];

        Ok(Self {
            ir_partitions,
            fft_size,
            block_size,
            r2c,
            c2r,
            fdl,
            fdl_position: 0,
            input_buffer,
            spectrum_buffer,
            output_buffer,
            overlap_buffer,
        })
    }

    /// Process a block of audio using uniform partitioned convolution
    ///
    /// # Arguments
    /// * `input` - Input audio block (must be exactly `block_size` samples)
    ///
    /// # Returns
    /// Convolved output block (same length as input)
    ///
    /// # Errors
    /// Returns `BufferSizeMismatch` if input length doesn't match `block_size`
    pub fn process_block(&mut self, input: &[Sample]) -> AudioResult<AudioBuffer> {
        if input.len() != self.block_size {
            return Err(AudioError::BufferSizeMismatch);
        }

        // Zero-pad input to FFT size
        self.input_buffer.fill(0.0);
        self.input_buffer[..self.block_size].copy_from_slice(input);

        // Real FFT the input block
        self.r2c
            .process(&mut self.input_buffer, &mut self.spectrum_buffer)
            .map_err(|_| AudioError::InsufficientData)?;

        // Store spectrum in the frequency-domain delay line
        self.fdl[self.fdl_position].copy_from_slice(&self.spectrum_buffer);

        // Clear spectrum buffer for accumulation
        for sample in self.spectrum_buffer.iter_mut() {
            *sample = Complex::new(0.0, 0.0);
        }

        // Convolve with all partitions
        let num_partitions = self.ir_partitions.len();
        for i in 0..num_partitions {
            // Calculate which FDL slot to read from
            let fdl_index = (self.fdl_position + num_partitions - i) % num_partitions;

            // Multiply in frequency domain and accumulate
            // Use iterator-based approach which allows LLVM to auto-vectorize
            for ((spec, &fdl_val), &ir_val) in self
                .spectrum_buffer
                .iter_mut()
                .zip(&self.fdl[fdl_index])
                .zip(&self.ir_partitions[i])
            {
                *spec += fdl_val * ir_val;
            }
        }

        // Inverse real FFT to get time-domain result
        self.c2r
            .process(&mut self.spectrum_buffer, &mut self.output_buffer)
            .map_err(|_| AudioError::InsufficientData)?;

        // Normalize by FFT size (real FFT doesn't do this automatically)
        let scale = 1.0 / (self.fft_size as Sample);
        for sample in self.output_buffer.iter_mut() {
            *sample *= scale;
        }

        // Overlap-add: extract output and update overlap buffer
        let mut output = vec![0.0; self.block_size];

        // Add overlap buffer to output
        for (out, (&output_val, &overlap_val)) in output
            .iter_mut()
            .zip(self.output_buffer.iter().zip(&self.overlap_buffer))
        {
            *out = output_val + overlap_val;
        }

        // Update overlap buffer
        // Simply copy the tail since partition_size == block_size
        let overlap_len = self.overlap_buffer.len();
        for i in 0..overlap_len {
            let src_idx = self.block_size + i;
            self.overlap_buffer[i] = if src_idx < self.output_buffer.len() {
                self.output_buffer[src_idx]
            } else {
                0.0
            };
        }

        // Advance FDL position
        self.fdl_position = (self.fdl_position + 1) % num_partitions;

        Ok(output)
    }

    /// Reset the processor state
    ///
    /// Clears all internal buffers and delay lines. Use this when you need to
    /// start processing a new audio stream or want to eliminate any residual
    /// reverb tail from previous processing.
    pub fn reset(&mut self) {
        self.fdl_position = 0;
        for spectrum in &mut self.fdl {
            spectrum.fill(Complex::new(0.0, 0.0));
        }
        self.overlap_buffer.fill(0.0);
        self.input_buffer.fill(0.0);
    }
}

/// Stereo convolution processor
///
/// Processes left and right channels independently with separate impulse responses.
/// This is suitable for stereo reverb where L and R have different characteristics
/// but don't cross-feed (left input only affects left output, etc.).
///
/// For full spatial reverb with cross-channel interaction, use [`TrueStereoConvolution`].
///
/// # Example
///
/// ```rust
/// use rconvolve::convolve::StereoConvolution;
///
/// let ir_left = vec![1.0, 0.5];
/// let ir_right = vec![0.8, 0.4];
/// let block_size = 4;
///
/// let mut stereo = StereoConvolution::new(ir_left, ir_right, block_size).unwrap();
///
/// // Process stereo blocks
/// let input_left = vec![1.0, 0.0, 0.0, 0.0];
/// let input_right = vec![0.5, 0.0, 0.0, 0.0];
/// let (out_left, out_right) = stereo.process_block(&input_left, &input_right).unwrap();
/// let _ = (out_left, out_right);
///
/// // Or process interleaved audio
/// let interleaved_input = vec![1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let interleaved_output = stereo.process_interleaved(&interleaved_input).unwrap();
/// let _ = interleaved_output;
/// ```
pub struct StereoConvolution {
    /// Left channel processor
    left: PartitionedConvolution,
    /// Right channel processor
    right: PartitionedConvolution,
    /// Block size for processing
    block_size: usize,
}

impl StereoConvolution {
    /// Create a new stereo convolution processor
    ///
    /// # Arguments
    /// * `ir_left` - Impulse response for left channel
    /// * `ir_right` - Impulse response for right channel
    /// * `block_size` - Processing block size (must match input block size)
    pub fn new(
        ir_left: AudioBuffer,
        ir_right: AudioBuffer,
        block_size: usize,
    ) -> AudioResult<Self> {
        let left = PartitionedConvolution::new(ir_left, block_size, None)?;
        let right = PartitionedConvolution::new(ir_right, block_size, None)?;

        Ok(Self {
            left,
            right,
            block_size,
        })
    }

    /// Create a stereo processor from a mono IR (applies same IR to both channels)
    pub fn from_mono(ir: AudioBuffer, block_size: usize) -> AudioResult<Self> {
        Self::new(ir.clone(), ir, block_size)
    }

    /// Process a stereo block of audio
    ///
    /// # Arguments
    /// * `input_left` - Left channel input samples
    /// * `input_right` - Right channel input samples
    ///
    /// # Returns
    /// Tuple of (left_output, right_output) buffers
    pub fn process_block(
        &mut self,
        input_left: &[Sample],
        input_right: &[Sample],
    ) -> AudioResult<(AudioBuffer, AudioBuffer)> {
        let out_left = self.left.process_block(input_left)?;
        let out_right = self.right.process_block(input_right)?;
        Ok((out_left, out_right))
    }

    /// Process interleaved stereo audio (L, R, L, R, ...)
    ///
    /// # Arguments
    /// * `interleaved` - Interleaved stereo samples (length must be 2 * block_size)
    ///
    /// # Returns
    /// Interleaved output buffer
    pub fn process_interleaved(&mut self, interleaved: &[Sample]) -> AudioResult<AudioBuffer> {
        if interleaved.len() != self.block_size * 2 {
            return Err(AudioError::BufferSizeMismatch);
        }

        // Deinterleave
        let mut left_in = vec![0.0; self.block_size];
        let mut right_in = vec![0.0; self.block_size];

        for i in 0..self.block_size {
            left_in[i] = interleaved[i * 2];
            right_in[i] = interleaved[i * 2 + 1];
        }

        // Process
        let (left_out, right_out) = self.process_block(&left_in, &right_in)?;

        // Interleave output
        let mut output = vec![0.0; self.block_size * 2];
        for i in 0..self.block_size {
            output[i * 2] = left_out[i];
            output[i * 2 + 1] = right_out[i];
        }

        Ok(output)
    }

    /// Reset both channel processors
    ///
    /// Clears all internal state. Use when starting a new audio stream.
    pub fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
    }

    /// Get the configured block size
    ///
    /// Returns the block size used for processing. Input blocks must match this size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

/// True stereo convolution processor (4-channel matrix)
///
/// Implements full spatial convolution with cross-channel interaction, simulating
/// how sound in a real space reflects between left and right channels.
///
/// # 4-Channel Matrix
///
/// The 4 impulse responses represent:
/// - **LL**: Left input → Left output (direct left reflection)
/// - **LR**: Left input → Right output (cross-feed left to right)
/// - **RL**: Right input → Left output (cross-feed right to left)  
/// - **RR**: Right input → Right output (direct right reflection)
///
/// # Example
///
/// ```rust
/// use rconvolve::convolve::TrueStereoConvolution;
///
/// let block_size = 4;
/// // From 4 separate IRs
/// let ir_ll = vec![1.0, 0.0];
/// let ir_lr = vec![0.3, 0.0];
/// let ir_rl = vec![0.2, 0.0];
/// let ir_rr = vec![1.0, 0.0];
/// let mut processor =
///     TrueStereoConvolution::new(ir_ll, ir_lr, ir_rl, ir_rr, block_size).unwrap();
///
/// // Process stereo blocks
/// let input_left = vec![1.0, 0.0, 0.0, 0.0];
/// let input_right = vec![0.0, 0.0, 0.0, 0.0];
/// let (out_left, out_right) = processor.process_block(&input_left, &input_right).unwrap();
/// let _ = (out_left, out_right);
/// ```
pub struct TrueStereoConvolution {
    /// Left input → Left output
    ll: PartitionedConvolution,
    /// Left input → Right output
    lr: PartitionedConvolution,
    /// Right input → Left output
    rl: PartitionedConvolution,
    /// Right input → Right output
    rr: PartitionedConvolution,
    /// Block size for processing
    block_size: usize,
}

impl TrueStereoConvolution {
    /// Create a new true stereo convolution processor
    ///
    /// # Arguments
    /// * `ir_ll` - Left input → Left output impulse response
    /// * `ir_lr` - Left input → Right output impulse response
    /// * `ir_rl` - Right input → Left output impulse response
    /// * `ir_rr` - Right input → Right output impulse response
    /// * `block_size` - Processing block size
    pub fn new(
        ir_ll: AudioBuffer,
        ir_lr: AudioBuffer,
        ir_rl: AudioBuffer,
        ir_rr: AudioBuffer,
        block_size: usize,
    ) -> AudioResult<Self> {
        let ll = PartitionedConvolution::new(ir_ll, block_size, None)?;
        let lr = PartitionedConvolution::new(ir_lr, block_size, None)?;
        let rl = PartitionedConvolution::new(ir_rl, block_size, None)?;
        let rr = PartitionedConvolution::new(ir_rr, block_size, None)?;

        Ok(Self {
            ll,
            lr,
            rl,
            rr,
            block_size,
        })
    }

    /// Create true stereo from a stereo IR pair with synthetic cross-feed
    ///
    /// This creates LR and RL channels by attenuating and delaying the opposite channel's IR,
    /// which approximates natural cross-channel reflections.
    ///
    /// # Arguments
    /// * `ir_left` - Left channel impulse response (becomes LL)
    /// * `ir_right` - Right channel impulse response (becomes RR)
    /// * `block_size` - Processing block size
    /// * `cross_feed_gain` - Gain for cross-feed channels (0.0 to 1.0, typical: 0.3-0.5)
    /// * `cross_feed_delay_samples` - Delay for cross-feed in samples (typical: 10-50 at 48kHz)
    pub fn from_stereo_with_crossfeed(
        ir_left: AudioBuffer,
        ir_right: AudioBuffer,
        block_size: usize,
        cross_feed_gain: Sample,
        cross_feed_delay_samples: usize,
    ) -> AudioResult<Self> {
        // Create cross-feed IRs by delaying and attenuating the opposite channel
        let lr = create_delayed_ir(&ir_left, cross_feed_delay_samples, cross_feed_gain);
        let rl = create_delayed_ir(&ir_right, cross_feed_delay_samples, cross_feed_gain);

        Self::new(ir_left, lr, rl, ir_right, block_size)
    }

    /// Create true stereo from a mono IR with synthetic stereo spread
    ///
    /// This creates a pseudo-stereo effect from a mono IR by:
    /// - Using the original IR for LL and RR
    /// - Creating slightly different versions for LR and RL
    pub fn from_mono_with_spread(
        ir: AudioBuffer,
        block_size: usize,
        spread: Sample, // 0.0 = mono, 1.0 = full stereo spread
    ) -> AudioResult<Self> {
        let cross_gain = spread * 0.4; // Cross-feed at 40% of spread amount
        let delay = ((1.0 - spread * 0.5) * 20.0) as usize; // 10-20 sample delay based on spread

        Self::from_stereo_with_crossfeed(ir.clone(), ir, block_size, cross_gain, delay)
    }

    /// Process a stereo block of audio with full matrix convolution
    ///
    /// # Arguments
    /// * `input_left` - Left channel input samples
    /// * `input_right` - Right channel input samples
    ///
    /// # Returns
    /// Tuple of (left_output, right_output) buffers
    pub fn process_block(
        &mut self,
        input_left: &[Sample],
        input_right: &[Sample],
    ) -> AudioResult<(AudioBuffer, AudioBuffer)> {
        // Process all four paths
        let ll_out = self.ll.process_block(input_left)?;
        let lr_out = self.lr.process_block(input_left)?;
        let rl_out = self.rl.process_block(input_right)?;
        let rr_out = self.rr.process_block(input_right)?;

        // Mix: Left output = LL + RL, Right output = LR + RR
        let mut left_out = vec![0.0; self.block_size];
        let mut right_out = vec![0.0; self.block_size];

        for i in 0..self.block_size {
            left_out[i] = ll_out[i] + rl_out[i];
            right_out[i] = lr_out[i] + rr_out[i];
        }

        Ok((left_out, right_out))
    }

    /// Process interleaved stereo audio (L, R, L, R, ...)
    pub fn process_interleaved(&mut self, interleaved: &[Sample]) -> AudioResult<AudioBuffer> {
        if interleaved.len() != self.block_size * 2 {
            return Err(AudioError::BufferSizeMismatch);
        }

        // Deinterleave
        let mut left_in = vec![0.0; self.block_size];
        let mut right_in = vec![0.0; self.block_size];

        for i in 0..self.block_size {
            left_in[i] = interleaved[i * 2];
            right_in[i] = interleaved[i * 2 + 1];
        }

        // Process
        let (left_out, right_out) = self.process_block(&left_in, &right_in)?;

        // Interleave output
        let mut output = vec![0.0; self.block_size * 2];
        for i in 0..self.block_size {
            output[i * 2] = left_out[i];
            output[i * 2 + 1] = right_out[i];
        }

        Ok(output)
    }

    /// Reset all channel processors
    ///
    /// Clears all internal state. Use when starting a new audio stream.
    pub fn reset(&mut self) {
        self.ll.reset();
        self.lr.reset();
        self.rl.reset();
        self.rr.reset();
    }

    /// Get the configured block size
    ///
    /// Returns the block size used for processing. Input blocks must match this size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

/// Helper function to create a delayed and attenuated IR for cross-feed
fn create_delayed_ir(ir: &[Sample], delay_samples: usize, gain: Sample) -> AudioBuffer {
    let mut delayed = vec![0.0; ir.len() + delay_samples];
    for (i, &sample) in ir.iter().enumerate() {
        delayed[i + delay_samples] = sample * gain;
    }
    delayed
}

/// Batch stereo convolution (non-real-time)
///
/// Convolves stereo audio with stereo impulse responses in a single operation.
pub fn stereo_fft_convolve(
    left: &[Sample],
    right: &[Sample],
    ir_left: &[Sample],
    ir_right: &[Sample],
) -> AudioResult<(AudioBuffer, AudioBuffer)> {
    let out_left = fft_convolve(left, ir_left)?;
    let out_right = fft_convolve(right, ir_right)?;
    Ok((out_left, out_right))
}

/// Batch true stereo convolution (non-real-time)
///
/// Convolves stereo audio with a 4-channel true stereo IR matrix.
pub fn true_stereo_fft_convolve(
    left: &[Sample],
    right: &[Sample],
    ir_ll: &[Sample],
    ir_lr: &[Sample],
    ir_rl: &[Sample],
    ir_rr: &[Sample],
) -> AudioResult<(AudioBuffer, AudioBuffer)> {
    let ll = fft_convolve(left, ir_ll)?;
    let lr = fft_convolve(left, ir_lr)?;
    let rl = fft_convolve(right, ir_rl)?;
    let rr = fft_convolve(right, ir_rr)?;

    // Output length is the maximum of all convolutions
    let out_len = ll.len().max(lr.len()).max(rl.len()).max(rr.len());

    let mut out_left = vec![0.0; out_len];
    let mut out_right = vec![0.0; out_len];

    for i in 0..out_len {
        out_left[i] = ll.get(i).copied().unwrap_or(0.0) + rl.get(i).copied().unwrap_or(0.0);
        out_right[i] = lr.get(i).copied().unwrap_or(0.0) + rr.get(i).copied().unwrap_or(0.0);
    }

    Ok((out_left, out_right))
}

fn normalize_buffer(buffer: &mut [Sample]) {
    let max_amplitude = buffer.iter().map(|&x| x.abs()).fold(0.0, f32::max);
    if max_amplitude > 0.0 {
        let scale = 1.0 / max_amplitude;
        for sample in buffer {
            *sample *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn test_fft_convolution_basic() {
        let signal = vec![1.0, 0.0, 0.0, 0.0];
        let kernel = vec![1.0, 0.5, 0.25];

        let result = fft_convolve(&signal, &kernel).unwrap();
        let expected = vec![1.0, 0.5, 0.25, 0.0, 0.0, 0.0];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(r, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_fft_vs_time_convolution() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![0.5, 0.3, 0.1];

        let fft_result = fft_convolve(&signal, &kernel).unwrap();
        let time_result = time_convolve(&signal, &kernel).unwrap();

        assert_eq!(fft_result.len(), time_result.len());

        for (fft, time) in fft_result.iter().zip(time_result.iter()) {
            assert_abs_diff_eq!(fft, time, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_apply_ir_with_config() {
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let ir = vec![0.5, 0.25];

        let config = ConvolutionConfig {
            normalize: false,
            trim_to_input: true,
            wet_level: 0.5,
            gain: 2.0,
        };

        let result = apply_ir_with_config(&input, &ir, &config).unwrap();

        assert_eq!(result.len(), input.len());

        // Check that dry/wet mix was applied
        assert_abs_diff_eq!(result[0], (0.5 * 1.0 + 0.5 * 0.5) * 2.0, epsilon = 1e-6);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_partitioned_convolution_simple() {
        // Very simple test: impulse input with IR longer than block size
        let ir = vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125];
        let block_size = 4;

        let mut partitioned = PartitionedConvolution::new(ir.clone(), block_size, None).unwrap();

        // First block: impulse at start
        let input1 = vec![1.0, 0.0, 0.0, 0.0];
        let output1 = partitioned.process_block(&input1).unwrap();

        println!("Input 1: {:?}", input1);
        println!("Output 1: {:?}", output1);

        // Second block: all zeros
        let input2 = vec![0.0, 0.0, 0.0, 0.0];
        let output2 = partitioned.process_block(&input2).unwrap();

        println!("Output 2 (should have IR tail): {:?}", output2);

        // The first output should start with 1.0 (the impulse convolved with IR[0])
        assert_abs_diff_eq!(output1[0], 1.0, epsilon = 1e-4);

        // The second output should have the tail from the IR
        // Since IR is 6 samples and block is 4, samples 4-5 of IR (0.0625, 0.03125)
        // should appear in block 2
        let energy: f32 = output2.iter().map(|&x| x * x).sum();
        println!("Output 2 energy: {}", energy);
        assert!(energy > 0.0, "Second block should have energy from IR tail");

        // Check specific values
        assert_abs_diff_eq!(output2[0], 0.0625, epsilon = 1e-4);
        assert_abs_diff_eq!(output2[1], 0.03125, epsilon = 1e-4);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_partitioned_vs_standard_convolution() {
        // Create a simple impulse response
        let ir = vec![1.0, 0.5, 0.25, 0.125];
        let block_size = 4;

        // Create a longer test signal (multiple blocks)
        let test_signal = vec![
            1.0, 0.0, 0.0, 0.0, // Block 1
            0.0, 1.0, 0.0, 0.0, // Block 2
            0.0, 0.0, 1.0, 0.0, // Block 3
        ];

        // Process with partitioned convolution
        let mut partitioned = PartitionedConvolution::new(ir.clone(), block_size, None).unwrap();
        let mut partitioned_output = Vec::new();

        for chunk in test_signal.chunks(block_size) {
            let block_output = partitioned.process_block(chunk).unwrap();
            partitioned_output.extend_from_slice(&block_output);
        }

        // Process with standard FFT convolution
        let standard_output = fft_convolve(&test_signal, &ir).unwrap();

        // Compare outputs (should match closely)
        println!("Partitioned output: {:?}", &partitioned_output[..12]);
        println!("Standard output:    {:?}", &standard_output[..12]);

        // Print more debug info
        println!(
            "Partitioned total length: {}, Standard total length: {}",
            partitioned_output.len(),
            standard_output.len()
        );

        let part_energy: f32 = partitioned_output.iter().map(|x| x * x).sum();
        let std_energy: f32 = standard_output.iter().map(|x| x * x).sum();
        println!(
            "Partitioned energy: {}, Standard energy: {}",
            part_energy, std_energy
        );

        for (_i, (part, std)) in partitioned_output
            .iter()
            .zip(standard_output.iter())
            .enumerate()
        {
            assert_abs_diff_eq!(part, std, epsilon = 1e-4);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_partitioned_with_large_partition_size() {
        // Test with block_size=512 (partition_size is forced to equal block_size)
        let block_size = 512;

        // Create a longer IR (like a reverb tail)
        let mut ir = vec![0.0; 4800]; // 100ms at 48kHz
        ir[0] = 1.0;
        for i in 1..ir.len() {
            ir[i] = (-5.0 * i as f32 / 48000.0).exp() * 0.3;
        }

        // Create test signal with impulses
        let mut test_signal = vec![0.0; block_size * 10]; // 10 blocks
        test_signal[0] = 1.0;
        test_signal[block_size] = 0.5;
        test_signal[block_size * 2] = 0.3;

        // Process with partitioned convolution (partition_size is now always block_size)
        let mut partitioned = PartitionedConvolution::new(ir.clone(), block_size, None).unwrap();
        let mut partitioned_output = Vec::new();

        for chunk in test_signal.chunks(block_size) {
            let block_output = partitioned.process_block(chunk).unwrap();
            partitioned_output.extend_from_slice(&block_output);
        }

        // Process with standard FFT convolution
        let standard_output = fft_convolve(&test_signal, &ir).unwrap();

        // Compare outputs
        println!("\nLarge partition test:");
        println!("Block size: {}, IR length: {}", block_size, ir.len());
        println!("First 10 samples:");
        println!("Partitioned: {:?}", &partitioned_output[..10]);
        println!("Standard:    {:?}", &standard_output[..10]);

        let mut max_diff = 0.0f32;
        let mut max_diff_idx = 0;
        for (i, (part, std)) in partitioned_output
            .iter()
            .zip(standard_output.iter())
            .enumerate()
        {
            let diff = (part - std).abs();
            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
            }
        }

        println!("Max difference: {} at sample {}", max_diff, max_diff_idx);

        for (i, (part, std)) in partitioned_output
            .iter()
            .zip(standard_output.iter())
            .enumerate()
        {
            if (part - std).abs() > 1e-3 {
                println!(
                    "Large diff at {}: part={}, std={}, diff={}",
                    i,
                    part,
                    std,
                    (part - std).abs()
                );
            }
            assert_abs_diff_eq!(part, std, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_normalize_buffer() {
        let mut buffer = vec![2.0, -4.0, 1.0, 3.0];
        normalize_buffer(&mut buffer);

        let max_val = buffer.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        assert_abs_diff_eq!(max_val, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_normalize_buffer_zero() {
        let mut buffer = vec![0.0, 0.0, 0.0];
        normalize_buffer(&mut buffer);
        assert_eq!(buffer, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_convolution_config_default() {
        let config = ConvolutionConfig::default();
        assert_eq!(config.normalize, false);
        assert_eq!(config.trim_to_input, false);
        assert_eq!(config.wet_level, 1.0);
        assert_eq!(config.gain, 1.0);
    }

    #[test]
    fn test_apply_ir_with_normalize() {
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let ir = vec![2.0, 2.0];

        let config = ConvolutionConfig {
            normalize: true,
            trim_to_input: false,
            wet_level: 1.0,
            gain: 1.0,
        };

        let result = apply_ir_with_config(&input, &ir, &config).unwrap();
        let max_val = result.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        assert_abs_diff_eq!(max_val, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_apply_ir_with_full_wet() {
        let input = vec![1.0, 0.5, 0.0, 0.0];
        let ir = vec![0.5, 0.25];

        let config = ConvolutionConfig {
            normalize: false,
            trim_to_input: false,
            wet_level: 1.0,
            gain: 1.0,
        };

        let result = apply_ir_with_config(&input, &ir, &config).unwrap();
        // With wet_level = 1.0, should just be the convolution
        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_apply_ir_with_gain_only() {
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let ir = vec![0.5];

        let config = ConvolutionConfig {
            normalize: false,
            trim_to_input: false,
            wet_level: 1.0,
            gain: 3.0,
        };

        let result = apply_ir_with_config(&input, &ir, &config).unwrap();
        assert_abs_diff_eq!(result[0], 1.5, epsilon = 1e-6);
    }

    #[test]
    fn test_apply_ir_empty_inputs() {
        let input: Vec<f32> = vec![];
        let ir = vec![0.5];
        assert_eq!(apply_ir(&input, &ir), Err(AudioError::InsufficientData));
        assert_eq!(apply_ir(&[1.0], &[]), Err(AudioError::InsufficientData));
    }

    #[test]
    fn test_partitioned_new_empty_ir() {
        let ir: Vec<f32> = vec![];
        let block_size = 4;
        let result = PartitionedConvolution::new(ir, block_size, None);
        assert!(matches!(result, Err(AudioError::InsufficientData)));
    }

    #[test]
    fn test_partitioned_process_block_size_mismatch() {
        let ir = vec![1.0, 0.5];
        let block_size = 4;
        let mut partitioned =
            PartitionedConvolution::new(ir, block_size, None).expect("should construct");
        let input_wrong = vec![1.0, 0.0, 0.0]; // len 3 instead of 4
        let result = partitioned.process_block(&input_wrong);
        assert!(matches!(result, Err(AudioError::BufferSizeMismatch)));
    }

    #[test]
    fn test_stereo_process_interleaved_mismatch() {
        let ir_left = vec![1.0, 0.5];
        let ir_right = vec![0.8, 0.4];
        let block_size = 4;
        let mut stereo = StereoConvolution::new(ir_left, ir_right, block_size).unwrap();
        let interleaved_wrong = vec![0.0; block_size * 2 - 1]; // wrong length
        let result = stereo.process_interleaved(&interleaved_wrong);
        assert!(matches!(result, Err(AudioError::BufferSizeMismatch)));
    }

    #[test]
    fn test_true_stereo_process_interleaved_mismatch() {
        let ir_ll = vec![1.0];
        let ir_lr = vec![0.3];
        let ir_rl = vec![0.2];
        let ir_rr = vec![1.0];
        let block_size = 4;
        let mut proc = TrueStereoConvolution::new(ir_ll, ir_lr, ir_rl, ir_rr, block_size).unwrap();
        let interleaved_wrong = vec![0.0; block_size * 2 - 1];
        let result = proc.process_interleaved(&interleaved_wrong);
        assert!(matches!(result, Err(AudioError::BufferSizeMismatch)));
    }

    #[test]
    fn test_block_size_getters() {
        let ir = vec![1.0, 0.5];
        let block_size = 4;
        let partitioned = PartitionedConvolution::new(ir.clone(), block_size, None).unwrap();
        assert_eq!(partitioned.block_size, block_size);

        let stereo = StereoConvolution::new(ir.clone(), ir.clone(), block_size).unwrap();
        assert_eq!(stereo.block_size(), block_size);

        let true_stereo =
            TrueStereoConvolution::from_mono_with_spread(ir.clone(), block_size, 0.5).unwrap();
        assert_eq!(true_stereo.block_size(), block_size);
    }

    #[test]
    fn test_fft_convolve_empty_inputs() {
        assert!(matches!(
            fft_convolve(&[], &[1.0]),
            Err(AudioError::InsufficientData)
        ));
        assert!(matches!(
            fft_convolve(&[1.0], &[]),
            Err(AudioError::InsufficientData)
        ));
    }

    #[test]
    fn test_time_convolve_empty_inputs() {
        assert!(matches!(
            time_convolve(&[], &[1.0]),
            Err(AudioError::InsufficientData)
        ));
        assert!(matches!(
            time_convolve(&[1.0], &[]),
            Err(AudioError::InsufficientData)
        ));
    }

    #[test]
    fn test_stereo_process_interleaved_success() {
        let ir_left = vec![1.0, 0.0];
        let ir_right = vec![1.0, 0.0];
        let block_size = 2;
        let mut stereo = StereoConvolution::new(ir_left, ir_right, block_size).unwrap();
        let interleaved_input = vec![1.0, 0.5, 0.0, 0.0]; // L0, R0, L1, R1
        let output = stereo.process_interleaved(&interleaved_input).unwrap();
        assert_eq!(output.len(), block_size * 2);
    }

    #[test]
    fn test_true_stereo_process_interleaved_success() {
        let ir_ll = vec![1.0];
        let ir_lr = vec![0.0];
        let ir_rl = vec![0.0];
        let ir_rr = vec![1.0];
        let block_size = 2;
        let mut proc = TrueStereoConvolution::new(ir_ll, ir_lr, ir_rl, ir_rr, block_size).unwrap();
        let interleaved = vec![1.0, 0.5, 0.0, 0.0]; // L0, R0, L1, R1
        let output = proc.process_interleaved(&interleaved).unwrap();
        assert_eq!(output.len(), block_size * 2);
    }

    #[test]
    fn test_apply_ir_with_wet_on_tail() {
        let input = vec![1.0, 0.0];
        let ir = vec![0.5, 0.25, 0.125];

        let config = ConvolutionConfig {
            normalize: false,
            trim_to_input: false,
            wet_level: 0.5,
            gain: 1.0,
        };

        let result = apply_ir_with_config(&input, &ir, &config).unwrap();
        // Last sample is beyond input length, should be wet_level * convolved
        assert_abs_diff_eq!(result[2], 0.5 * 0.125, epsilon = 1e-6);
    }

    #[test]
    fn test_partitioned_convolution_reset() {
        let ir = vec![1.0, 0.5, 0.25];
        let block_size = 4;

        let mut partitioned = PartitionedConvolution::new(ir, block_size, None).unwrap();

        // Process a block
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let _ = partitioned.process_block(&input).unwrap();

        // Reset
        partitioned.reset();

        // Process same block again - should get same result as fresh processor
        let output_after_reset = partitioned.process_block(&input).unwrap();
        assert_abs_diff_eq!(output_after_reset[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_partitioned_convolution_empty_ir() {
        let ir = vec![];
        let block_size = 4;

        let result = PartitionedConvolution::new(ir, block_size, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_partitioned_convolution_wrong_block_size() {
        let ir = vec![1.0, 0.5];
        let block_size = 4;

        let mut partitioned = PartitionedConvolution::new(ir, block_size, None).unwrap();

        let input = vec![1.0, 0.0, 0.0]; // Wrong size
        let result = partitioned.process_block(&input);
        assert_eq!(result, Err(AudioError::BufferSizeMismatch));
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(100), 128);
        assert_eq!(next_power_of_two(1024), 1024);
    }

    // ============================================================================
    // Stereo Convolution Tests
    // ============================================================================

    #[test]
    fn test_stereo_convolution_basic() {
        let ir_left = vec![1.0, 0.5, 0.25];
        let ir_right = vec![0.8, 0.4, 0.2];
        let block_size = 4;

        let mut stereo = StereoConvolution::new(ir_left, ir_right, block_size).unwrap();

        let input_left = vec![1.0, 0.0, 0.0, 0.0];
        let input_right = vec![0.5, 0.0, 0.0, 0.0];

        let (out_left, out_right) = stereo.process_block(&input_left, &input_right).unwrap();

        // Left channel should start with 1.0 * 1.0 = 1.0
        assert_abs_diff_eq!(out_left[0], 1.0, epsilon = 1e-4);
        // Right channel should start with 0.5 * 0.8 = 0.4
        assert_abs_diff_eq!(out_right[0], 0.4, epsilon = 1e-4);
    }

    #[test]
    fn test_stereo_convolution_from_mono() {
        let ir = vec![1.0, 0.5];
        let block_size = 4;

        let mut stereo = StereoConvolution::from_mono(ir, block_size).unwrap();

        let input_left = vec![1.0, 0.0, 0.0, 0.0];
        let input_right = vec![2.0, 0.0, 0.0, 0.0];

        let (out_left, out_right) = stereo.process_block(&input_left, &input_right).unwrap();

        // Both channels use same IR, so ratio should be preserved
        assert_abs_diff_eq!(out_right[0] / out_left[0], 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_stereo_convolution_interleaved() {
        let ir_left = vec![1.0, 0.5];
        let ir_right = vec![0.8, 0.4];
        let block_size = 4;

        let mut stereo = StereoConvolution::new(ir_left, ir_right, block_size).unwrap();

        // Interleaved: L, R, L, R, L, R, L, R
        let interleaved = vec![1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let output = stereo.process_interleaved(&interleaved).unwrap();

        // Check that output is properly interleaved
        assert_eq!(output.len(), 8);
        // Left output at index 0
        assert_abs_diff_eq!(output[0], 1.0, epsilon = 1e-4);
        // Right output at index 1
        assert_abs_diff_eq!(output[1], 0.4, epsilon = 1e-4);
    }

    #[test]
    fn test_stereo_convolution_reset() {
        let ir = vec![1.0, 0.5];
        let block_size = 4;

        let mut stereo = StereoConvolution::from_mono(ir, block_size).unwrap();

        let input = vec![1.0, 0.0, 0.0, 0.0];
        let _ = stereo.process_block(&input, &input).unwrap();

        stereo.reset();

        let (out_left, _) = stereo.process_block(&input, &input).unwrap();
        assert_abs_diff_eq!(out_left[0], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_true_stereo_convolution_basic() {
        // Create simple IRs for testing
        let ir_ll = vec![1.0, 0.0];
        let ir_lr = vec![0.3, 0.0]; // Cross-feed L->R
        let ir_rl = vec![0.3, 0.0]; // Cross-feed R->L
        let ir_rr = vec![1.0, 0.0];
        let block_size = 4;

        let mut true_stereo =
            TrueStereoConvolution::new(ir_ll, ir_lr, ir_rl, ir_rr, block_size).unwrap();

        let input_left = vec![1.0, 0.0, 0.0, 0.0];
        let input_right = vec![0.0, 0.0, 0.0, 0.0];

        let (out_left, out_right) = true_stereo
            .process_block(&input_left, &input_right)
            .unwrap();

        // Left output = LL (1.0) + RL (0.0) = 1.0
        assert_abs_diff_eq!(out_left[0], 1.0, epsilon = 1e-4);
        // Right output = LR (0.3) + RR (0.0) = 0.3 (cross-feed from left)
        assert_abs_diff_eq!(out_right[0], 0.3, epsilon = 1e-4);
    }

    #[test]
    fn test_true_stereo_from_stereo_with_crossfeed() {
        let ir_left = vec![1.0, 0.5, 0.25];
        let ir_right = vec![1.0, 0.5, 0.25];
        let block_size = 4;

        let mut true_stereo = TrueStereoConvolution::from_stereo_with_crossfeed(
            ir_left, ir_right, block_size, 0.3, // 30% cross-feed
            2,   // 2 sample delay
        )
        .unwrap();

        let input_left = vec![1.0, 0.0, 0.0, 0.0];
        let input_right = vec![0.0, 0.0, 0.0, 0.0];

        let (out_left, out_right) = true_stereo
            .process_block(&input_left, &input_right)
            .unwrap();

        // Left should have direct signal
        assert!(out_left[0] > 0.9);
        // Right should have delayed cross-feed (appears at sample 2)
        assert!(out_right[0] < 0.1); // No immediate cross-feed
        assert!(out_right[2] > 0.2); // Delayed cross-feed
    }

    #[test]
    fn test_true_stereo_from_mono_with_spread() {
        let ir = vec![1.0, 0.5, 0.25, 0.125];
        let block_size = 32; // Larger block to capture delayed cross-feed

        let mut true_stereo = TrueStereoConvolution::from_mono_with_spread(
            ir, block_size, 0.5, // 50% spread
        )
        .unwrap();

        // Use longer input to allow cross-feed delay to manifest
        let mut input_left = vec![0.0; block_size];
        input_left[0] = 1.0;
        let input_right = vec![0.0; block_size];

        let (out_left, out_right) = true_stereo
            .process_block(&input_left, &input_right)
            .unwrap();

        // Left should have strong signal
        assert!(out_left[0] > 0.5);
        // Right should have some cross-feed (weaker than left)
        let right_energy: f32 = out_right.iter().map(|x| x * x).sum();
        let left_energy: f32 = out_left.iter().map(|x| x * x).sum();
        assert!(
            right_energy > 0.0,
            "Right channel should have cross-feed energy, got {}",
            right_energy
        );
        assert!(
            right_energy < left_energy,
            "Cross-feed should be less than direct signal"
        );
    }

    #[test]
    fn test_true_stereo_reset() {
        let ir = vec![1.0, 0.5];
        let block_size = 4;

        let mut true_stereo =
            TrueStereoConvolution::from_mono_with_spread(ir, block_size, 0.5).unwrap();

        let input = vec![1.0, 0.0, 0.0, 0.0];
        let _ = true_stereo.process_block(&input, &input).unwrap();

        true_stereo.reset();

        let (out_left, _) = true_stereo.process_block(&input, &vec![0.0; 4]).unwrap();
        assert!(out_left[0] > 0.9);
    }

    #[test]
    fn test_stereo_fft_convolve() {
        let left = vec![1.0, 0.0, 0.0, 0.0];
        let right = vec![0.5, 0.0, 0.0, 0.0];
        let ir_left = vec![1.0, 0.5];
        let ir_right = vec![0.8, 0.4];

        let (out_left, out_right) =
            stereo_fft_convolve(&left, &right, &ir_left, &ir_right).unwrap();

        assert_abs_diff_eq!(out_left[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out_right[0], 0.4, epsilon = 1e-6);
    }

    #[test]
    fn test_true_stereo_fft_convolve() {
        let left = vec![1.0, 0.0, 0.0, 0.0];
        let right = vec![0.0, 0.0, 0.0, 0.0];
        let ir_ll = vec![1.0];
        let ir_lr = vec![0.3];
        let ir_rl = vec![0.2];
        let ir_rr = vec![1.0];

        let (out_left, out_right) =
            true_stereo_fft_convolve(&left, &right, &ir_ll, &ir_lr, &ir_rl, &ir_rr).unwrap();

        // Left = LL + RL = 1.0 + 0.0 = 1.0
        assert_abs_diff_eq!(out_left[0], 1.0, epsilon = 1e-6);
        // Right = LR + RR = 0.3 + 0.0 = 0.3
        assert_abs_diff_eq!(out_right[0], 0.3, epsilon = 1e-6);
    }

    #[test]
    fn test_create_delayed_ir() {
        let ir = vec![1.0, 0.5, 0.25];
        let delayed = create_delayed_ir(&ir, 3, 0.5);

        assert_eq!(delayed.len(), 6);
        assert_abs_diff_eq!(delayed[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(delayed[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(delayed[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(delayed[3], 0.5, epsilon = 1e-6); // 1.0 * 0.5
        assert_abs_diff_eq!(delayed[4], 0.25, epsilon = 1e-6); // 0.5 * 0.5
        assert_abs_diff_eq!(delayed[5], 0.125, epsilon = 1e-6); // 0.25 * 0.5
    }
}
