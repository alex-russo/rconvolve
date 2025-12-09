//! Exponential sine sweep generation for hall measurement
//!
//! This module provides simplified exponential sine sweep generation
//! for acoustic measurement and impulse response extraction.

use crate::{AudioBuffer, AudioError, AudioResult, Sample};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use libm::{expf, logf, sinf};

use core::f32::consts::PI;

/// Sweep configuration for hall measurement
#[derive(Debug, Clone)]
pub struct SweepConfig {
    /// Sample rate in Hz
    pub sample_rate: Sample,
    /// Duration in seconds
    pub duration: Sample,
    /// Starting frequency in Hz
    pub start_freq: Sample,
    /// Ending frequency in Hz
    pub end_freq: Sample,
}

impl SweepConfig {
    /// Create a new sweep configuration
    pub fn new(
        sample_rate: Sample,
        duration: Sample,
        start_freq: Sample,
        end_freq: Sample,
    ) -> Self {
        Self {
            sample_rate,
            duration,
            start_freq,
            end_freq,
        }
    }

    /// Validate the configuration
    fn validate(&self) -> AudioResult<()> {
        if self.sample_rate <= 0.0 {
            return Err(AudioError::InvalidSampleRate);
        }
        if self.duration <= 0.0 {
            return Err(AudioError::InvalidDuration);
        }
        if self.start_freq <= 0.0 || self.end_freq <= 0.0 || self.start_freq >= self.end_freq {
            return Err(AudioError::InvalidFrequencyRange);
        }
        if self.end_freq > self.sample_rate / 2.0 {
            return Err(AudioError::InvalidFrequencyRange);
        }
        Ok(())
    }
}

/// Generate an exponential sine sweep from configuration
pub fn generate(config: &SweepConfig) -> AudioResult<AudioBuffer> {
    config.validate()?;
    exponential(
        config.sample_rate,
        config.duration,
        config.start_freq,
        config.end_freq,
    )
}

/// Generate an exponential sine sweep
///
/// Exponential sweeps provide equal energy per octave, making them ideal
/// for room acoustic measurements and impulse response extraction.
///
/// # Arguments
/// * `sample_rate` - Sample rate in Hz (e.g., 48000.0)
/// * `duration` - Duration in seconds (e.g., 2.0)
/// * `start_freq` - Starting frequency in Hz (e.g., 20.0)
/// * `end_freq` - Ending frequency in Hz (e.g., 20000.0)
///
/// # Example
/// ```
/// use rconvolve::sweep;
/// let sweep = sweep::exponential(48000.0, 2.0, 20.0, 20000.0).unwrap();
/// ```
pub fn exponential(
    sample_rate: Sample,
    duration: Sample,
    start_freq: Sample,
    end_freq: Sample,
) -> AudioResult<AudioBuffer> {
    // Validate parameters
    if sample_rate <= 0.0 {
        return Err(AudioError::InvalidSampleRate);
    }
    if duration <= 0.0 {
        return Err(AudioError::InvalidDuration);
    }
    if start_freq <= 0.0 || end_freq <= 0.0 || start_freq >= end_freq {
        return Err(AudioError::InvalidFrequencyRange);
    }
    if end_freq > sample_rate / 2.0 {
        return Err(AudioError::InvalidFrequencyRange);
    }

    let num_samples = (sample_rate * duration) as usize;
    let mut buffer = Vec::with_capacity(num_samples);

    #[cfg(feature = "std")]
    let log_ratio = (end_freq / start_freq).ln();
    #[cfg(not(feature = "std"))]
    let log_ratio = logf(end_freq / start_freq);

    let k = log_ratio / duration;

    for n in 0..num_samples {
        let t = (n as Sample) / sample_rate;

        #[cfg(feature = "std")]
        let phase = (2.0 * PI * start_freq * ((k * t).exp() - 1.0)) / k;
        #[cfg(not(feature = "std"))]
        let phase = (2.0 * PI * start_freq * (expf(k * t) - 1.0)) / k;

        #[cfg(feature = "std")]
        let sample = phase.sin();
        #[cfg(not(feature = "std"))]
        let sample = sinf(phase);

        buffer.push(sample);
    }

    Ok(buffer)
}

/// Generate an inverse filter for deconvolution
///
/// This creates a time-reversed version of the exponential sweep with
/// spectral compensation, used for deconvolving recorded sweeps to
/// extract impulse responses.
///
/// # Arguments
/// * `config` - Sweep configuration
///
/// # Example
/// ```
/// use rconvolve::sweep::{SweepConfig, inverse_filter};
/// let config = SweepConfig::new(48000.0, 2.0, 20.0, 20000.0);
/// let inverse = inverse_filter(&config).unwrap();
/// ```
pub fn inverse_filter(config: &SweepConfig) -> AudioResult<AudioBuffer> {
    // Generate the original sweep
    let mut sweep = generate(config)?;

    // Time reverse the sweep
    sweep.reverse();

    // Apply spectral compensation for exponential sweeps
    apply_spectral_compensation(&mut sweep, config);

    Ok(sweep)
}

fn apply_spectral_compensation(buffer: &mut [Sample], config: &SweepConfig) {
    #[cfg(feature = "std")]
    let log_ratio = (config.end_freq / config.start_freq).ln();
    #[cfg(not(feature = "std"))]
    let log_ratio = logf(config.end_freq / config.start_freq);

    let k = log_ratio / config.duration;

    for (n, sample) in buffer.iter_mut().enumerate() {
        let t = (n as Sample) / config.sample_rate;

        // Apply -6dB/octave compensation for exponential sweeps
        #[cfg(feature = "std")]
        let compensation = (k * t).exp().sqrt();
        #[cfg(not(feature = "std"))]
        let compensation = expf(k * t * 0.5);

        *sample *= compensation;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_sweep_generation() {
        let sweep = exponential(1000.0, 1.0, 100.0, 500.0).unwrap();

        assert_eq!(sweep.len(), 1000);
        assert!(sweep.iter().any(|&x| x.abs() > 0.1));

        // Check that we have some variation (not all zeros)
        let max_val = sweep.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        assert!(max_val > 0.5);
    }

    #[test]
    fn test_sweep_validation() {
        // Test invalid sample rate
        assert!(exponential(-1.0, 1.0, 100.0, 500.0).is_err());
        assert!(exponential(0.0, 1.0, 100.0, 500.0).is_err());

        // Test invalid duration
        assert!(exponential(1000.0, 0.0, 100.0, 500.0).is_err());
        assert!(exponential(1000.0, -1.0, 100.0, 500.0).is_err());

        // Test invalid frequency range
        assert!(exponential(1000.0, 1.0, 500.0, 100.0).is_err());
        assert!(exponential(1000.0, 1.0, 0.0, 100.0).is_err());
        assert!(exponential(1000.0, 1.0, -10.0, 100.0).is_err());

        // Test frequency above Nyquist
        assert!(exponential(1000.0, 1.0, 100.0, 600.0).is_err());

        // Test valid configuration
        assert!(exponential(1000.0, 1.0, 100.0, 400.0).is_ok());
    }

    #[test]
    fn test_inverse_filter_generation() {
        let config = SweepConfig::new(1000.0, 1.0, 100.0, 500.0);
        let inverse = inverse_filter(&config).unwrap();

        assert_eq!(inverse.len(), 1000);

        // Check that we have some variation (not all zeros)
        let max_val = inverse.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        assert!(max_val > 0.1);
    }

    #[test]
    fn test_inverse_filter_invalid() {
        let config = SweepConfig::new(-1.0, 1.0, 100.0, 500.0);
        assert_eq!(inverse_filter(&config), Err(AudioError::InvalidSampleRate));
    }

    #[test]
    fn test_sweep_config() {
        let config = SweepConfig::new(48000.0, 2.0, 20.0, 20000.0);
        let sweep = generate(&config).unwrap();

        assert_eq!(sweep.len(), 96000); // 2 seconds at 48kHz

        // Check amplitude is reasonable
        let max_val = sweep.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        assert!(max_val > 0.8 && max_val <= 1.0);
    }

    #[test]
    fn test_sweep_config_clone() {
        let config1 = SweepConfig::new(48000.0, 2.0, 20.0, 20000.0);
        let config2 = config1.clone();
        assert_eq!(config1.sample_rate, config2.sample_rate);
        assert_eq!(config1.duration, config2.duration);
    }

    #[test]
    fn test_generate_invalid() {
        let config = SweepConfig::new(0.0, 1.0, 100.0, 500.0);
        assert_eq!(generate(&config), Err(AudioError::InvalidSampleRate));
    }
}
