//! Utility DSP functions for audio processing
//!
//! This module provides essential DSP utilities including normalization
//! and windowing functions for hall measurement applications.

use crate::{AudioBuffer, AudioError, AudioResult, Sample};

#[cfg(not(feature = "std"))]
use alloc::vec;

/// Find the next power of two greater than or equal to n
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut power = 1;
    while power < n {
        power *= 2;
    }
    power
}

#[cfg(not(feature = "std"))]
use libm::{cosf, sqrtf};

use core::f32::consts::PI;

/// Window function types for general-purpose windowing
///
/// **Note**: For IR extraction deconvolution, use [`deconvolve::WindowType`](crate::deconvolve::WindowType)
/// which has the same options but is specifically designed for deconvolution post-processing.
///
/// # Window Characteristics
///
/// - **Rectangular**: No windowing (all 1.0)
/// - **Hann**: Good frequency resolution, moderate sidelobe suppression
/// - **Hamming**: Similar to Hann but with slightly better sidelobe suppression
/// - **Blackman**: Best sidelobe suppression, wider main lobe
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowFunction {
    /// Rectangular (no windowing)
    Rectangular,
    /// Hann window
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
}

/// Normalization methods for audio signals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationMethod {
    /// Peak normalization to specified level.
    Peak {
        /// Target peak level (typically 0.0 to 1.0).
        target_level: Sample,
    },
    /// RMS normalization to specified level.
    Rms {
        /// Target RMS level (typically 0.0 to 1.0).
        target_level: Sample,
    },
}

/// Apply a window function to an audio buffer
pub fn apply_window(buffer: &mut [Sample], window: WindowFunction) {
    let len = buffer.len();
    if len <= 1 {
        return;
    }

    for (i, sample) in buffer.iter_mut().enumerate() {
        let window_value = calculate_window_value(window, i, len);
        *sample *= window_value;
    }
}

/// Generate a window function
pub fn generate_window(length: usize, window: WindowFunction) -> AudioBuffer {
    let mut buffer = vec![1.0; length];
    apply_window(&mut buffer, window);
    buffer
}

fn calculate_window_value(window: WindowFunction, index: usize, length: usize) -> Sample {
    if length <= 1 {
        return 1.0;
    }

    let n = index as Sample;
    let n_max = (length - 1) as Sample;

    match window {
        WindowFunction::Rectangular => 1.0,

        WindowFunction::Hann => {
            #[cfg(feature = "std")]
            let val = 0.5 * (1.0 - ((2.0 * PI * n) / n_max).cos());
            #[cfg(not(feature = "std"))]
            let val = 0.5 * (1.0 - cosf((2.0 * PI * n) / n_max));
            val
        }

        WindowFunction::Hamming => {
            #[cfg(feature = "std")]
            let val = 0.54 - 0.46 * ((2.0 * PI * n) / n_max).cos();
            #[cfg(not(feature = "std"))]
            let val = 0.54 - 0.46 * cosf((2.0 * PI * n) / n_max);
            val
        }

        WindowFunction::Blackman => {
            #[cfg(feature = "std")]
            let val =
                0.42 - 0.5 * ((2.0 * PI * n) / n_max).cos() + 0.08 * ((4.0 * PI * n) / n_max).cos();
            #[cfg(not(feature = "std"))]
            let val =
                0.42 - 0.5 * cosf((2.0 * PI * n) / n_max) + 0.08 * cosf((4.0 * PI * n) / n_max);
            val
        }
    }
}

/// Normalize an audio buffer using the specified method
pub fn normalize(buffer: &mut [Sample], method: NormalizationMethod) -> AudioResult<()> {
    if buffer.is_empty() {
        return Err(AudioError::InsufficientData);
    }

    match method {
        NormalizationMethod::Peak { target_level } => {
            let peak = buffer.iter().map(|&x| x.abs()).fold(0.0, f32::max);
            if peak > 0.0 {
                let scale = target_level / peak;
                for sample in buffer {
                    *sample *= scale;
                }
            }
        }

        NormalizationMethod::Rms { target_level } => {
            let rms = calculate_rms(buffer);
            if rms > 0.0 {
                let scale = target_level / rms;
                for sample in buffer {
                    *sample *= scale;
                }
            }
        }
    }

    Ok(())
}

/// Calculate RMS (Root Mean Square) of a buffer
pub fn calculate_rms(buffer: &[Sample]) -> Sample {
    if buffer.is_empty() {
        return 0.0;
    }

    let sum_squares: Sample = buffer.iter().map(|&x| x * x).sum();

    #[cfg(feature = "std")]
    let rms = (sum_squares / (buffer.len() as Sample)).sqrt();
    #[cfg(not(feature = "std"))]
    let rms = sqrtf(sum_squares / (buffer.len() as Sample));

    rms
}

/// Calculate peak amplitude of a buffer
pub fn calculate_peak(buffer: &[Sample]) -> Sample {
    buffer.iter().map(|&x| x.abs()).fold(0.0, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[cfg(not(feature = "std"))]
    use alloc::{format, vec, vec::Vec};

    #[test]
    fn test_calculate_window_value_length_one() {
        // Should always return 1.0 for any window type if length <= 1
        for &window in &[
            WindowFunction::Rectangular,
            WindowFunction::Hann,
            WindowFunction::Hamming,
            WindowFunction::Blackman,
        ] {
            assert_abs_diff_eq!(calculate_window_value(window, 0, 1), 1.0, epsilon = 1e-6);
            assert_abs_diff_eq!(calculate_window_value(window, 0, 0), 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_window_functions() {
        let length = 10;
        let hann = generate_window(length, WindowFunction::Hann);
        let hamming = generate_window(length, WindowFunction::Hamming);
        let blackman = generate_window(length, WindowFunction::Blackman);
        let rect = generate_window(length, WindowFunction::Rectangular);

        // Windows should taper to near zero at edges (except rectangular)
        assert!(hann[0] < 0.1);
        assert!(hann[length - 1] < 0.1);
        assert!(hamming[0] < 0.1);
        assert!(hamming[length - 1] < 0.1);
        assert!(blackman[0] < 0.1);

        // Rectangular window should be all 1.0
        assert_abs_diff_eq!(rect[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(rect[length - 1], 1.0, epsilon = 1e-6);

        // Peak should be in the middle
        assert!(hann[length / 2] > 0.9);
        assert!(hamming[length / 2] > 0.9);
    }

    #[test]
    fn test_apply_window() {
        let mut buffer = vec![1.0; 10];
        apply_window(&mut buffer, WindowFunction::Hann);

        // Edges should be attenuated
        assert!(buffer[0] < 0.1);
        assert!(buffer[9] < 0.1);

        // Middle should be close to 1.0
        assert!(buffer[5] > 0.9);
    }

    #[test]
    fn test_apply_window_empty() {
        let mut buffer: Vec<f32> = vec![];
        apply_window(&mut buffer, WindowFunction::Hann);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_apply_window_single_sample() {
        let mut buffer = vec![1.0];
        apply_window(&mut buffer, WindowFunction::Hann);
        assert_eq!(buffer, vec![1.0]);
    }

    #[test]
    fn test_normalization() {
        let mut buffer = vec![0.5, -1.0, 0.25, 0.75];

        // Peak normalization
        normalize(&mut buffer, NormalizationMethod::Peak { target_level: 0.5 }).unwrap();
        let peak = calculate_peak(&buffer);
        assert_abs_diff_eq!(peak, 0.5, epsilon = 1e-6);

        // RMS normalization
        let mut buffer2 = vec![0.5, -1.0, 0.25, 0.75];
        normalize(&mut buffer2, NormalizationMethod::Rms { target_level: 0.5 }).unwrap();
        let rms = calculate_rms(&buffer2);
        assert_abs_diff_eq!(rms, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_normalize_zero_buffer() {
        let mut buffer = vec![0.0, 0.0, 0.0];
        // Normalizing a zero buffer succeeds but leaves it unchanged
        let result = normalize(&mut buffer, NormalizationMethod::Peak { target_level: 1.0 });
        assert_eq!(result, Ok(()));
        assert_eq!(buffer, vec![0.0, 0.0, 0.0]);

        let mut buffer2 = vec![0.0, 0.0, 0.0];
        let result2 = normalize(&mut buffer2, NormalizationMethod::Rms { target_level: 1.0 });
        assert_eq!(result2, Ok(()));
        assert_eq!(buffer2, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_empty() {
        let mut buffer: Vec<f32> = vec![];
        let result = normalize(&mut buffer, NormalizationMethod::Peak { target_level: 1.0 });
        assert_eq!(result, Err(AudioError::InsufficientData));
    }

    #[test]
    fn test_rms_calculation() {
        let buffer = vec![1.0, -1.0, 1.0, -1.0];
        let rms = calculate_rms(&buffer);
        assert_abs_diff_eq!(rms, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_rms_empty() {
        let buffer: Vec<f32> = vec![];
        let rms = calculate_rms(&buffer);
        assert_eq!(rms, 0.0);
    }

    #[test]
    fn test_calculate_peak() {
        let buffer = vec![0.5, -1.5, 0.25, 0.75];
        let peak = calculate_peak(&buffer);
        assert_abs_diff_eq!(peak, 1.5, epsilon = 1e-6);
    }

    #[test]
    fn test_calculate_peak_empty() {
        let buffer: Vec<f32> = vec![];
        let peak = calculate_peak(&buffer);
        assert_eq!(peak, 0.0);
    }

    #[test]
    fn test_next_power_of_two_edge_cases() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(1025), 2048);
    }

    #[test]
    fn test_window_function_clone() {
        let w1 = WindowFunction::Hann;
        let w2 = w1.clone();
        assert_eq!(w1, w2);
    }

    #[test]
    fn test_normalization_method_clone() {
        let n1 = NormalizationMethod::Peak { target_level: 1.0 };
        let n2 = n1.clone();
        assert_eq!(format!("{:?}", n1), format!("{:?}", n2));
    }
}
