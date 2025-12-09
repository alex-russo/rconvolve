#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![doc = include_str!("../README.md")]

pub mod convolve;
pub mod deconvolve;
pub mod sweep;
pub mod utils;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use convolve::*;
pub use deconvolve::*;
pub use sweep::*;
pub use utils::*;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Audio sample type (32-bit float).
pub type Sample = f32;

/// Buffer of audio samples.
pub type AudioBuffer = Vec<Sample>;

/// Audio processing errors.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioError {
    /// Sample rate must be positive and finite.
    InvalidSampleRate,
    /// Duration must be positive and finite.
    InvalidDuration,
    /// Frequency range is invalid (e.g., start >= end, or out of Nyquist bounds).
    InvalidFrequencyRange,
    /// Input buffer sizes do not match expected dimensions.
    BufferSizeMismatch,
    /// Not enough data provided for the requested operation.
    InsufficientData,
    /// An error occurred during FFT processing.
    FftError,
}

impl core::fmt::Display for AudioError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AudioError::InvalidSampleRate => write!(f, "Invalid sample rate"),
            AudioError::InvalidDuration => write!(f, "Invalid duration"),
            AudioError::InvalidFrequencyRange => write!(f, "Invalid frequency range"),
            AudioError::BufferSizeMismatch => write!(f, "Buffer size mismatch"),
            AudioError::InsufficientData => write!(f, "Insufficient data"),
            AudioError::FftError => write!(f, "FFT processing error"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AudioError {}

/// Result type for audio processing operations
pub type AudioResult<T> = Result<T, AudioError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn test_audio_error_display() {
        assert_eq!(
            AudioError::InvalidSampleRate.to_string(),
            "Invalid sample rate"
        );
        assert_eq!(AudioError::InvalidDuration.to_string(), "Invalid duration");
        assert_eq!(
            AudioError::InvalidFrequencyRange.to_string(),
            "Invalid frequency range"
        );
        assert_eq!(
            AudioError::BufferSizeMismatch.to_string(),
            "Buffer size mismatch"
        );
        assert_eq!(
            AudioError::InsufficientData.to_string(),
            "Insufficient data"
        );
        assert_eq!(AudioError::FftError.to_string(), "FFT processing error");
    }

    #[test]
    fn test_audio_error_clone() {
        let err = AudioError::InvalidSampleRate;
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }
}
