// rconvolve Hall Measurement App - Audio Utilities
// Helper functions for audio analysis and playback

import { getOrCreateAudioContext, SWEEP_CONFIG } from "./state.js";

// ============================================================================
// Audio Analysis Functions
// ============================================================================

/**
 * Calculate the peak amplitude of audio data
 * @param {Float32Array} audioData - Audio samples
 * @returns {number} Peak amplitude (0-1)
 */
export function calculatePeak(audioData) {
  let peak = 0;
  for (let i = 0; i < audioData.length; i++) {
    const abs = Math.abs(audioData[i]);
    if (abs > peak) peak = abs;
  }
  return peak;
}

/**
 * Calculate the RMS (Root Mean Square) level of audio data
 * @param {Float32Array} audioData - Audio samples
 * @returns {number} RMS level
 */
export function calculateRMS(audioData) {
  let sum = 0;
  for (let i = 0; i < audioData.length; i++) {
    sum += audioData[i] * audioData[i];
  }
  return Math.sqrt(sum / audioData.length);
}

/**
 * Calculate the high-frequency noise ratio in an impulse response
 * This measures the ratio of minimum to maximum energy in the IR tail
 * @param {Float32Array} audioData - Impulse response samples
 * @returns {number} Noise ratio (0-1, lower is better)
 */
export function calculateHighFreqNoise(audioData) {
  // For impulse responses, we need to measure actual noise, not high-frequency content
  // We'll look at the tail of the IR where there should only be decay/noise

  if (audioData.length < 1000) return 0; // Too short to analyze

  // Skip the first 10% (initial transient) and last 10% (potential artifacts)
  const startIdx = Math.floor(audioData.length * 0.1);
  const endIdx = Math.floor(audioData.length * 0.9);
  const analyzeLength = endIdx - startIdx;

  if (analyzeLength < 100) return 0;

  // Calculate the envelope decay to separate signal from noise
  const windowSize = Math.floor(analyzeLength / 20); // 5% windows
  let maxEnergy = 0;
  let minEnergy = Infinity;

  for (let i = startIdx; i < endIdx - windowSize; i += windowSize) {
    let energy = 0;
    for (let j = i; j < i + windowSize; j++) {
      energy += audioData[j] * audioData[j];
    }
    energy = Math.sqrt(energy / windowSize);
    maxEnergy = Math.max(maxEnergy, energy);
    minEnergy = Math.min(minEnergy, energy);
  }

  // Noise ratio is the ratio of minimum to maximum energy in the tail
  // Good IRs should have smooth decay, bad ones will have noise floors
  return maxEnergy > 0 ? minEnergy / maxEnergy : 0;
}

// ============================================================================
// Audio Playback Functions
// ============================================================================

/**
 * Play audio data through the audio context
 * @param {Float32Array} audioData - Audio samples to play
 * @param {number} volume - Volume level (0-1)
 * @param {number} sampleRate - Optional sample rate override
 */
export async function playAudio(audioData, volume = 0.3, sampleRate = null) {
  try {
    const ctx = await getOrCreateAudioContext();

    // Use provided sample rate or fall back to SWEEP_CONFIG
    const rate = sampleRate || SWEEP_CONFIG.sampleRate;
    const buffer = ctx.createBuffer(1, audioData.length, rate);
    buffer.copyToChannel(audioData, 0);

    const source = ctx.createBufferSource();
    const gain = ctx.createGain();
    gain.gain.value = volume;

    source.buffer = buffer;
    source.connect(gain);
    gain.connect(ctx.destination);
    source.start();

    console.log(`🎵 Playing ${audioData.length} samples`);
  } catch (error) {
    console.error("Playback error:", error);
  }
}
