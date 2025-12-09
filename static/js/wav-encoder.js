// rconvolve Hall Measurement App - WAV Encoder
// Functions for encoding audio data to WAV format

import { calculatePeak } from "./audio-utils.js";

/**
 * Helper function to write a string to a DataView
 * @param {DataView} view - DataView to write to
 * @param {number} offset - Byte offset
 * @param {string} string - String to write
 */
function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

/**
 * Encode mono audio data to WAV format
 * @param {Float32Array} audioData - Audio samples
 * @param {number} sampleRate - Sample rate in Hz
 * @returns {ArrayBuffer} WAV file data
 */
export function encodeWAV(audioData, sampleRate) {
  const length = audioData.length;
  const arrayBuffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(arrayBuffer);

  // Find peak for normalization
  const peak = calculatePeak(audioData);
  const normalizationGain = peak > 0 ? 0.8 / peak : 1.0; // Normalize to 80% of full scale

  // RIFF chunk descriptor
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + length * 2, true); // ChunkSize
  writeString(view, 8, "WAVE");

  // fmt sub-chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
  view.setUint16(22, 1, true); // NumChannels (1 for mono)
  view.setUint32(24, sampleRate, true); // SampleRate
  view.setUint32(28, sampleRate * 2, true); // ByteRate
  view.setUint16(32, 2, true); // BlockAlign
  view.setUint16(34, 16, true); // BitsPerSample

  // data sub-chunk
  writeString(view, 36, "data");
  view.setUint32(40, length * 2, true); // Subchunk2Size

  // Convert float samples to 16-bit PCM with normalization
  let offset = 44;
  for (let i = 0; i < length; i++) {
    // Normalize the sample first
    const normalizedSample = audioData[i] * normalizationGain;
    const clampedSample = Math.max(-1, Math.min(1, normalizedSample)); // Clamp to [-1, 1]
    const pcmSample =
      clampedSample < 0 ? clampedSample * 0x8000 : clampedSample * 0x7fff;
    view.setInt16(offset, Math.round(pcmSample), true);
    offset += 2;
  }

  return arrayBuffer;
}

/**
 * Encode stereo audio data to WAV format
 * @param {Float32Array} leftData - Left channel samples
 * @param {Float32Array} rightData - Right channel samples
 * @param {number} sampleRate - Sample rate in Hz
 * @returns {ArrayBuffer} WAV file data
 */
export function encodeWAVStereo(leftData, rightData, sampleRate) {
  const length = leftData.length;
  const dataSize = length * 4; // 2 channels * 2 bytes per sample
  const arrayBuffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(arrayBuffer);

  // Find peak across both channels for normalization
  const peakLeft = calculatePeak(leftData);
  const peakRight = calculatePeak(rightData);
  const peak = Math.max(peakLeft, peakRight);
  const normalizationGain = peak > 0 ? 0.8 / peak : 1.0;

  // RIFF chunk descriptor
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true); // ChunkSize
  writeString(view, 8, "WAVE");

  // fmt sub-chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
  view.setUint16(22, 2, true); // NumChannels (2 for stereo)
  view.setUint32(24, sampleRate, true); // SampleRate
  view.setUint32(28, sampleRate * 4, true); // ByteRate (sampleRate * numChannels * bytesPerSample)
  view.setUint16(32, 4, true); // BlockAlign (numChannels * bytesPerSample)
  view.setUint16(34, 16, true); // BitsPerSample

  // data sub-chunk
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true); // Subchunk2Size

  // Interleave and convert float samples to 16-bit PCM
  let offset = 44;
  for (let i = 0; i < length; i++) {
    // Left channel
    const normalizedLeft = leftData[i] * normalizationGain;
    const clampedLeft = Math.max(-1, Math.min(1, normalizedLeft));
    const pcmLeft =
      clampedLeft < 0 ? clampedLeft * 0x8000 : clampedLeft * 0x7fff;
    view.setInt16(offset, Math.round(pcmLeft), true);
    offset += 2;

    // Right channel
    const normalizedRight = rightData[i] * normalizationGain;
    const clampedRight = Math.max(-1, Math.min(1, normalizedRight));
    const pcmRight =
      clampedRight < 0 ? clampedRight * 0x8000 : clampedRight * 0x7fff;
    view.setInt16(offset, Math.round(pcmRight), true);
    offset += 2;
  }

  return arrayBuffer;
}
