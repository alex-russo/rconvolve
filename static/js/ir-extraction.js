// rconvolve Hall Measurement App - IR Extraction
// Impulse response extraction from recorded sweep responses

import {
  sweepSignal,
  irProcessor,
  recordedResponse,
  recordedResponseLeft,
  recordedResponseRight,
  SWEEP_CONFIG,
  WasmSweepGenerator,
  WasmIRProcessor,
  setSweepSignal,
  setIRProcessor,
  setExtractedIR,
  setActiveIR,
  updateSweepConfig,
} from "./state.js";
import {
  calculatePeak,
  calculateRMS,
  calculateHighFreqNoise,
  playAudio,
} from "./audio-utils.js";
import { encodeWAV, encodeWAVStereo } from "./wav-encoder.js";
import { showOutput, visualizeAudio } from "./visualization.js";
import { getCurrentSweepSettings } from "./sweep.js";

/**
 * Extract impulse response from the recorded sweep response
 */
export function extractIR() {
  try {
    if (!recordedResponse) {
      throw new Error("Load a recorded file first!");
    }

    const irLengthSecs = parseFloat(document.getElementById("irLength").value);
    const regularization = parseFloat(
      document.getElementById("regularization").value
    );

    // Get current sweep settings
    const currentSettings = getCurrentSweepSettings();

    // Generate reference sweep if needed or if settings changed
    if (
      !sweepSignal ||
      SWEEP_CONFIG.sampleRate !== currentSettings.sampleRate ||
      SWEEP_CONFIG.duration !== currentSettings.duration ||
      SWEEP_CONFIG.startFreq !== currentSettings.startFreq ||
      SWEEP_CONFIG.endFreq !== currentSettings.endFreq
    ) {
      // Update global config
      updateSweepConfig(currentSettings);

      // Generate new reference sweep
      const sweep = WasmSweepGenerator.exponential(
        currentSettings.sampleRate,
        currentSettings.duration,
        currentSettings.startFreq,
        currentSettings.endFreq
      );
      setSweepSignal(sweep);

      // Create new IR processor
      setIRProcessor(new WasmIRProcessor(sweep));
    }

    // Check if we have stereo data available
    const hasStereoData =
      recordedResponseLeft &&
      recordedResponseRight &&
      recordedResponseLeft.length > 0 &&
      recordedResponseRight.length > 0 &&
      recordedResponseLeft !== recordedResponseRight;

    showOutput(
      "irOutput",
      `🔍 Extracting impulse response...\n` +
        `   Using sweep: ${currentSettings.duration}s, ${currentSettings.startFreq}-${currentSettings.endFreq} Hz @ ${currentSettings.sampleRate} Hz\n` +
        `   Reference sweep: ${sweepSignal.length} samples\n` +
        `   Recorded response: ${recordedResponse.length} samples\n` +
        `   Stereo extraction: ${
          hasStereoData ? "Yes (L+R channels)" : "No (mono)"
        }\n` +
        `   Duration mismatch: ${Math.abs(
          sweepSignal.length - recordedResponse.length
        )} samples\n` +
        `   Requested IR length: ${Math.floor(
          irLengthSecs * currentSettings.sampleRate
        )} samples (${irLengthSecs}s)\n` +
        `   Regularization: ${regularization}\n` +
        `   ${
          Math.abs(sweepSignal.length - recordedResponse.length) >
          currentSettings.sampleRate
            ? "⚠️  LARGE DURATION MISMATCH DETECTED!"
            : "✅ Duration match OK"
        }`,
      ""
    );

    const irLength = Math.floor(irLengthSecs * currentSettings.sampleRate);

    // Extract IR using the WASM processor (always extract mono for backwards compatibility)
    const extractedIR = irProcessor.extract_impulse_response_advanced(
      recordedResponse,
      irLength,
      regularization
    );

    // Extract stereo IRs if stereo data is available
    let extractedIR_Left, extractedIR_Right;
    if (hasStereoData) {
      extractedIR_Left = irProcessor.extract_impulse_response_advanced(
        recordedResponseLeft,
        irLength,
        regularization
      );
      extractedIR_Right = irProcessor.extract_impulse_response_advanced(
        recordedResponseRight,
        irLength,
        regularization
      );
    } else {
      // Use the mono IR for both channels
      extractedIR_Left = extractedIR;
      extractedIR_Right = extractedIR;
    }

    // Store extracted IRs
    setExtractedIR(extractedIR, extractedIR_Left, extractedIR_Right);

    // Analyze the extracted IR
    const peak = calculatePeak(extractedIR);
    const rms = calculateRMS(extractedIR);
    const dynamicRange = 20 * Math.log10(peak / rms);
    const noiseLevel = calculateHighFreqNoise(extractedIR);

    // For stereo, also analyze L/R
    let stereoInfo = "";
    if (hasStereoData) {
      const peakL = calculatePeak(extractedIR_Left);
      const peakR = calculatePeak(extractedIR_Right);
      stereoInfo =
        `\n   Left peak: ${(20 * Math.log10(peakL)).toFixed(1)} dBFS\n` +
        `   Right peak: ${(20 * Math.log10(peakR)).toFixed(1)} dBFS`;
    }

    // Quality assessment based on proper IR metrics
    let quality = "Good";
    let warnings = [];

    // Check peak amplitude (should be reasonable)
    if (peak < 0.01) {
      quality = "Poor";
      warnings.push(
        "Very low peak - try different regularization or check recording quality"
      );
    } else if (peak < 0.1) {
      quality = "Fair";
      warnings.push("Low peak amplitude - may need level adjustment");
    }

    // Check dynamic range (very important for IRs)
    if (dynamicRange < 30) {
      quality = "Poor";
      warnings.push(
        "Low dynamic range - indicates poor recording or processing"
      );
    } else if (dynamicRange < 40) {
      quality = "Fair";
      warnings.push("Moderate dynamic range - acceptable but could be better");
    }

    // Check noise level (using corrected algorithm)
    if (noiseLevel > 0.1) {
      quality = "Poor";
      warnings.push("High noise floor detected in IR tail");
    } else if (noiseLevel > 0.05) {
      if (quality === "Good") quality = "Fair";
      warnings.push("Some noise detected - try higher regularization");
    }

    showOutput(
      "irOutput",
      `✅ Impulse Response Extracted! ${
        hasStereoData ? "(True Stereo)" : "(Mono)"
      }\n` +
        `   Length: ${extractedIR.length} samples (${(
          extractedIR.length / currentSettings.sampleRate
        ).toFixed(2)}s)\n` +
        `   Peak: ${peak.toFixed(6)} (${(20 * Math.log10(peak)).toFixed(
          1
        )} dBFS)\n` +
        `   RMS: ${rms.toFixed(6)}\n` +
        `   Dynamic range: ${dynamicRange.toFixed(1)} dB (${
          dynamicRange >= 40
            ? "Excellent"
            : dynamicRange >= 30
            ? "Good"
            : "Poor"
        })\n` +
        `   Noise ratio: ${(noiseLevel * 100).toFixed(1)}% (${
          noiseLevel <= 0.05 ? "Low" : noiseLevel <= 0.1 ? "Moderate" : "High"
        })\n` +
        `   Quality: ${quality}` +
        stereoInfo +
        (warnings.length > 0
          ? `\n⚠️  ${warnings.join("\n⚠️  ")}`
          : "\n🎉 IR looks great! Ready to use in reverb plugins."),
      quality === "Good" ? "success" : quality === "Fair" ? "" : "error"
    );

    visualizeAudio(extractedIR, "irCanvas", "Extracted Impulse Response");

    // Enable preview and download
    document.getElementById("previewBtn").disabled = false;
    document.getElementById("downloadBtn").disabled = false;

    // Set as active IR for real-time convolution
    setActiveIR(extractedIR, extractedIR_Left, extractedIR_Right);

    const irDurationMs = (
      (extractedIR.length / currentSettings.sampleRate) *
      1000
    ).toFixed(0);
    const stereoLabel = hasStereoData ? " (True Stereo)" : "";
    document.getElementById(
      "currentIRInfo"
    ).textContent = `🎯 Extracted IR${stereoLabel} (${extractedIR.length} samples, ${irDurationMs}ms)`;
    document.getElementById("currentIRInfo").style.color = "#2196F3";

    // Enable real-time convolution and "Use Extracted IR" button
    document.getElementById("startConvBtn").disabled = false;
    document.getElementById("useExtractedBtn").disabled = false;
  } catch (error) {
    showOutput("irOutput", `❌ Extraction failed: ${error.message}`, "error");
  }
}

/**
 * Preview the extracted impulse response by playing it
 * @param {Float32Array} extractedIR - The extracted IR to preview
 */
export function previewIR(extractedIR) {
  if (!extractedIR) {
    showOutput("irOutput", "❌ Extract IR first!", "error");
    return;
  }

  try {
    // Create a boosted version for audible preview
    const peak = calculatePeak(extractedIR);
    const gainBoost = Math.min(10.0, 0.1 / peak); // Up to 10x boost

    const boostedIR = new Float32Array(extractedIR.length);
    for (let i = 0; i < extractedIR.length; i++) {
      boostedIR[i] = extractedIR[i] * gainBoost;
    }

    // Play the boosted IR
    playAudio(boostedIR, 0.5);

    showOutput(
      "irOutput",
      `🎧 Playing IR preview (${gainBoost.toFixed(1)}x boost)\n` +
        `   Listen for: sharp initial transient + smooth decay\n` +
        `   Good IR = click followed by natural reverb tail`,
      "success"
    );
  } catch (error) {
    showOutput("irOutput", `❌ Preview error: ${error.message}`, "error");
  }
}

/**
 * Download the extracted IR as a WAV file
 * @param {Float32Array} extractedIR - The extracted IR (mono or main channel)
 * @param {Float32Array|null} extractedIR_Left - Left channel for stereo
 * @param {Float32Array|null} extractedIR_Right - Right channel for stereo
 */
export function downloadIR(
  extractedIR,
  extractedIR_Left = null,
  extractedIR_Right = null
) {
  if (!extractedIR) {
    showOutput("irOutput", "❌ Extract IR first!", "error");
    return;
  }

  try {
    // Get current settings for sample rate
    const currentSettings = getCurrentSweepSettings();

    // Check if we have true stereo IR
    const hasTrueStereo =
      extractedIR_Left &&
      extractedIR_Right &&
      extractedIR_Left !== extractedIR_Right;

    let wavBuffer;
    let filename;
    let formatInfo;

    if (hasTrueStereo) {
      // Create stereo WAV file
      wavBuffer = encodeWAVStereo(
        extractedIR_Left,
        extractedIR_Right,
        currentSettings.sampleRate
      );
      filename = "impulse_response_stereo.wav";
      formatInfo = "Stereo, 16-bit";
    } else {
      // Create mono WAV file
      wavBuffer = encodeWAV(extractedIR, currentSettings.sampleRate);
      filename = "impulse_response.wav";
      formatInfo = "Mono, 16-bit";
    }

    const blob = new Blob([wavBuffer], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);

    // Create download link
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();

    URL.revokeObjectURL(url);

    showOutput(
      "irOutput",
      `💾 Downloaded IR as WAV file\n` +
        `   File: ${filename}\n` +
        `   Format: ${formatInfo}, ${currentSettings.sampleRate} Hz\n` +
        `   Duration: ${(
          extractedIR.length / currentSettings.sampleRate
        ).toFixed(2)}s` +
        (hasTrueStereo ? `\n   Type: True Stereo (L+R channels)` : ""),
      "success"
    );
  } catch (error) {
    showOutput("irOutput", `❌ Download error: ${error.message}`, "error");
  }
}

/**
 * Set the regularization parameter via preset buttons
 * @param {number} value - Regularization value
 */
export function setRegularization(value) {
  document.getElementById("regularization").value = value.toString();
}
