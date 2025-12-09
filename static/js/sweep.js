// rconvolve Hall Measurement App - Sweep Generation
// Sine sweep generation and related utilities

import {
  getOrCreateAudioContext,
  sweepSignal,
  SWEEP_CONFIG,
  WasmSweepGenerator,
  WasmIRProcessor,
  setSweepSignal,
  setIRProcessor,
  updateSweepConfig,
} from "./state.js";
import { calculatePeak } from "./audio-utils.js";
import { encodeWAV } from "./wav-encoder.js";
import { showOutput, visualizeAudio } from "./visualization.js";

/**
 * Generate a test sweep signal using WASM
 */
export async function generateSweep() {
  try {
    const sampleRate = parseFloat(document.getElementById("sampleRate").value);
    const duration = parseFloat(document.getElementById("duration").value);
    const startFreq = parseFloat(document.getElementById("startFreq").value);
    const endFreq = parseFloat(document.getElementById("endFreq").value);

    // Generate sweep using WASM
    const sweep = WasmSweepGenerator.exponential(
      sampleRate,
      duration,
      startFreq,
      endFreq
    );
    setSweepSignal(sweep);

    // Update the reference sweep config
    updateSweepConfig({
      sampleRate,
      duration,
      startFreq,
      endFreq,
    });

    // Update IR processor with new sweep
    setIRProcessor(new WasmIRProcessor(sweep));

    showOutput(
      "sweepOutput",
      `✅ Generated sweep: ${sweep.length} samples (${duration}s)\n` +
        `   Sample rate: ${sampleRate} Hz\n` +
        `   Frequency range: ${startFreq} - ${endFreq} Hz\n` +
        `   Peak level: ${calculatePeak(sweep).toFixed(3)}`,
      "success"
    );

    visualizeAudio(sweep, "sweepCanvas", "Generated Sine Sweep");

    // Enable download
    document.getElementById("downloadSweepBtn").disabled = false;

    // Preview playback (full sweep)
    const ctx = await getOrCreateAudioContext();

    const previewBuffer = ctx.createBuffer(1, sweep.length, sampleRate);
    previewBuffer.copyToChannel(sweep, 0);

    const source = ctx.createBufferSource();
    const gainNode = ctx.createGain();
    gainNode.gain.value = 0.1; // Quiet preview

    source.buffer = previewBuffer;
    source.connect(gainNode);
    gainNode.connect(ctx.destination);
    source.start();
  } catch (error) {
    showOutput(
      "sweepOutput",
      `❌ Sweep generation error: ${error.message}`,
      "error"
    );
  }
}

/**
 * Download the generated sweep as a WAV file
 */
export function downloadSweep() {
  if (!sweepSignal) {
    showOutput("sweepOutput", "❌ Generate sweep first!", "error");
    return;
  }

  try {
    const wavBuffer = encodeWAV(sweepSignal, SWEEP_CONFIG.sampleRate);
    const blob = new Blob([wavBuffer], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "test_sweep.wav";
    a.click();

    URL.revokeObjectURL(url);

    showOutput(
      "sweepOutput",
      `💾 Downloaded sweep as WAV file\n` +
        `   File: test_sweep.wav\n` +
        `   Duration: ${SWEEP_CONFIG.duration}s at ${SWEEP_CONFIG.sampleRate} Hz`,
      "success"
    );
  } catch (error) {
    showOutput("sweepOutput", `❌ Download error: ${error.message}`, "error");
  }
}

/**
 * Toggle visibility of custom sweep settings
 */
export function toggleSweepSettings() {
  const useSweepSettings = document.getElementById("useSweepSettings").checked;
  const customSettings = document.getElementById("customSweepSettings");

  if (useSweepSettings) {
    customSettings.style.display = "none";
  } else {
    customSettings.style.display = "block";
    // Copy current sweep settings to custom fields
    document.getElementById("recordedSampleRate").value =
      document.getElementById("sampleRate").value;
    document.getElementById("recordedDuration").value =
      document.getElementById("duration").value;
    document.getElementById("recordedStartFreq").value =
      document.getElementById("startFreq").value;
    document.getElementById("recordedEndFreq").value =
      document.getElementById("endFreq").value;
  }
}

/**
 * Get the current sweep settings (from generator or custom fields)
 * @returns {{sampleRate: number, duration: number, startFreq: number, endFreq: number}}
 */
export function getCurrentSweepSettings() {
  const useSweepSettings = document.getElementById("useSweepSettings").checked;

  if (useSweepSettings) {
    return {
      sampleRate: parseFloat(document.getElementById("sampleRate").value),
      duration: parseFloat(document.getElementById("duration").value),
      startFreq: parseFloat(document.getElementById("startFreq").value),
      endFreq: parseFloat(document.getElementById("endFreq").value),
    };
  } else {
    return {
      sampleRate: parseFloat(
        document.getElementById("recordedSampleRate").value
      ),
      duration: parseFloat(document.getElementById("recordedDuration").value),
      startFreq: parseFloat(document.getElementById("recordedStartFreq").value),
      endFreq: parseFloat(document.getElementById("recordedEndFreq").value),
    };
  }
}
