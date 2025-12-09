// rconvolve Hall Measurement App - IR Loader
// Functions for loading impulse response files

import {
  getOrCreateAudioContext,
  SWEEP_CONFIG,
  setLoadedIR,
  setActiveIR,
  extractedIR,
  extractedIR_Left,
  extractedIR_Right,
} from "./state.js";
import { showOutput } from "./visualization.js";

/**
 * Load an external IR file from a file input
 * @param {Event} event - File input change event
 */
export async function loadExternalIR(event) {
  const file = event.target.files[0];
  if (!file) return;

  try {
    showOutput("realtimeStatus", `⏳ Loading ${file.name}...`, "");

    // Create audio context if needed
    const ctx = await getOrCreateAudioContext();

    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

    // Get the first channel as Float32Array
    const loadedIR = audioBuffer.getChannelData(0);
    setLoadedIR(loadedIR);

    // Handle stereo IR files
    let activeIR_Left, activeIR_Right;
    if (audioBuffer.numberOfChannels >= 2) {
      activeIR_Left = new Float32Array(audioBuffer.getChannelData(0));
      activeIR_Right = new Float32Array(audioBuffer.getChannelData(1));
    } else {
      // Mono file - use same for both channels
      activeIR_Left = loadedIR;
      activeIR_Right = loadedIR;
    }

    setActiveIR(loadedIR, activeIR_Left, activeIR_Right);

    const hasStereo = audioBuffer.numberOfChannels >= 2;

    // Update UI
    const durationMs = (
      (loadedIR.length / audioBuffer.sampleRate) *
      1000
    ).toFixed(0);
    const stereoLabel = hasStereo ? " (Stereo)" : "";
    document.getElementById(
      "currentIRInfo"
    ).textContent = `📁 ${file.name}${stereoLabel} (${loadedIR.length} samples, ${durationMs}ms @ ${audioBuffer.sampleRate}Hz)`;
    document.getElementById("currentIRInfo").style.color = "#4CAF50";

    // Enable start button
    document.getElementById("startConvBtn").disabled = false;

    showOutput(
      "realtimeStatus",
      `✅ Loaded impulse response: ${file.name}\n` +
        `   Samples: ${loadedIR.length}\n` +
        `   Duration: ${durationMs}ms\n` +
        `   Sample rate: ${audioBuffer.sampleRate}Hz\n` +
        `   Channels: ${audioBuffer.numberOfChannels}${
          hasStereo ? " (True Stereo ready)" : " (mono)"
        }`,
      "success"
    );
  } catch (error) {
    showOutput(
      "realtimeStatus",
      `❌ Failed to load IR file: ${error.message}`,
      "error"
    );
  }

  // Reset file input so same file can be loaded again
  event.target.value = "";
}

/**
 * Use the extracted IR for real-time convolution
 */
export function useExtractedIR() {
  if (!extractedIR) {
    showOutput(
      "realtimeStatus",
      "❌ No extracted IR available. Extract one first!",
      "error"
    );
    return;
  }

  setActiveIR(
    extractedIR,
    extractedIR_Left || extractedIR,
    extractedIR_Right || extractedIR
  );

  const hasTrueStereo =
    extractedIR_Left &&
    extractedIR_Right &&
    extractedIR_Left !== extractedIR_Right;

  const durationMs = (
    (extractedIR.length / SWEEP_CONFIG.sampleRate) *
    1000
  ).toFixed(0);
  const stereoLabel = hasTrueStereo ? " (True Stereo)" : "";
  document.getElementById(
    "currentIRInfo"
  ).textContent = `🎯 Extracted IR${stereoLabel} (${extractedIR.length} samples, ${durationMs}ms)`;
  document.getElementById("currentIRInfo").style.color = "#2196F3";

  showOutput(
    "realtimeStatus",
    `✅ Switched to extracted impulse response${stereoLabel} (${extractedIR.length} samples)`,
    "success"
  );
}

/**
 * Load a demo IR file from the server
 */
export async function loadDemoIR() {
  try {
    showOutput("realtimeStatus", "⏳ Loading demo IR...", "info");

    const response = await fetch("irtest.wav");
    if (!response.ok) {
      throw new Error(`Failed to fetch demo IR: ${response.statusText}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: SWEEP_CONFIG.sampleRate,
    });

    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const loadedIR = audioBuffer.getChannelData(0);
    setLoadedIR(loadedIR);

    // Handle stereo demo IR
    let activeIR_Left, activeIR_Right;
    if (audioBuffer.numberOfChannels >= 2) {
      activeIR_Left = new Float32Array(audioBuffer.getChannelData(0));
      activeIR_Right = new Float32Array(audioBuffer.getChannelData(1));
    } else {
      activeIR_Left = loadedIR;
      activeIR_Right = loadedIR;
    }

    setActiveIR(loadedIR, activeIR_Left, activeIR_Right);

    const hasStereo = audioBuffer.numberOfChannels >= 2;

    const durationMs = (
      (loadedIR.length / SWEEP_CONFIG.sampleRate) *
      1000
    ).toFixed(0);
    const stereoLabel = hasStereo ? " (Stereo)" : "";
    document.getElementById(
      "currentIRInfo"
    ).textContent = `🎵 Demo IR${stereoLabel} (${loadedIR.length} samples, ${durationMs}ms)`;
    document.getElementById("currentIRInfo").style.color = "#9C27B0";

    showOutput(
      "realtimeStatus",
      `✅ Loaded demo impulse response (${
        loadedIR.length
      } samples, ${durationMs}ms)${hasStereo ? " - True Stereo" : ""}`,
      "success"
    );
    document.getElementById("startConvBtn").disabled = false;

    await audioCtx.close();
  } catch (error) {
    console.error("Error loading demo IR:", error);
    showOutput(
      "realtimeStatus",
      `❌ Failed to load demo IR: ${error.message}`,
      "error"
    );
  }
}
