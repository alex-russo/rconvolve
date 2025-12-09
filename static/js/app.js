// rconvolve Hall Measurement App
// Main Application Entry Point

// ============================================================================
// Module Imports
// ============================================================================

import {
  init,
  WasmSweepGenerator,
  WasmIRProcessor,
  audioContext,
  SWEEP_CONFIG,
  setAudioContext,
  setSweepSignal,
  setIRProcessor,
} from "./state.js";

import { refreshAudioDevices, refreshRealtimeDevices } from "./devices.js";

import { loadExternalIR, useExtractedIR, loadDemoIR } from "./ir-loader.js";

import {
  startRealTimeConvolution,
  stopRealTimeConvolution,
  switchRealtimeInput,
  switchRealtimeOutput,
  updateDryWetMix,
  updateOutputGain,
  updateStereoSpread,
  updateStereoModeUI,
} from "./realtime.js";

import {
  startRecording,
  stopRecording,
  useRecordedAudio,
  downloadRecording,
} from "./recording.js";

import { generateSweep, downloadSweep, toggleSweepSettings } from "./sweep.js";

import {
  extractIR,
  previewIR,
  downloadIR,
  setRegularization,
} from "./ir-extraction.js";

import { showOutput } from "./visualization.js";

import { setRecordedResponse } from "./state.js";

import { calculatePeak, calculateRMS } from "./audio-utils.js";
import { visualizeAudio } from "./visualization.js";

// ============================================================================
// Initialization
// ============================================================================

async function initializeApp() {
  try {
    // Initialize WASM
    await init();

    // Initialize Web Audio
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    setAudioContext(ctx);

    // Generate the reference sweep (same as used for recording)
    const sweepSignal = WasmSweepGenerator.exponential(
      SWEEP_CONFIG.sampleRate,
      SWEEP_CONFIG.duration,
      SWEEP_CONFIG.startFreq,
      SWEEP_CONFIG.endFreq
    );
    setSweepSignal(sweepSignal);

    // Create IR processor
    const irProcessor = new WasmIRProcessor(sweepSignal);
    setIRProcessor(irProcessor);

    document.getElementById("wasmStatus").textContent = "Ready";
    document.getElementById("wasmStatus").className = "status ready";

    // Enable sweep generation
    document.getElementById("generateBtn").disabled = false;

    // Initialize audio devices
    await refreshAudioDevices();
    await refreshRealtimeDevices();

    console.log("✅ rconvolve app initialized");
    console.log(`📊 Reference sweep: ${sweepSignal.length} samples`);
  } catch (error) {
    console.error("Initialization failed:", error);
    document.getElementById("wasmStatus").textContent = "Failed";
    document.getElementById("wasmStatus").className = "status error";
    showOutput(
      "fileOutput",
      `❌ Initialization failed: ${error.message}`,
      "error"
    );
  }
}

// ============================================================================
// File Upload Handler
// ============================================================================

async function handleFileUpload(event) {
  try {
    const file = event.target.files[0];
    if (!file) return;

    showOutput("fileOutput", "📁 Loading file...", "");

    // Get or create audio context
    let ctx = audioContext;
    if (!ctx || ctx.state === "closed") {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      setAudioContext(ctx);
    }

    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

    // Convert to Float32Array (first channel for backwards compatibility)
    const recordedResponse = new Float32Array(audioBuffer.getChannelData(0));

    // Also store stereo data if available for true stereo IR extraction
    let recordedResponseLeft = null;
    let recordedResponseRight = null;
    if (audioBuffer.numberOfChannels >= 2) {
      recordedResponseLeft = new Float32Array(audioBuffer.getChannelData(0));
      recordedResponseRight = new Float32Array(audioBuffer.getChannelData(1));
    } else {
      // Mono file - use same data for both channels
      recordedResponseLeft = recordedResponse;
      recordedResponseRight = recordedResponse;
    }

    setRecordedResponse(
      recordedResponse,
      recordedResponseLeft,
      recordedResponseRight
    );

    const hasStereo = audioBuffer.numberOfChannels >= 2;

    // Analyze the file
    const peak = calculatePeak(recordedResponse);
    const rms = calculateRMS(recordedResponse);
    const duration = recordedResponse.length / audioBuffer.sampleRate;

    showOutput(
      "fileOutput",
      `✅ File loaded successfully!\n` +
        `   Duration: ${duration.toFixed(2)}s\n` +
        `   Sample rate: ${audioBuffer.sampleRate} Hz\n` +
        `   Samples: ${recordedResponse.length}\n` +
        `   Channels: ${audioBuffer.numberOfChannels}${
          hasStereo ? " (True Stereo extraction available)" : " (Mono)"
        }\n` +
        `   Peak level: ${peak.toFixed(3)}\n` +
        `   RMS level: ${rms.toFixed(4)}\n` +
        `   Dynamic range: ${(20 * Math.log10(peak / rms)).toFixed(1)} dB\n` +
        `\n💡 Using sweep generator settings for processing\n` +
        `   (Uncheck "Use sweep generator settings" if you need custom settings)`,
      "success"
    );

    visualizeAudio(recordedResponse, "fileCanvas", "Recorded Sweep Response");

    // Enable extraction
    document.getElementById("extractBtn").disabled = false;
  } catch (error) {
    showOutput("fileOutput", `❌ File load error: ${error.message}`, "error");
  }
}

// ============================================================================
// Wrapper Functions for IR Preview/Download (need current state)
// ============================================================================

function handlePreviewIR() {
  // Import current state values
  import("./state.js").then((state) => {
    previewIR(state.extractedIR);
  });
}

function handleDownloadIR() {
  // Import current state values
  import("./state.js").then((state) => {
    downloadIR(
      state.extractedIR,
      state.extractedIR_Left,
      state.extractedIR_Right
    );
  });
}

// ============================================================================
// Event Listeners & Setup
// ============================================================================

function setupEventListeners() {
  // Dry/wet mix and output controls
  document
    .getElementById("dryWetMix")
    .addEventListener("input", updateDryWetMix);
  document
    .getElementById("outputGain")
    .addEventListener("input", updateOutputGain);
  document
    .getElementById("stereoSpread")
    .addEventListener("input", updateStereoSpread);
  document
    .getElementById("stereoMode")
    .addEventListener("change", updateStereoModeUI);

  // Initialize stereo mode UI state
  updateStereoModeUI();

  // Real-time device switching listeners
  document
    .getElementById("realtimeInput")
    .addEventListener("change", switchRealtimeInput);
  document
    .getElementById("realtimeOutput")
    .addEventListener("change", switchRealtimeOutput);
}

// ============================================================================
// Window Exports for onclick Handlers
// ============================================================================

window.refreshAudioDevices = refreshAudioDevices;
window.refreshRealtimeDevices = refreshRealtimeDevices;
window.loadExternalIR = loadExternalIR;
window.loadDemoIR = loadDemoIR;
window.useExtractedIR = useExtractedIR;
window.startRealTimeConvolution = startRealTimeConvolution;
window.stopRealTimeConvolution = stopRealTimeConvolution;
window.updateDryWetMix = updateDryWetMix;
window.updateOutputGain = updateOutputGain;
window.updateStereoSpread = updateStereoSpread;
window.updateStereoModeUI = updateStereoModeUI;
window.startRecording = startRecording;
window.stopRecording = stopRecording;
window.useRecordedAudio = useRecordedAudio;
window.generateSweep = generateSweep;
window.downloadSweep = downloadSweep;
window.downloadRecording = downloadRecording;
window.toggleSweepSettings = toggleSweepSettings;
window.handleFileUpload = handleFileUpload;
window.extractIR = extractIR;
window.previewIR = handlePreviewIR;
window.downloadIR = handleDownloadIR;
window.setRegularization = setRegularization;

// ============================================================================
// Application Start
// ============================================================================

// Set up event listeners
setupEventListeners();

// Initialize when page loads
initializeApp();
