// rconvolve Hall Measurement App - Global State
// Centralized state management for the application

import init, {
  WasmSweepGenerator,
  WasmIRProcessor,
  WasmConvolutionProcessor,
  WasmTrueStereoConvolutionProcessor,
} from "../rconvolve.js";

// ============================================================================
// WASM Exports (re-exported for other modules)
// ============================================================================

export {
  init,
  WasmSweepGenerator,
  WasmIRProcessor,
  WasmConvolutionProcessor,
  WasmTrueStereoConvolutionProcessor,
};

// ============================================================================
// Audio Context State
// ============================================================================

export let audioContext = null;

export function setAudioContext(ctx) {
  audioContext = ctx;
}

export async function getOrCreateAudioContext() {
  if (!audioContext || audioContext.state === "closed") {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }
  return audioContext;
}

// ============================================================================
// Sweep & IR Processor State
// ============================================================================

export let sweepSignal = null;
export let irProcessor = null;

export function setSweepSignal(signal) {
  sweepSignal = signal;
}

export function setIRProcessor(processor) {
  irProcessor = processor;
}

// ============================================================================
// Recorded Response State
// ============================================================================

export let recordedResponse = null;
export let recordedResponseLeft = null;
export let recordedResponseRight = null;

export function setRecordedResponse(response, left = null, right = null) {
  recordedResponse = response;
  recordedResponseLeft = left;
  recordedResponseRight = right;
}

// ============================================================================
// Extracted IR State
// ============================================================================

export let extractedIR = null;
export let extractedIR_Left = null;
export let extractedIR_Right = null;

export function setExtractedIR(ir, left = null, right = null) {
  extractedIR = ir;
  extractedIR_Left = left;
  extractedIR_Right = right;
}

// ============================================================================
// Active IR State (for real-time convolution)
// ============================================================================

export let loadedIR = null;
export let activeIR = null;
export let activeIR_Left = null;
export let activeIR_Right = null;

export function setLoadedIR(ir) {
  loadedIR = ir;
}

export function setActiveIR(ir, left = null, right = null) {
  activeIR = ir;
  activeIR_Left = left;
  activeIR_Right = right;
}

// ============================================================================
// Real-time Convolution State
// ============================================================================

export let convolutionProcessor = null;
export let trueStereoProcessor = null;
export let realtimeStream = null;
export let scriptProcessor = null;
export let isProcessing = false;
export let isStereoMode = false;
export let inputAnalyser = null;
export let outputAnalyser = null;
export let animationFrameId = null;

export function setConvolutionProcessor(processor) {
  convolutionProcessor = processor;
}

export function setTrueStereoProcessor(processor) {
  trueStereoProcessor = processor;
}

export function setRealtimeStream(stream) {
  realtimeStream = stream;
}

export function setScriptProcessor(processor) {
  scriptProcessor = processor;
}

export function setIsProcessing(value) {
  isProcessing = value;
}

export function setIsStereoMode(value) {
  isStereoMode = value;
}

export function setInputAnalyser(analyser) {
  inputAnalyser = analyser;
}

export function setOutputAnalyser(analyser) {
  outputAnalyser = analyser;
}

export function setAnimationFrameId(id) {
  animationFrameId = id;
}

// ============================================================================
// Recording State
// ============================================================================

export let mediaRecorder = null;
export let recordedChunks = [];
export let recordedAudioBuffer = null;
export let recordingStream = null;
export let isRecording = false;
export let recordingTimeout = null;

export function setMediaRecorder(recorder) {
  mediaRecorder = recorder;
}

export function setRecordedChunks(chunks) {
  recordedChunks = chunks;
}

export function setRecordedAudioBuffer(buffer) {
  recordedAudioBuffer = buffer;
}

export function setRecordingStream(stream) {
  recordingStream = stream;
}

export function setIsRecording(value) {
  isRecording = value;
}

export function setRecordingTimeout(timeout) {
  recordingTimeout = timeout;
}

// ============================================================================
// Default Sweep Configuration
// ============================================================================

export const SWEEP_CONFIG = {
  sampleRate: 48000,
  duration: 2.0,
  startFreq: 20,
  endFreq: 20000,
};

export function updateSweepConfig(config) {
  Object.assign(SWEEP_CONFIG, config);
}
