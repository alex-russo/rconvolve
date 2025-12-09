// rconvolve Hall Measurement App - Visualization
// Canvas drawing and visualization functions

import { calculatePeak } from "./audio-utils.js";
import {
  isProcessing,
  inputAnalyser,
  outputAnalyser,
  animationFrameId,
  setAnimationFrameId,
} from "./state.js";

/**
 * Display a message in an output element
 * @param {string} elementId - ID of the output element
 * @param {string} message - Message to display
 * @param {string} type - CSS class type (success, error, info, etc.)
 */
export function showOutput(elementId, message, type = "") {
  const element = document.getElementById(elementId);
  element.textContent = message;
  element.className = `output ${type}`;
}

/**
 * Visualize audio waveform on a canvas
 * @param {Float32Array} audioData - Audio samples to visualize
 * @param {string} canvasId - ID of the canvas element
 * @param {string} title - Title to display on the canvas
 */
export function visualizeAudio(audioData, canvasId, title) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  const peak = calculatePeak(audioData);
  if (peak === 0) {
    canvas.classList.remove("has-content");
    return;
  }

  // Mark canvas as having content
  canvas.classList.add("has-content");

  // Draw waveform
  ctx.strokeStyle = "#667eea";
  ctx.lineWidth = 1;
  ctx.beginPath();

  const step = audioData.length / width;
  for (let x = 0; x < width; x++) {
    const sampleIndex = Math.floor(x * step);
    const sample = audioData[sampleIndex] || 0;
    const y = height / 2 - (((sample / peak) * height) / 2) * 0.8;

    if (x === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  // Draw title and info
  ctx.fillStyle = "#000000";
  ctx.font = "bold 14px sans-serif";
  ctx.fillText(title, 10, 20);

  ctx.fillStyle = "#333333";
  ctx.font = "12px sans-serif";
  ctx.fillText(`Peak: ${peak.toFixed(4)}`, width - 100, height - 10);
}

/**
 * Visualize real-time audio input/output on a canvas
 * Uses frequency domain visualization with bars
 */
export function visualizeRealtime() {
  if (!isProcessing) return;

  const canvas = document.getElementById("realtimeCanvas");
  canvas.classList.add("has-content");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  // Draw input (top half)
  if (inputAnalyser) {
    const inputData = new Uint8Array(inputAnalyser.frequencyBinCount);
    inputAnalyser.getByteFrequencyData(inputData);

    ctx.fillStyle = "#667eea";
    const barWidth = width / inputData.length;
    for (let i = 0; i < inputData.length; i++) {
      const barHeight = (inputData[i] / 255) * (height / 2 - 5);
      ctx.fillRect(
        i * barWidth,
        height / 2 - barHeight - 2,
        barWidth - 1,
        barHeight
      );
    }
  }

  // Draw output (bottom half)
  if (outputAnalyser) {
    const outputData = new Uint8Array(outputAnalyser.frequencyBinCount);
    outputAnalyser.getByteFrequencyData(outputData);

    ctx.fillStyle = "#764ba2";
    const barWidth = width / outputData.length;
    for (let i = 0; i < outputData.length; i++) {
      const barHeight = (outputData[i] / 255) * (height / 2 - 5);
      ctx.fillRect(i * barWidth, height / 2 + 2, barWidth - 1, barHeight);
    }
  }

  // Draw labels
  ctx.fillStyle = "#000000";
  ctx.font = "10px sans-serif";
  ctx.fillText("Input", 5, 12);
  ctx.fillStyle = "white";
  ctx.fillText("Output", 5, height / 2 + 14);

  setAnimationFrameId(requestAnimationFrame(visualizeRealtime));
}

/**
 * Stop the real-time visualization loop
 */
export function stopVisualization() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    setAnimationFrameId(null);
  }
}

/**
 * Clear a canvas
 * @param {string} canvasId - ID of the canvas element
 */
export function clearCanvas(canvasId) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas.classList.remove("has-content");
}
