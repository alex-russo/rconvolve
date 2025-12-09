// rconvolve Hall Measurement App - Recording
// Audio recording functionality

import {
  audioContext,
  getOrCreateAudioContext,
  sweepSignal,
  SWEEP_CONFIG,
  recordedAudioBuffer,
  recordingStream,
  isRecording,
  recordingTimeout,
  setRecordedAudioBuffer,
  setRecordingStream,
  setIsRecording,
  setRecordingTimeout,
  setRecordedResponse,
} from "./state.js";
import { calculatePeak } from "./audio-utils.js";
import { encodeWAV, encodeWAVStereo } from "./wav-encoder.js";
import { showOutput, visualizeAudio } from "./visualization.js";

// Recording buffers for raw PCM capture
let recordingLeftChannel = [];
let recordingRightChannel = [];
let recordingScriptProcessor = null;
let recordingSourceNode = null;

/**
 * Start recording audio with optional sweep playback
 */
export async function startRecording() {
  if (!sweepSignal) {
    showOutput("recordOutput", "❌ Generate a sweep first!", "error");
    return;
  }

  try {
    const inputDeviceId = document.getElementById("audioInput").value;
    const outputDeviceId = document.getElementById("audioOutput").value;
    const recordDuration = parseFloat(
      document.getElementById("recordDuration").value
    );
    const playSweep = document.getElementById(
      "playSweepWhileRecording"
    ).checked;

    // Get audio stream from selected input - request stereo if available
    const constraints = {
      audio: inputDeviceId
        ? {
            deviceId: { exact: inputDeviceId },
            channelCount: { ideal: 2 }, // Request stereo
          }
        : {
            channelCount: { ideal: 2 }, // Request stereo
          },
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    setRecordingStream(stream);

    // Create or resume audio context
    const ctx = await getOrCreateAudioContext();

    // Clear recording buffers
    recordingLeftChannel = [];
    recordingRightChannel = [];

    // Create source from stream
    recordingSourceNode = ctx.createMediaStreamSource(stream);
    const numInputChannels = recordingSourceNode.channelCount;

    // Create script processor for raw PCM capture (2 input channels, 0 output)
    const bufferSize = 4096;
    recordingScriptProcessor = ctx.createScriptProcessor(
      bufferSize,
      numInputChannels,
      1
    );

    recordingScriptProcessor.onaudioprocess = (event) => {
      if (!isRecording) return;

      // Capture left channel
      const leftData = event.inputBuffer.getChannelData(0);
      recordingLeftChannel.push(new Float32Array(leftData));

      // Capture right channel if stereo
      if (event.inputBuffer.numberOfChannels > 1) {
        const rightData = event.inputBuffer.getChannelData(1);
        recordingRightChannel.push(new Float32Array(rightData));
      }
    };

    // Connect: source -> script processor -> destination (required for script processor to work)
    recordingSourceNode.connect(recordingScriptProcessor);
    recordingScriptProcessor.connect(ctx.destination);

    setIsRecording(true);

    document.getElementById("recordBtn").disabled = true;
    document.getElementById("stopRecordBtn").disabled = false;
    document.getElementById("useRecordedBtn").disabled = true;
    document.getElementById("downloadRecordingBtn").disabled = true;

    showOutput(
      "recordOutput",
      `🔴 Recording for ${recordDuration} seconds... (${numInputChannels}ch input)`,
      ""
    );

    // Play sweep if requested
    if (playSweep) {
      const sampleRate = SWEEP_CONFIG.sampleRate;
      const sweepBuffer = ctx.createBuffer(1, sweepSignal.length, sampleRate);
      sweepBuffer.copyToChannel(sweepSignal, 0);

      const source = ctx.createBufferSource();
      source.buffer = sweepBuffer;

      // Set output device if supported
      if (outputDeviceId && ctx.setSinkId) {
        await ctx.setSinkId(outputDeviceId);
      }

      source.connect(ctx.destination);
      source.start();
    }

    // Auto-stop after duration
    const timeout = setTimeout(() => {
      if (isRecording) {
        stopRecording();
      }
    }, recordDuration * 1000);
    setRecordingTimeout(timeout);
  } catch (error) {
    showOutput(
      "recordOutput",
      `❌ Recording failed: ${error.message}`,
      "error"
    );
  }
}

/**
 * Stop recording and process the captured audio
 */
export function stopRecording() {
  if (recordingTimeout) {
    clearTimeout(recordingTimeout);
    setRecordingTimeout(null);
  }

  setIsRecording(false);

  // Disconnect and clean up script processor
  if (recordingScriptProcessor) {
    recordingScriptProcessor.disconnect();
    recordingScriptProcessor = null;
  }
  if (recordingSourceNode) {
    recordingSourceNode.disconnect();
    recordingSourceNode = null;
  }

  // Stop the stream
  if (recordingStream) {
    recordingStream.getTracks().forEach((track) => track.stop());
    setRecordingStream(null);
  }

  // Process the recorded audio
  processRecordedAudio();

  document.getElementById("recordBtn").disabled = false;
  document.getElementById("stopRecordBtn").disabled = true;
}

/**
 * Process and analyze the recorded audio data
 */
function processRecordedAudio() {
  if (recordingLeftChannel.length === 0) {
    showOutput("recordOutput", "❌ No audio was recorded!", "error");
    return;
  }

  // Concatenate all chunks into single buffers
  const totalSamples = recordingLeftChannel.reduce(
    (sum, chunk) => sum + chunk.length,
    0
  );
  const leftBuffer = new Float32Array(totalSamples);
  const rightBuffer =
    recordingRightChannel.length > 0 ? new Float32Array(totalSamples) : null;

  let offset = 0;
  for (let i = 0; i < recordingLeftChannel.length; i++) {
    leftBuffer.set(recordingLeftChannel[i], offset);
    if (rightBuffer && recordingRightChannel[i]) {
      rightBuffer.set(recordingRightChannel[i], offset);
    }
    offset += recordingLeftChannel[i].length;
  }

  // Create an AudioBuffer to store the result
  const sampleRate = audioContext.sampleRate;
  const numChannels = rightBuffer ? 2 : 1;
  const duration = totalSamples / sampleRate;

  // Create a fake AudioBuffer-like object with the data we need
  const buffer = {
    duration: duration,
    sampleRate: sampleRate,
    numberOfChannels: numChannels,
    length: totalSamples,
    getChannelData: (channel) => {
      if (channel === 0) return leftBuffer;
      if (channel === 1 && rightBuffer) return rightBuffer;
      return leftBuffer;
    },
  };
  setRecordedAudioBuffer(buffer);

  // Get channel selection and extract appropriate channel data
  const channelSelection = document.getElementById("recordingChannel").value;
  let channelData;
  let channelInfo;

  if (numChannels === 1) {
    // Mono input - just use what we have
    channelData = leftBuffer;
    channelInfo = "Mono";
  } else {
    // Stereo input - apply channel selection
    if (channelSelection === "stereo") {
      // Keep stereo - use mix for visualization but preserve both channels
      channelData = new Float32Array(leftBuffer.length);
      for (let i = 0; i < leftBuffer.length; i++) {
        channelData[i] = (leftBuffer[i] + rightBuffer[i]) * 0.5;
      }
      channelInfo = "Stereo (L+R preserved)";
    } else if (channelSelection === "left") {
      channelData = leftBuffer;
      channelInfo = "Left channel";
    } else if (channelSelection === "right") {
      channelData = rightBuffer;
      channelInfo = "Right channel";
    } else {
      // Mix both channels
      channelData = new Float32Array(leftBuffer.length);
      for (let i = 0; i < leftBuffer.length; i++) {
        channelData[i] = (leftBuffer[i] + rightBuffer[i]) * 0.5;
      }
      channelInfo = "Mixed (L+R → Mono)";
    }
  }

  // Store the processed channel for later use
  buffer._processedChannel = channelData;
  buffer._channelInfo = channelInfo;
  buffer._isStereoMode = channelSelection === "stereo" && numChannels >= 2;

  showOutput(
    "recordOutput",
    `✅ Recording complete!\n` +
      `   Duration: ${duration.toFixed(2)}s\n` +
      `   Sample rate: ${sampleRate} Hz\n` +
      `   Input channels: ${numChannels} (using: ${channelInfo})\n` +
      `   Samples: ${channelData.length}\n` +
      `   Peak level: ${calculatePeak(channelData).toFixed(3)}`,
    "success"
  );

  visualizeAudio(
    channelData,
    "recordCanvas",
    `Recorded Audio (${channelInfo})`
  );
  document.getElementById("useRecordedBtn").disabled = false;
  document.getElementById("downloadRecordingBtn").disabled = false;

  // Clear the chunk arrays to free memory
  recordingLeftChannel = [];
  recordingRightChannel = [];
}

/**
 * Download the recorded audio as a WAV file
 */
export function downloadRecording() {
  if (!recordedAudioBuffer) {
    showOutput("recordOutput", "❌ No recording available!", "error");
    return;
  }

  try {
    const sampleRate = recordedAudioBuffer.sampleRate;
    const numChannels = recordedAudioBuffer.numberOfChannels;
    let wavBuffer;
    let channelInfo;

    if (numChannels >= 2) {
      // Download as stereo WAV
      const leftData = recordedAudioBuffer.getChannelData(0);
      const rightData = recordedAudioBuffer.getChannelData(1);
      wavBuffer = encodeWAVStereo(leftData, rightData, sampleRate);
      channelInfo = "Stereo";
    } else {
      // Download as mono WAV
      const channelData =
        recordedAudioBuffer._processedChannel ||
        recordedAudioBuffer.getChannelData(0);
      wavBuffer = encodeWAV(channelData, sampleRate);
      channelInfo = recordedAudioBuffer._channelInfo || "Mono";
    }

    const blob = new Blob([wavBuffer], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "recording.wav";
    a.click();

    URL.revokeObjectURL(url);

    showOutput(
      "recordOutput",
      `💾 Downloaded recording as WAV file\n` +
        `   File: recording.wav\n` +
        `   Format: ${channelInfo}\n` +
        `   Duration: ${recordedAudioBuffer.duration.toFixed(
          2
        )}s at ${sampleRate} Hz`,
      "success"
    );
  } catch (error) {
    showOutput("recordOutput", `❌ Download error: ${error.message}`, "error");
  }
}

/**
 * Use the recorded audio for IR extraction
 */
export function useRecordedAudio() {
  if (!recordedAudioBuffer) {
    showOutput("recordOutput", "❌ No recording available!", "error");
    return;
  }

  // Use the processed channel if available, otherwise fall back to channel 0
  const channelData =
    recordedAudioBuffer._processedChannel ||
    recordedAudioBuffer.getChannelData(0);
  const channelInfo = recordedAudioBuffer._channelInfo || "Mono";
  const isStereoMode = recordedAudioBuffer._isStereoMode || false;
  const recordedResponse = new Float32Array(channelData);

  // Store stereo data for true stereo IR extraction only if stereo mode is selected
  let recordedResponseLeft = null;
  let recordedResponseRight = null;
  if (isStereoMode && recordedAudioBuffer.numberOfChannels >= 2) {
    recordedResponseLeft = new Float32Array(
      recordedAudioBuffer.getChannelData(0)
    );
    recordedResponseRight = new Float32Array(
      recordedAudioBuffer.getChannelData(1)
    );
  }

  setRecordedResponse(
    recordedResponse,
    recordedResponseLeft,
    recordedResponseRight
  );

  const stereoStatus =
    isStereoMode && recordedAudioBuffer.numberOfChannels >= 2
      ? "Yes (True Stereo IR)"
      : "No (Mono IR)";

  // Update file output section
  showOutput(
    "fileOutput",
    `✅ Using recorded audio (${channelInfo})\n` +
      `   Duration: ${recordedAudioBuffer.duration.toFixed(2)}s\n` +
      `   Sample rate: ${recordedAudioBuffer.sampleRate} Hz\n` +
      `   Samples: ${recordedResponse.length}\n` +
      `   Stereo extraction: ${stereoStatus}`,
    "success"
  );

  visualizeAudio(
    recordedResponse,
    "fileCanvas",
    `Recorded Response (${channelInfo})`
  );

  // Enable IR extraction
  document.getElementById("extractBtn").disabled = false;

  showOutput(
    "recordOutput",
    `✅ Recording loaded (${channelInfo})! Ready to extract IR.`,
    "success"
  );
}
