// rconvolve Hall Measurement App - Audio Devices
// Audio device enumeration and management

import { showOutput } from "./visualization.js";

/**
 * Get the number of audio channels supported by a device
 * @param {string} deviceId - Media device ID
 * @returns {Promise<number|null>} Number of channels, or null if unknown
 */
export async function getDeviceChannelCount(deviceId) {
  try {
    // Request stereo explicitly to detect if device supports it
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: { exact: deviceId },
        channelCount: { ideal: 2 }, // Request stereo
      },
    });
    const track = stream.getAudioTracks()[0];
    const settings = track.getSettings();

    // channelCount in settings reflects what we actually got
    let channelCount = settings.channelCount;

    // If channelCount not available, try to detect via AudioContext
    if (!channelCount) {
      try {
        const audioCtx = new (window.AudioContext ||
          window.webkitAudioContext)();
        const source = audioCtx.createMediaStreamSource(stream);
        channelCount = source.channelCount;
        await audioCtx.close();
      } catch (e) {
        channelCount = 1; // Default to mono if detection fails
      }
    }

    stream.getTracks().forEach((t) => t.stop());
    return channelCount;
  } catch (e) {
    return null; // Unknown
  }
}

/**
 * Refresh and populate the audio device selectors for recording
 */
export async function refreshAudioDevices() {
  try {
    // Request permissions first
    await navigator.mediaDevices.getUserMedia({ audio: true });

    const devices = await navigator.mediaDevices.enumerateDevices();

    const audioInputSelect = document.getElementById("audioInput");
    const audioOutputSelect = document.getElementById("audioOutput");

    // Clear existing options
    audioInputSelect.innerHTML =
      '<option value="">Select input device...</option>';
    audioOutputSelect.innerHTML =
      '<option value="">Select output device...</option>';

    // Collect input devices for channel detection
    const inputDevices = devices.filter((d) => d.kind === "audioinput");
    const outputDevices = devices.filter((d) => d.kind === "audiooutput");

    // Add output devices immediately (no channel info needed)
    outputDevices.forEach((device) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent =
        device.label || `Output (${device.deviceId.slice(0, 8)}...)`;
      audioOutputSelect.appendChild(option);
    });

    // Add input devices with channel count
    for (const device of inputDevices) {
      const option = document.createElement("option");
      option.value = device.deviceId;
      const label = device.label || `Input (${device.deviceId.slice(0, 8)}...)`;

      // Get channel count for this device
      const channels = await getDeviceChannelCount(device.deviceId);
      if (channels !== null) {
        option.textContent = `${label} [${channels}ch]`;
      } else {
        option.textContent = label;
      }

      audioInputSelect.appendChild(option);
    }

    // Enable record button if input is available
    if (audioInputSelect.options.length > 1) {
      document.getElementById("recordBtn").disabled = false;
    }

    showOutput(
      "recordOutput",
      `✅ Found ${audioInputSelect.options.length - 1} input(s) and ${
        audioOutputSelect.options.length - 1
      } output(s)`,
      "success"
    );
  } catch (error) {
    showOutput(
      "recordOutput",
      `❌ Could not access audio devices: ${error.message}`,
      "error"
    );
  }
}

/**
 * Refresh and populate the real-time convolution device selectors
 */
export async function refreshRealtimeDevices() {
  try {
    await navigator.mediaDevices.getUserMedia({ audio: true });
    const devices = await navigator.mediaDevices.enumerateDevices();

    const realtimeInputSelect = document.getElementById("realtimeInput");
    const realtimeOutputSelect = document.getElementById("realtimeOutput");

    realtimeInputSelect.innerHTML =
      '<option value="">Select input device...</option>';
    realtimeOutputSelect.innerHTML =
      '<option value="">Select output device...</option>';

    // Collect input devices for channel detection
    const inputDevices = devices.filter((d) => d.kind === "audioinput");
    const outputDevices = devices.filter((d) => d.kind === "audiooutput");

    // Add output devices immediately
    outputDevices.forEach((device) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent =
        device.label || `Output (${device.deviceId.slice(0, 8)}...)`;
      realtimeOutputSelect.appendChild(option);
    });

    // Add input devices with channel count
    for (const device of inputDevices) {
      const option = document.createElement("option");
      option.value = device.deviceId;
      const label = device.label || `Input (${device.deviceId.slice(0, 8)}...)`;

      // Get channel count for this device
      const channels = await getDeviceChannelCount(device.deviceId);
      if (channels !== null) {
        option.textContent = `${label} [${channels}ch]`;
      } else {
        option.textContent = label;
      }

      realtimeInputSelect.appendChild(option);
    }

    showOutput(
      "realtimeStatus",
      `✅ Found ${realtimeInputSelect.options.length - 1} input(s) and ${
        realtimeOutputSelect.options.length - 1
      } output(s)`,
      "success"
    );
  } catch (error) {
    showOutput(
      "realtimeStatus",
      `❌ Could not access audio devices: ${error.message}`,
      "error"
    );
  }
}
