// rconvolve Hall Measurement App - Real-Time Convolution
// Real-time audio processing with convolution

import {
  audioContext,
  getOrCreateAudioContext,
  activeIR,
  activeIR_Left,
  activeIR_Right,
  convolutionProcessor,
  trueStereoProcessor,
  realtimeStream,
  scriptProcessor,
  isProcessing,
  inputAnalyser,
  WasmConvolutionProcessor,
  WasmTrueStereoConvolutionProcessor,
  setConvolutionProcessor,
  setTrueStereoProcessor,
  setRealtimeStream,
  setScriptProcessor,
  setIsProcessing,
  setIsStereoMode,
  setInputAnalyser,
  setOutputAnalyser,
} from "./state.js";
import {
  showOutput,
  visualizeRealtime,
  stopVisualization,
  clearCanvas,
} from "./visualization.js";

/**
 * Start real-time convolution processing
 */
export async function startRealTimeConvolution() {
  if (!activeIR) {
    showOutput(
      "realtimeStatus",
      "❌ Load or extract an impulse response first!",
      "error"
    );
    return;
  }

  try {
    const inputDeviceId = document.getElementById("realtimeInput").value;
    const outputDeviceId = document.getElementById("realtimeOutput").value;
    const blockSize = parseInt(document.getElementById("blockSize").value);

    // Check stereo mode settings
    const stereoModeSelect = document.getElementById("stereoMode");
    const stereoMode = stereoModeSelect ? stereoModeSelect.value : "mono";
    setIsStereoMode(stereoMode !== "mono");

    // Create or resume audio context
    const ctx = await getOrCreateAudioContext();

    // Set output device if supported
    if (outputDeviceId && ctx.setSinkId) {
      await ctx.setSinkId(outputDeviceId);
    }

    // Get audio stream from selected input - request stereo for true stereo mode
    const channelCount = stereoMode !== "mono" ? 2 : 1;
    const constraints = {
      audio: inputDeviceId
        ? {
            deviceId: { exact: inputDeviceId },
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            channelCount: { ideal: channelCount },
          }
        : {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            channelCount: { ideal: channelCount },
          },
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    setRealtimeStream(stream);

    // Determine actual input channel count
    const track = stream.getAudioTracks()[0];
    const settings = track.getSettings();
    const actualInputChannels = settings.channelCount || 1;

    // Check if we have a true stereo IR (different L and R)
    const hasStereoIR =
      activeIR_Left &&
      activeIR_Right &&
      activeIR_Left !== activeIR_Right &&
      activeIR_Left.length > 0;

    // Determine if we're using stereo processing
    const inputIsStereo = actualInputChannels >= 2;
    const useStereoProcessing = stereoMode !== "mono";

    // Create convolution processor based on mode
    if (stereoMode === "mono") {
      // Mono processing
      setConvolutionProcessor(
        new WasmConvolutionProcessor(activeIR, blockSize)
      );
      setTrueStereoProcessor(null);
    } else if (stereoMode === "stereo") {
      // Stereo processing - use IR channels as-is, no synthetic crossfeed
      if (hasStereoIR) {
        // Stereo IR: process L and R independently (no crossfeed)
        setTrueStereoProcessor(
          WasmTrueStereoConvolutionProcessor.from_stereo_with_crossfeed(
            activeIR_Left,
            activeIR_Right,
            blockSize,
            0.0, // No crossfeed - preserve original stereo image
            0 // No delay
          )
        );
      } else {
        // Mono IR in stereo mode: duplicate to both channels
        setTrueStereoProcessor(
          WasmTrueStereoConvolutionProcessor.from_mono_with_spread(
            activeIR,
            blockSize,
            0.0 // No spread - same IR on both channels
          )
        );
      }
      setConvolutionProcessor(null);
    } else if (stereoMode === "stereoSpread") {
      // Stereo + Spread: create synthetic stereo width from mono IR
      const spreadValue = document.getElementById("stereoSpread")?.value || 50;
      const spread = parseFloat(spreadValue) / 100;

      setTrueStereoProcessor(
        WasmTrueStereoConvolutionProcessor.from_mono_with_spread(
          activeIR,
          blockSize,
          spread
        )
      );
      setConvolutionProcessor(null);
    } else {
      // Fallback to mono
      setConvolutionProcessor(
        new WasmConvolutionProcessor(activeIR, blockSize)
      );
      setTrueStereoProcessor(null);
    }

    // Create audio nodes
    const source = ctx.createMediaStreamSource(stream);

    // Create analysers for visualization
    const inAnalyser = ctx.createAnalyser();
    inAnalyser.fftSize = 256;
    setInputAnalyser(inAnalyser);

    const outAnalyser = ctx.createAnalyser();
    outAnalyser.fftSize = 256;
    setOutputAnalyser(outAnalyser);

    // Create script processor for convolution
    // Input channels: based on actual mic input
    // Output channels: stereo if using stereo processing, else mono
    const inputChannels = inputIsStereo ? 2 : 1;
    const outputChannels = useStereoProcessing ? 2 : 1;
    const processor = ctx.createScriptProcessor(
      blockSize,
      inputChannels,
      outputChannels
    );
    setScriptProcessor(processor);

    // Create gain nodes for dry/wet mix
    const dryGain = ctx.createGain();
    const wetGain = ctx.createGain();
    const outputGain = ctx.createGain();

    // Store references for later adjustment
    window.dryGainNode = dryGain;
    window.wetGainNode = wetGain;
    window.outputGainNode = outputGain;

    // Set initial gains
    updateDryWetMix();
    updateOutputGain();

    processor.onaudioprocess = (event) => {
      try {
        if (useStereoProcessing && trueStereoProcessor) {
          // True stereo processing (works with mono or stereo input)
          const inputLeft = event.inputBuffer.getChannelData(0);
          // If input is mono, use same signal for both L and R
          const inputRight =
            event.inputBuffer.numberOfChannels > 1
              ? event.inputBuffer.getChannelData(1)
              : inputLeft;
          const outputLeft = event.outputBuffer.getChannelData(0);
          const outputRight =
            event.outputBuffer.numberOfChannels > 1
              ? event.outputBuffer.getChannelData(1)
              : outputLeft;

          // Process through true stereo WASM convolution
          const processed = trueStereoProcessor.process_block(
            inputLeft,
            inputRight
          );

          // Deinterleave output (L, R, L, R, ...) to separate channels
          for (let i = 0; i < outputLeft.length; i++) {
            outputLeft[i] = processed[i * 2] || 0;
            outputRight[i] = processed[i * 2 + 1] || 0;
          }
        } else {
          // Mono processing (original behavior)
          const inputData = event.inputBuffer.getChannelData(0);
          const outputData = event.outputBuffer.getChannelData(0);

          const processed = convolutionProcessor.process_block(inputData);

          for (let i = 0; i < outputData.length; i++) {
            outputData[i] = processed[i] || 0;
          }
        }
      } catch (error) {
        // On error, pass through dry signal
        const inputData = event.inputBuffer.getChannelData(0);
        const outputData = event.outputBuffer.getChannelData(0);
        for (let i = 0; i < outputData.length; i++) {
          outputData[i] = inputData[i];
        }
        if (useStereoProcessing && event.outputBuffer.numberOfChannels > 1) {
          const inputRight =
            event.inputBuffer.numberOfChannels > 1
              ? event.inputBuffer.getChannelData(1)
              : inputData;
          const outputRight = event.outputBuffer.getChannelData(1);
          for (let i = 0; i < outputRight.length; i++) {
            outputRight[i] = inputRight[i];
          }
        }
      }
    };

    // Connect audio graph
    source.connect(inAnalyser);
    source.connect(dryGain);
    source.connect(processor);

    processor.connect(wetGain);

    dryGain.connect(outputGain);
    wetGain.connect(outputGain);

    outputGain.connect(outAnalyser);
    outputGain.connect(ctx.destination);

    setIsProcessing(true);

    document.getElementById("startConvBtn").disabled = true;
    document.getElementById("stopConvBtn").disabled = false;

    // Start visualization
    visualizeRealtime();

    const latencyMs = ((blockSize / ctx.sampleRate) * 1000).toFixed(1);
    const modeText =
      stereoMode === "mono"
        ? "Mono"
        : stereoMode === "stereo"
        ? "Stereo"
        : "Stereo + Spread";
    const inputText = inputIsStereo ? "Stereo" : "Mono";
    const irText = hasStereoIR ? "Stereo" : "Mono";
    showOutput(
      "realtimeStatus",
      `▶️ Real-time convolution active!\n` +
        `   Mode: ${modeText}\n` +
        `   Input: ${inputText}\n` +
        `   IR: ${irText}\n` +
        `   Block size: ${blockSize} samples\n` +
        `   Sample rate: ${ctx.sampleRate} Hz\n` +
        `   Estimated latency: ${latencyMs} ms\n` +
        `   IR length: ${activeIR.length} samples\n` +
        `\n🎧 Speak or play audio into your input device!`,
      "success"
    );
  } catch (error) {
    showOutput(
      "realtimeStatus",
      `❌ Failed to start processing: ${error.message}`,
      "error"
    );
  }
}

/**
 * Stop real-time convolution processing
 */
export function stopRealTimeConvolution() {
  setIsProcessing(false);

  stopVisualization();

  if (scriptProcessor) {
    scriptProcessor.disconnect();
    setScriptProcessor(null);
  }

  if (realtimeStream) {
    realtimeStream.getTracks().forEach((track) => track.stop());
    setRealtimeStream(null);
  }

  if (convolutionProcessor) {
    convolutionProcessor.reset();
  }

  if (trueStereoProcessor) {
    trueStereoProcessor.reset();
    setTrueStereoProcessor(null);
  }

  setIsStereoMode(false);

  document.getElementById("startConvBtn").disabled = false;
  document.getElementById("stopConvBtn").disabled = true;

  // Clear canvas
  clearCanvas("realtimeCanvas");

  showOutput("realtimeStatus", "⏹️ Real-time processing stopped.", "");
}

/**
 * Switch input device while processing is active
 */
export async function switchRealtimeInput() {
  if (!isProcessing) return;

  const inputDeviceId = document.getElementById("realtimeInput").value;

  try {
    // Stop old stream
    if (realtimeStream) {
      realtimeStream.getTracks().forEach((track) => track.stop());
    }

    // Get new stream with selected input
    const constraints = {
      audio: inputDeviceId
        ? {
            deviceId: { exact: inputDeviceId },
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          }
        : {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          },
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    setRealtimeStream(stream);

    // Recreate the audio graph with the new input
    const source = audioContext.createMediaStreamSource(stream);

    // Reconnect to existing nodes
    source.connect(inputAnalyser);
    source.connect(window.dryGainNode);
    source.connect(scriptProcessor);

    showOutput(
      "realtimeStatus",
      `🔄 Switched input device\n` + `   Real-time convolution continues...`,
      "success"
    );
  } catch (error) {
    showOutput(
      "realtimeStatus",
      `❌ Failed to switch input: ${error.message}`,
      "error"
    );
  }
}

/**
 * Switch output device while processing is active
 */
export async function switchRealtimeOutput() {
  if (!isProcessing) return;

  const outputDeviceId = document.getElementById("realtimeOutput").value;

  try {
    // Set new output device if supported
    if (audioContext.setSinkId) {
      await audioContext.setSinkId(outputDeviceId || "");
      showOutput(
        "realtimeStatus",
        `🔄 Switched output device\n` + `   Real-time convolution continues...`,
        "success"
      );
    } else {
      showOutput(
        "realtimeStatus",
        `⚠️ Output device switching not supported in this browser`,
        ""
      );
    }
  } catch (error) {
    showOutput(
      "realtimeStatus",
      `❌ Failed to switch output: ${error.message}`,
      "error"
    );
  }
}

/**
 * Update the dry/wet mix based on UI slider
 */
export function updateDryWetMix() {
  const mix = parseInt(document.getElementById("dryWetMix").value) / 100;
  document.getElementById("dryWetValue").textContent = `${Math.round(
    mix * 100
  )}%`;

  if (window.dryGainNode && window.wetGainNode) {
    window.dryGainNode.gain.value = 1 - mix;
    window.wetGainNode.gain.value = mix;
  }
}

/**
 * Update the output gain based on UI slider
 */
export function updateOutputGain() {
  const gain = parseInt(document.getElementById("outputGain").value) / 100;
  document.getElementById("gainValue").textContent = `${Math.round(
    gain * 100
  )}%`;

  if (window.outputGainNode) {
    window.outputGainNode.gain.value = gain;
  }
}

/**
 * Update the stereo spread value display
 */
export function updateStereoSpread() {
  const spread = parseInt(document.getElementById("stereoSpread").value);
  document.getElementById("stereoSpreadValue").textContent = `${spread}%`;
}

/**
 * Update the stereo mode UI (show/hide spread controls)
 */
export function updateStereoModeUI() {
  const stereoMode = document.getElementById("stereoMode").value;
  const spreadControls = document.getElementById("stereoSpreadControls");
  if (spreadControls) {
    spreadControls.style.display =
      stereoMode === "stereoSpread" ? "inline" : "none";
  }
}
