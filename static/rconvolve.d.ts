/* tslint:disable */
/* eslint-disable */
/**
 * Apply convolution reverb to audio data.
 *
 * Convenience function for one-shot (non-realtime) convolution.
 */
export function convolve_audio(audio: Float32Array, impulse_response: Float32Array): Float32Array;
/**
 * Batch stereo convolution (non-real-time)
 */
export function stereo_convolve_audio(left: Float32Array, right: Float32Array, ir_left: Float32Array, ir_right: Float32Array): Float32Array;
/**
 * WebAssembly bindings for real-time partitioned convolution.
 *
 * Provides block-based convolution processing suitable for real-time audio.
 */
export class WasmConvolutionProcessor {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new convolution processor with an impulse response
   * Uses multi-stage partitioned convolution for optimal performance with long IRs
   */
  constructor(impulse_response: Float32Array, block_size: number);
  /**
   * Process an audio block and return convolved result
   */
  process_block(audio_block: Float32Array): Float32Array;
  /**
   * Reset the processor state (clears all internal buffers)
   */
  reset(): void;
}
/**
 * WebAssembly bindings for impulse response extraction.
 *
 * Extracts impulse responses from recorded sweep responses using deconvolution.
 */
export class WasmIRProcessor {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new IR processor with the original sweep
   */
  constructor(original_sweep: Float32Array);
  /**
   * Extract impulse response with custom settings
   */
  extract_impulse_response_advanced(recorded_response: Float32Array, ir_length: number, regularization: number): Float32Array;
}
/**
 * Stereo convolution processor for WASM
 * Processes left and right channels independently with separate IRs
 */
export class WasmStereoConvolutionProcessor {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new stereo convolution processor with separate L/R impulse responses
   */
  constructor(ir_left: Float32Array, ir_right: Float32Array, block_size: number);
  /**
   * Create a stereo processor from a mono IR (applies same IR to both channels)
   */
  static from_mono(ir: Float32Array, block_size: number): WasmStereoConvolutionProcessor;
  /**
   * Process stereo audio blocks
   * Returns interleaved output (L, R, L, R, ...)
   */
  process_block(input_left: Float32Array, input_right: Float32Array): Float32Array;
  /**
   * Process interleaved stereo audio (L, R, L, R, ...)
   */
  process_interleaved(interleaved: Float32Array): Float32Array;
  /**
   * Reset the processor state
   */
  reset(): void;
}
/**
 * WebAssembly bindings for exponential sweep generation.
 *
 * Use this to generate test signals for impulse response measurement.
 */
export class WasmSweepGenerator {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Generate exponential sweep and return as Float32Array
   */
  static exponential(sample_rate: number, duration: number, start_freq: number, end_freq: number): Float32Array;
}
/**
 * True stereo convolution processor for WASM
 * Uses a 4-channel matrix (LL, LR, RL, RR) for full spatial reverb
 */
export class WasmTrueStereoConvolutionProcessor {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new true stereo convolution processor with 4 IRs
   *
   * - ir_ll: Left input → Left output
   * - ir_lr: Left input → Right output
   * - ir_rl: Right input → Left output
   * - ir_rr: Right input → Right output
   */
  constructor(ir_ll: Float32Array, ir_lr: Float32Array, ir_rl: Float32Array, ir_rr: Float32Array, block_size: number);
  /**
   * Create true stereo from a stereo IR pair with synthetic cross-feed
   *
   * This creates LR and RL channels by attenuating and delaying the opposite channel's IR.
   *
   * - cross_feed_gain: 0.0 to 1.0, typical: 0.3-0.5
   * - cross_feed_delay_samples: typical: 10-50 at 48kHz
   */
  static from_stereo_with_crossfeed(ir_left: Float32Array, ir_right: Float32Array, block_size: number, cross_feed_gain: number, cross_feed_delay_samples: number): WasmTrueStereoConvolutionProcessor;
  /**
   * Create true stereo from a mono IR with synthetic stereo spread
   *
   * - spread: 0.0 = mono, 1.0 = full stereo spread
   */
  static from_mono_with_spread(ir: Float32Array, block_size: number, spread: number): WasmTrueStereoConvolutionProcessor;
  /**
   * Process stereo audio blocks with full matrix convolution
   * Returns interleaved output (L, R, L, R, ...)
   */
  process_block(input_left: Float32Array, input_right: Float32Array): Float32Array;
  /**
   * Process interleaved stereo audio (L, R, L, R, ...)
   */
  process_interleaved(interleaved: Float32Array): Float32Array;
  /**
   * Reset the processor state
   */
  reset(): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasmsweepgenerator_free: (a: number, b: number) => void;
  readonly wasmsweepgenerator_exponential: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly __wbg_wasmirprocessor_free: (a: number, b: number) => void;
  readonly wasmirprocessor_new: (a: any) => number;
  readonly wasmirprocessor_extract_impulse_response_advanced: (a: number, b: any, c: number, d: number) => [number, number, number];
  readonly __wbg_wasmconvolutionprocessor_free: (a: number, b: number) => void;
  readonly wasmconvolutionprocessor_new: (a: any, b: number) => [number, number, number];
  readonly wasmconvolutionprocessor_process_block: (a: number, b: any) => [number, number, number];
  readonly wasmconvolutionprocessor_reset: (a: number) => void;
  readonly convolve_audio: (a: any, b: any) => [number, number, number];
  readonly __wbg_wasmstereoconvolutionprocessor_free: (a: number, b: number) => void;
  readonly wasmstereoconvolutionprocessor_new: (a: any, b: any, c: number) => [number, number, number];
  readonly wasmstereoconvolutionprocessor_from_mono: (a: any, b: number) => [number, number, number];
  readonly wasmstereoconvolutionprocessor_process_block: (a: number, b: any, c: any) => [number, number, number];
  readonly wasmstereoconvolutionprocessor_process_interleaved: (a: number, b: any) => [number, number, number];
  readonly wasmstereoconvolutionprocessor_reset: (a: number) => void;
  readonly __wbg_wasmtruestereoconvolutionprocessor_free: (a: number, b: number) => void;
  readonly wasmtruestereoconvolutionprocessor_new: (a: any, b: any, c: any, d: any, e: number) => [number, number, number];
  readonly wasmtruestereoconvolutionprocessor_from_stereo_with_crossfeed: (a: any, b: any, c: number, d: number, e: number) => [number, number, number];
  readonly wasmtruestereoconvolutionprocessor_from_mono_with_spread: (a: any, b: number, c: number) => [number, number, number];
  readonly wasmtruestereoconvolutionprocessor_process_block: (a: number, b: any, c: any) => [number, number, number];
  readonly wasmtruestereoconvolutionprocessor_process_interleaved: (a: number, b: any) => [number, number, number];
  readonly wasmtruestereoconvolutionprocessor_reset: (a: number) => void;
  readonly stereo_convolve_audio: (a: any, b: any, c: any, d: any) => [number, number, number];
  readonly __wbindgen_export_0: WebAssembly.Table;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
