let wasm;

let cachedFloat32ArrayMemory0 = null;

function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_export_0.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}
/**
 * Apply convolution reverb to audio data.
 *
 * Convenience function for one-shot (non-realtime) convolution.
 * @param {Float32Array} audio
 * @param {Float32Array} impulse_response
 * @returns {Float32Array}
 */
export function convolve_audio(audio, impulse_response) {
    const ret = wasm.convolve_audio(audio, impulse_response);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Batch stereo convolution (non-real-time)
 * @param {Float32Array} left
 * @param {Float32Array} right
 * @param {Float32Array} ir_left
 * @param {Float32Array} ir_right
 * @returns {Float32Array}
 */
export function stereo_convolve_audio(left, right, ir_left, ir_right) {
    const ret = wasm.stereo_convolve_audio(left, right, ir_left, ir_right);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

const WasmConvolutionProcessorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmconvolutionprocessor_free(ptr >>> 0, 1));
/**
 * WebAssembly bindings for real-time partitioned convolution.
 *
 * Provides block-based convolution processing suitable for real-time audio.
 */
export class WasmConvolutionProcessor {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmConvolutionProcessorFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmconvolutionprocessor_free(ptr, 0);
    }
    /**
     * Create a new convolution processor with an impulse response
     * Uses multi-stage partitioned convolution for optimal performance with long IRs
     * @param {Float32Array} impulse_response
     * @param {number} block_size
     */
    constructor(impulse_response, block_size) {
        const ret = wasm.wasmconvolutionprocessor_new(impulse_response, block_size);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmConvolutionProcessorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Process an audio block and return convolved result
     * @param {Float32Array} audio_block
     * @returns {Float32Array}
     */
    process_block(audio_block) {
        const ret = wasm.wasmconvolutionprocessor_process_block(this.__wbg_ptr, audio_block);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Reset the processor state (clears all internal buffers)
     */
    reset() {
        wasm.wasmconvolutionprocessor_reset(this.__wbg_ptr);
    }
}
if (Symbol.dispose) WasmConvolutionProcessor.prototype[Symbol.dispose] = WasmConvolutionProcessor.prototype.free;

const WasmIRProcessorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmirprocessor_free(ptr >>> 0, 1));
/**
 * WebAssembly bindings for impulse response extraction.
 *
 * Extracts impulse responses from recorded sweep responses using deconvolution.
 */
export class WasmIRProcessor {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmIRProcessorFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmirprocessor_free(ptr, 0);
    }
    /**
     * Create a new IR processor with the original sweep
     * @param {Float32Array} original_sweep
     */
    constructor(original_sweep) {
        const ret = wasm.wasmirprocessor_new(original_sweep);
        this.__wbg_ptr = ret >>> 0;
        WasmIRProcessorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Extract impulse response with custom settings
     * @param {Float32Array} recorded_response
     * @param {number} ir_length
     * @param {number} regularization
     * @returns {Float32Array}
     */
    extract_impulse_response_advanced(recorded_response, ir_length, regularization) {
        const ret = wasm.wasmirprocessor_extract_impulse_response_advanced(this.__wbg_ptr, recorded_response, ir_length, regularization);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) WasmIRProcessor.prototype[Symbol.dispose] = WasmIRProcessor.prototype.free;

const WasmStereoConvolutionProcessorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmstereoconvolutionprocessor_free(ptr >>> 0, 1));
/**
 * Stereo convolution processor for WASM
 * Processes left and right channels independently with separate IRs
 */
export class WasmStereoConvolutionProcessor {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmStereoConvolutionProcessor.prototype);
        obj.__wbg_ptr = ptr;
        WasmStereoConvolutionProcessorFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmStereoConvolutionProcessorFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmstereoconvolutionprocessor_free(ptr, 0);
    }
    /**
     * Create a new stereo convolution processor with separate L/R impulse responses
     * @param {Float32Array} ir_left
     * @param {Float32Array} ir_right
     * @param {number} block_size
     */
    constructor(ir_left, ir_right, block_size) {
        const ret = wasm.wasmstereoconvolutionprocessor_new(ir_left, ir_right, block_size);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmStereoConvolutionProcessorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create a stereo processor from a mono IR (applies same IR to both channels)
     * @param {Float32Array} ir
     * @param {number} block_size
     * @returns {WasmStereoConvolutionProcessor}
     */
    static from_mono(ir, block_size) {
        const ret = wasm.wasmstereoconvolutionprocessor_from_mono(ir, block_size);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmStereoConvolutionProcessor.__wrap(ret[0]);
    }
    /**
     * Process stereo audio blocks
     * Returns interleaved output (L, R, L, R, ...)
     * @param {Float32Array} input_left
     * @param {Float32Array} input_right
     * @returns {Float32Array}
     */
    process_block(input_left, input_right) {
        const ret = wasm.wasmstereoconvolutionprocessor_process_block(this.__wbg_ptr, input_left, input_right);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Process interleaved stereo audio (L, R, L, R, ...)
     * @param {Float32Array} interleaved
     * @returns {Float32Array}
     */
    process_interleaved(interleaved) {
        const ret = wasm.wasmstereoconvolutionprocessor_process_interleaved(this.__wbg_ptr, interleaved);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Reset the processor state
     */
    reset() {
        wasm.wasmstereoconvolutionprocessor_reset(this.__wbg_ptr);
    }
}
if (Symbol.dispose) WasmStereoConvolutionProcessor.prototype[Symbol.dispose] = WasmStereoConvolutionProcessor.prototype.free;

const WasmSweepGeneratorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmsweepgenerator_free(ptr >>> 0, 1));
/**
 * WebAssembly bindings for exponential sweep generation.
 *
 * Use this to generate test signals for impulse response measurement.
 */
export class WasmSweepGenerator {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmSweepGeneratorFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmsweepgenerator_free(ptr, 0);
    }
    /**
     * Generate exponential sweep and return as Float32Array
     * @param {number} sample_rate
     * @param {number} duration
     * @param {number} start_freq
     * @param {number} end_freq
     * @returns {Float32Array}
     */
    static exponential(sample_rate, duration, start_freq, end_freq) {
        const ret = wasm.wasmsweepgenerator_exponential(sample_rate, duration, start_freq, end_freq);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) WasmSweepGenerator.prototype[Symbol.dispose] = WasmSweepGenerator.prototype.free;

const WasmTrueStereoConvolutionProcessorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmtruestereoconvolutionprocessor_free(ptr >>> 0, 1));
/**
 * True stereo convolution processor for WASM
 * Uses a 4-channel matrix (LL, LR, RL, RR) for full spatial reverb
 */
export class WasmTrueStereoConvolutionProcessor {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmTrueStereoConvolutionProcessor.prototype);
        obj.__wbg_ptr = ptr;
        WasmTrueStereoConvolutionProcessorFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmTrueStereoConvolutionProcessorFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmtruestereoconvolutionprocessor_free(ptr, 0);
    }
    /**
     * Create a new true stereo convolution processor with 4 IRs
     *
     * - ir_ll: Left input → Left output
     * - ir_lr: Left input → Right output
     * - ir_rl: Right input → Left output
     * - ir_rr: Right input → Right output
     * @param {Float32Array} ir_ll
     * @param {Float32Array} ir_lr
     * @param {Float32Array} ir_rl
     * @param {Float32Array} ir_rr
     * @param {number} block_size
     */
    constructor(ir_ll, ir_lr, ir_rl, ir_rr, block_size) {
        const ret = wasm.wasmtruestereoconvolutionprocessor_new(ir_ll, ir_lr, ir_rl, ir_rr, block_size);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmTrueStereoConvolutionProcessorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create true stereo from a stereo IR pair with synthetic cross-feed
     *
     * This creates LR and RL channels by attenuating and delaying the opposite channel's IR.
     *
     * - cross_feed_gain: 0.0 to 1.0, typical: 0.3-0.5
     * - cross_feed_delay_samples: typical: 10-50 at 48kHz
     * @param {Float32Array} ir_left
     * @param {Float32Array} ir_right
     * @param {number} block_size
     * @param {number} cross_feed_gain
     * @param {number} cross_feed_delay_samples
     * @returns {WasmTrueStereoConvolutionProcessor}
     */
    static from_stereo_with_crossfeed(ir_left, ir_right, block_size, cross_feed_gain, cross_feed_delay_samples) {
        const ret = wasm.wasmtruestereoconvolutionprocessor_from_stereo_with_crossfeed(ir_left, ir_right, block_size, cross_feed_gain, cross_feed_delay_samples);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTrueStereoConvolutionProcessor.__wrap(ret[0]);
    }
    /**
     * Create true stereo from a mono IR with synthetic stereo spread
     *
     * - spread: 0.0 = mono, 1.0 = full stereo spread
     * @param {Float32Array} ir
     * @param {number} block_size
     * @param {number} spread
     * @returns {WasmTrueStereoConvolutionProcessor}
     */
    static from_mono_with_spread(ir, block_size, spread) {
        const ret = wasm.wasmtruestereoconvolutionprocessor_from_mono_with_spread(ir, block_size, spread);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTrueStereoConvolutionProcessor.__wrap(ret[0]);
    }
    /**
     * Process stereo audio blocks with full matrix convolution
     * Returns interleaved output (L, R, L, R, ...)
     * @param {Float32Array} input_left
     * @param {Float32Array} input_right
     * @returns {Float32Array}
     */
    process_block(input_left, input_right) {
        const ret = wasm.wasmtruestereoconvolutionprocessor_process_block(this.__wbg_ptr, input_left, input_right);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Process interleaved stereo audio (L, R, L, R, ...)
     * @param {Float32Array} interleaved
     * @returns {Float32Array}
     */
    process_interleaved(interleaved) {
        const ret = wasm.wasmtruestereoconvolutionprocessor_process_interleaved(this.__wbg_ptr, interleaved);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Reset the processor state
     */
    reset() {
        wasm.wasmtruestereoconvolutionprocessor_reset(this.__wbg_ptr);
    }
}
if (Symbol.dispose) WasmTrueStereoConvolutionProcessor.prototype[Symbol.dispose] = WasmTrueStereoConvolutionProcessor.prototype.free;

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg_length_a8cca01d07ea9653 = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_newfromslice_eb3df67955925a7c = function(arg0, arg1) {
        const ret = new Float32Array(getArrayF32FromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbg_prototypesetcall_5521f1dd01df76fd = function(arg0, arg1, arg2) {
        Float32Array.prototype.set.call(getArrayF32FromWasm0(arg0, arg1), arg2);
    };
    imports.wbg.__wbg_wbindgenthrow_451ec1a8469d7eb6 = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(String) -> Externref`.
        const ret = getStringFromWasm0(arg0, arg1);
        return ret;
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_export_0;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
        ;
    };

    return imports;
}

function __wbg_init_memory(imports, memory) {

}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();

    __wbg_init_memory(imports);

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('rconvolve_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    __wbg_init_memory(imports);

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
