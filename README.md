# rconvolve

[![Crates.io](https://img.shields.io/crates/v/rconvolve.svg)](https://crates.io/crates/rconvolve)
[![Docs.rs](https://docs.rs/rconvolve/badge.svg)](https://docs.rs/rconvolve)

Fast convolution and impulse response extraction for audio applications in Rust.

## Overview

`rconvolve` provides FFT-based convolution and deconvolution for audio processing. It supports:

- Batch and real-time convolution for applying impulse responses to audio
- Exponential sine sweep generation for acoustic measurement
- Impulse response extraction from recorded sweeps via deconvolution
- Mono, stereo, and true stereo (4-channel matrix) processing
- WebAssembly support for browser-based applications
- `no_std` compatibility (requires `alloc`) for embedded systems

**Live Demo**: Try the WebAssembly demo at [rconvolve.pages.dev](https://rconvolve.pages.dev/)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rconvolve = "*"
```

### Feature Flags

| Feature | Default | Description                                                  |
| ------- | ------- | ------------------------------------------------------------ |
| `std`   | Yes     | Enables standard library. Disable for `no_std` environments. |
| `wasm`  | No      | Enables WebAssembly bindings via `wasm-bindgen`.             |

## Quick Start

### Batch Convolution

See the example in the [rconvolve::convolve module documentation](https://docs.rs/rconvolve/latest/rconvolve/convolve/index.html).

### Real-Time Convolution

See the example in the [rconvolve::convolve::PartitionedConvolution documentation](https://docs.rs/rconvolve/latest/rconvolve/convolve/struct.PartitionedConvolution.html).

### Impulse Response Extraction

See the example in the [rconvolve::deconvolve module documentation](https://docs.rs/rconvolve/latest/rconvolve/deconvolve/index.html).

## Examples

The crate includes some examples:

### apply_reverb

Apply convolution reverb to a WAV file:

```bash
cargo run --example apply_reverb -- input.wav impulse_response.wav output.wav
```

### realtime_reverb

Real-time convolution reverb using system audio input/output:

```bash
cargo run --example realtime_reverb -- impulse_response.wav
cargo run --example realtime_reverb -- --test-ir  # Use generated test IR
```

### simple_wav_to_ir

Convert a recorded sweep to an impulse response:

```bash
cargo run --example simple_wav_to_ir -- recorded_sweep.wav output_ir.wav
```

## WebAssembly

```bash
cargo install wasm-pack
wasm-pack build --target web --release
```

See the [full documentation](https://docs.rs/rconvolve) for the WASM API reference, or try the [live demo](https://rconvolve.pages.dev/).

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
