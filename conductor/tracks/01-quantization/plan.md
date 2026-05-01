# Implementation Plan: NF4 Quantization

## Phase 1: NF4 Foundation
- [ ] Define the 16 NF4 constant values.
- [ ] Implement `quantize_block` function.
- [ ] Implement bit-packing logic (2x4-bit to 1x8-bit).

## Phase 2: Weight Conversion
- [ ] Script to read Gemma-4-E2B `.safetensors`.
- [ ] Perform per-layer quantization.
- [ ] Save output as GoMLX `.tensor` or custom binary format.

## Phase 3: Verification
- [ ] Implement a unit test to quantize and dequantize a random Gaussian tensor.
- [ ] Measure Mean Squared Error (MSE) to ensure fidelity.
