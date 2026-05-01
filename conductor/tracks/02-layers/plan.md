# Implementation Plan: QLoRA & PLE Layers

## Phase 1: Core QLoRA Layer
- [ ] Implement `QLoRALinear` struct with frozen weight pointers.
- [ ] Implement bitwise unpacking in the GoMLX graph.
- [ ] Implement NF4 dequantization logic in the graph.

## Phase 2: LoRA & Scaling
- [ ] Add trainable `LoRA_A` (initialized Gaussian) and `LoRA_B` (initialized Zero).
- [ ] Implement scaling factor $(\alpha / r)$.

## Phase 3: PLE Integration
- [ ] Implement PLE residual addition logic.
- [ ] Ensure layer indexing is correctly handled during model construction.

## Phase 4: Integration Test
- [ ] Verify forward pass with mock inputs.
- [ ] Check gradient flow to LoRA matrices.
