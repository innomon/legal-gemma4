# Track: NF4 Quantization & Setup

Implement the 4-bit NormalFloat4 (NF4) quantization logic and convert base Gemma-4-E2B weights.

## Objectives
- Implement NF4 ROM lookup table.
- Develop weight blocking and `absmax` calculation.
- Implement bit-packing for memory efficiency.
- Convert `.safetensors` to GoMLX-compatible `.tensor` format.

## Related Documents
- [Implementation Plan](./plan.md)
- [Specification](./spec.md)
