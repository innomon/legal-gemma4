# Specification: NF4 Quantization

## NormalFloat4 (NF4) Definition
NF4 is an information-theoretically optimal 4-bit quantization format for normally distributed weights.
- **Lookup Table:** 16 values derived from the quantile of a Gaussian distribution.
- **Block Size:** 64 elements per quantization block.
- **Scaling:** One `absmax` (FP16/BF16) per block.

## Functional Requirements
1. **NF4 LUT:** Static array of 16 float values.
2. **Quantizer:** Function that takes an FP16/BF16 tensor and returns a `uint8` packed tensor and an `absmax` tensor.
3. **Dequantizer:** Function (GoMLX Graph) that unpacks `uint8` and gathers from LUT, then scales by `absmax`.
4. **Storage:** Save packed weights to disk to minimize RAM usage during initialization.
