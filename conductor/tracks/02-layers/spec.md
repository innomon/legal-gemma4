# Specification: QLoRA & PLE Layers

## QLoRA Forward Pass
The layer must implement:
1. **Frozen Path:** Unpack `uint8`, dequantize via NF4 LUT, scale by `absmax`, and perform dot product with input.
2. **LoRA Path:** Small trainable matrices $A$ and $B$. $Y_{lora} = (X \cdot A \cdot B) \cdot \frac{\alpha}{r}$.
3. **Combination:** $Y = Y_{frozen} + Y_{lora}$.

## Per-Layer Embeddings (PLE)
Gemma 4 specific:
- A unique residual embedding signal must be added at every transformer block.
- Implementation must track the layer index to apply the correct PLE signal.

## Technical Requirements
- Use GoMLX `ml.Context` for managing LoRA weights.
- Optimize the unpacking logic using bitwise operations (`ml.BitwiseAnd`, `ml.ShiftRight`).
