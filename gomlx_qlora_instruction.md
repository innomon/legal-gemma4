# Instruction for QLoRA Implementation and Finetuning (GoMLX)

## Context & Objectives
Implement a custom **QLoRA Linear Layer** from scratch in **GoMLX** without external dependencies. The goal is to finetune the **Gemma-4-E2B** model on an **M4 Mac (16GB RAM)** using the `viber1/indian-law-dataset` to reach a performance level capable of passing the **All India Bar Examination (AIBE)**.

---

## 1. Core Architecture: QLoRA + Gemma 4
### The QLoRA Equation
The forward pass for a QLoRA Linear Layer is defined as:
$$Y = (\text{Dequantize}(W_{4bit}) \cdot X) + (X \cdot A \cdot B) \cdot \frac{\alpha}{r}$$

### Gemma 4 Specifics: Per-Layer Embeddings (PLE)
Gemma 4 uses **PLE**, meaning a residual embedding signal must be added at every transformer block.
- **Base Model Choice:** `Gemma-4-E2B` (2.3B active parameters).
- **RAM Footprint:** ~1.3GB for 4-bit weights, leaving ~10GB for gradients and KV cache.

---

## 2. Implementation Roadmap

### Phase 1: Quantization (One-Time Setup)
1. **NF4 Lookup Table:** Implement the 16-value NormalFloat4 ROM.
2. **Quantization Logic:** - Reshape weights into blocks (e.g., 64 elements).
    - Calculate `absmax` per block.
    - Map FP16/BF16 weights to the closest NF4 index.
    - **Bit-Packing:** Pack two 4-bit indices into a single `uint8` tensor to save memory.
3. **Save to Disk:** Convert `.safetensors` to GoMLX `.tensor` files and save the quantized version to avoid re-calculating during training.

### Phase 2: The GoMLX QLoRA Layer
Implement a struct that handles the dual-path forward pass:
- **Frozen Path:** Use `ml.BitwiseAnd` and `ml.ShiftRight` to unpack `uint8` weights, gather from the NF4 LUT, multiply by `absmax`, and perform `ml.Dot` with input.
- **LoRA Path:** Implement trainable `LoRA_A` and `LoRA_B` matrices.
- **PLE Integration:** Add the constant `PLE_signal` unique to each layer index.

### Phase 3: Finetuning on Indian Law
- **Dataset:** `viber1/indian-law-dataset`.
- **Formatting:** Use an "Advocate" prompt: `### Question: [Q] \n### Advocate Analysis: [A]`.
- **Hyperparameters:**
    - **Rank (r):** 8 or 16.
    - **Alpha:** 32.
    - **Target Context:** 2048 - 4096 tokens (to fit in 16GB RAM).
    - **Epochs:** 3-5 for deep domain memorization.

### Phase 4: Merging & Export
1. **Dequantize & Add:** Load frozen weights and LoRA adapters. Compute $W_{merged} = W_{base} + (A \cdot B) \times \text{scale}$.
2. **Safetensor Export:** Write a custom exporter to save the merged weights in `F16` or `BF16` back to `.safetensors` format.

---

## 3. Evaluation: The Bar Exam
- **Benchmark:** All India Bar Examination (AIBE) previous year papers (AIBE XVII - XX).
- **Metric:** Zero-shot accuracy on Multiple Choice Questions (MCQs).
- **Passing Grade:** Aim for >55% accuracy (AIBE pass mark is 40%).

---

## Technical Constraints for M4 (16GB) / Rpi5 (16GB)
- **Backend:** Use GoMLX `go-darwinml` for M4 and `go` for Rpi5
- **Memory Management:** Avoid loading full FP32 models; use lazy loading and disk-based checkpoints.
- **Batch Size:** 1-4 depending on context length.

**Code Reference** you can refer code from `/home/innomon/orez/llm/go-turboquant` where Gemma4 is implemented. **Note** it has turbo quant and MTP, we do not want to use them in this project.

**Remember** for go cammands and registry create handcrafted code, don't use cobra/spf13 or similar libs


