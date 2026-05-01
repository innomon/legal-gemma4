# Legal-Gemma4 Project Instructions

## Project Overview
This project implements a custom **QLoRA Linear Layer** in **GoMLX** for finetuning the **Gemma-4-E2B** model on the `viber1/indian-law-dataset`. The objective is to achieve a performance level capable of passing the **All India Bar Examination (AIBE)**.

## Core Mandates
- **No External Dependencies:** The QLoRA implementation must be handcrafted in GoMLX.
- **Memory Efficiency:** Optimize for M4 Mac with 16GB RAM. Use NF4 quantization (4-bit) and lazy loading.
- **Gemma 4 Architecture:** Must support **Per-Layer Embeddings (PLE)** residual signals at every transformer block.
- **CLI Standard:** NEVER use `spf13/cobra` or `pflag`. Use a handcrafted command registry for all CLI operations.
- **Backend:** Use `go-darwinml` for M4 Mac acceleration.

## Technical Architecture
- **Quantization:** NF4 (NormalFloat4) lookup table with 64-element block size and `absmax` scaling.
- **Bit-Packing:** Pack two 4-bit indices into `uint8` tensors.
- **QLoRA Equation:** $Y = (\text{Dequantize}(W_{4bit}) \cdot X) + (X \cdot A \cdot B) \cdot \frac{\alpha}{r}$
- **LoRA Hyperparameters:** Rank ($r$) 8 or 16, Alpha ($\alpha$) 32.

## Development Workflow
1. **Research & Quantization:** Implement NF4 logic and convert base weights.
2. **Layer Implementation:** Develop the `QLoRALinear` struct and PLE integration.
3. **Training:** Finetune on Indian Law dataset with "Advocate" prompt format.
4. **Validation:** Zero-shot MCQ evaluation on AIBE past papers.

## Coding Conventions
- **GoMLX Idioms:** Use `ml.Context` for state management and `ml.Graph` for computations.
- **Errors:** Handle all errors explicitly; no panics in library code.
- **Documentation:** Use `godoc-canonical` style for all public symbols.
