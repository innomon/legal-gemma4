# Workflow: Legal-Gemma4

## Development Process
1. **Offline Quantization:** Pre-quantize weights to NF4 and save to disk.
2. **Modular Layer Implementation:** Test QLoRA layers in isolation.
3. **Training Cycle:**
    - Load quantized weights.
    - Attach LoRA adapters.
    - Train on `indian-law-dataset`.
    - Periodic checkpointing of adapters only.
4. **Benchmarking:** Zero-shot MCQ evaluation.
5. **Deployment:** Merge adapters and export to `.safetensors`.

## CLI Interface
- Handcrafted command registry (no Cobra).
- Commands for: `quantize`, `train`, `eval`, `export`.
