# Specification: Training & Dataset

## Dataset
- **Source:** `viber1/indian-law-dataset`.
- **Format:** Advocate Analysis prompt.
    - `### Question: [Q] \n### Advocate Analysis: [A]`

## Training Configuration
- **Rank (r):** 8 or 16.
- **Alpha:** 32.
- **Context Length:** 2048 - 4096 tokens.
- **Batch Size:** 1 - 4.
- **Optimizer:** AdamW with linear warmup and decay.
- **Backend:** `go-darwinml` (M4 acceleration).

## Memory Management
- Checkpoint saving to disk.
- Lazy weight loading.
- Gradient accumulation (if needed to reach effective batch size).
