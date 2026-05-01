# Specification: Evaluation & Export

## AIBE Benchmark
- **Goal:** Reach >55% accuracy on Multiple Choice Questions.
- **Data:** Previous year papers (AIBE XVII - XX).
- **Method:** Zero-shot prompting with multiple-choice answer extraction.

## Model Merging
- **Logic:** $W_{merged} = W_{base} + (A \cdot B) \times \text{scale}$.
- **Precision:** Convert back to FP16 or BF16 for final export.

## Export Format
- **Format:** `.safetensors`.
- **Custom Exporter:** Handcrafted logic to write weights into the safetensors file structure.
