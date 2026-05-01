# Project Blueprint: PLE-RAG Finetuning on Gemma-4-E2B
**Target Hardware:** Mac M4 (16GB RAM)
**Framework:** GoMLX (XLA/Metal)
**Domain:** Indian Law (All India Bar Examination - AIBE)

---

## 1. Architectural Innovation: PLE-RAG
Instead of standard text-based RAG, this architecture internalizes the retrieval process into the **Per-Layer Embedding (PLE)** path of Gemma-4-E2B.

### The Concept
- **Latent Retrieval:** The first hidden state ($h_0$) of the query is used as a search vector.
- **Asymmetric Lookup:** The query searches against a pre-computed **Instruction Vector Space** ($K_{inst}$).
- **Latent Injection:** The top-N corresponding **Response Vectors** ($K_{resp}$) are retrieved and injected into the transformer's decoder layers via the PLE residual path.

---

## 2. Implementation Roadmap

### Phase 1: Pre-processing & Quantization
1. **Model Conversion:** Convert `Gemma-4-E2B` Safetensors to GoMLX `.tensor` files.
2. **4-bit Quantization (NF4):** - Implement NF4 ROM and block-wise quantization.
   - Pack two 4-bit indices into `uint8` for disk storage (~1.3GB footprint).
3. **KB Encoding:** Use the base model to encode the `viber1/indian-law-dataset` into two parallel vector tables: `Instruction` and `Response`.

### Phase 2: Custom GoMLX QLoRA + PLE-RAG Layer
Implement a layer that performs:
1. **Unpacking:** Bitwise operations to retrieve 4-bit weights from `uint8`.
2. **Dequantization:** NF4 LUT gather + AbsMax scaling.
3. **LoRA Path:** Trainable $A$ and $B$ adapters for task-specific adaptation.
4. **RAG Path:**
   - $Similarity = h_0 \cdot K_{inst}^T$
   - $Weights = Softmax(Similarity / Temperature)$
   - $Context = Weights \cdot K_{resp}$
   - $Output = (W_{frozen} \cdot X) + (LoRA_{adapter} \cdot X) + Context$

---

## 3. Training Strategy: Dual-Task Loss
To ensure the model both **finds** the right law and **uses** it correctly, we optimize a combined loss function:

$$L_{total} = L_{generation} + \lambda L_{retrieval}$$

- **Generation Loss ($L_{gen}$):** Standard Cross-Entropy for next-token prediction on the legal response.
- **Retrieval Loss ($L_{ret}$):** Contrastive InfoNCE loss ensuring the query $h_0$ aligns with the correct instruction vector in the batch.

### Hyperparameters for 16GB M4
- **Quantization:** 4-bit NF4 (Frozen Base).
- **Rank (r):** 16.
- **Context Length:** 2048 - 4096 tokens.
- **Retrieval Lambda ($\lambda$):** 0.1.
- **Softmax Temperature:** 0.1 (for "sharp" retrieval).

---

## 4. Final Integration & Evaluation
1. **Weight Merging:** Merge LoRA adapters back into the base weights ($W_{new} = W_{base} + AB$) using a temporary dequantization buffer.
2. **Export:** Save as `.safetensors` in `F16/BF16` precision.
3. **The Bar Exam Test:**
   - **Dataset:** AIBE (All India Bar Examination) past papers.
   - **Procedure:** Zero-shot MCQ evaluation.
   - **Target:** >55% accuracy (Pass mark 40%).

---

## 5. Summary of Memory Efficiency
| Component | Technique | Memory Savings |
| :--- | :--- | :--- |
| **Weights** | 4-bit NF4 Quantization | ~8x reduction |
| **Gradients** | QLoRA + PLE Adapters | Only <5% of params trainable |
| **Context** | Latent RAG (Internalized) | 0 context-token overhead for KB |
| **Computation** | XLA Fusion + Unified Memory | High-speed on M4 GPU/NPU |
