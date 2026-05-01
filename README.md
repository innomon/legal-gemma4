# Legal-Gemma4: LLM Finetuning for Indian Law

Legal-Gemma4 is a Go-native, memory-efficient implementation of the **Gemma-4-E2B** model architecture, optimized for finetuning on Indian Law datasets. It features a handcrafted **QLoRA** implementation and **NF4 quantization** to enable high-performance training on consumer hardware like the Mac M4 (16GB RAM).

The primary goal of this project is to achieve a performance level capable of passing the **All India Bar Examination (AIBE)** using the `viber1/indian-law-dataset`.

## Key Features

- **Go-Native QLoRA:** Handcrafted Linear layers with dual-path forward passes (frozen 4-bit weights + trainable LoRA adapters).
- **NF4 Quantization:** Information-theoretically optimal 4-bit NormalFloat quantization with block-based scaling.
- **Gemma 4 Compliance:** Full support for **Per-Layer Embeddings (PLE)** residual signals at every transformer block.
- **Memory Efficient:** Designed for 16GB RAM constraints using bit-packing and lazy weight management.
- **Safetensors Support:** Handcrafted reader and writer for seamless integration with Hugging Face model formats.
- **No Heavy Dependencies:** Built on [GoMLX](https://github.com/gomlx/gomlx) without reliance on complex Python environments or large C++ binaries.

## Architecture

- **Base Model:** Gemma-4-E2B (2.3B parameters).
- **Quantization:** NF4 with 64-element block size and `absmax` scaling.
- **LoRA Hyperparameters:** Rank ($r$) 8/16, Alpha ($\alpha$) 32.
- **Equation:** $Y = (\text{Dequantize}(W_{4bit}) \cdot X) + (X \cdot A \cdot B) \cdot \frac{\alpha}{r}$

## Getting Started

### Prerequisites

- Go 1.25 or later.
- [GoMLX](https://github.com/gomlx/gomlx) installed with the `go-darwinml` backend (for Mac M4 acceleration).

```bash
go mod tidy
```

### 1. Quantization

Prepare the base Gemma-4 weights by converting them from `.safetensors` to the quantized GoMLX format:

```bash
go run cmd/quantize/main.go --input path/to/model.safetensors --output quantized_model
```

### 2. Training

Finetune the model on the Indian Law dataset using the "Advocate" prompt format:

```bash
go run cmd/train/main.go \
    --model quantized_model \
    --dataset indian_law.jsonl \
    --tokenizer tokenizer.json \
    --epochs 3 \
    --batch_size 1
```

### 3. Model Merging & Export

Merge the trained LoRA adapters back into the base weights and export to a standard `.safetensors` file:

```bash
go run cmd/export/main.go --model quantized_model --output legal-gemma4-final.safetensors
```

## Evaluation (AIBE Benchmark)

The model is evaluated using zero-shot MCQ accuracy on past papers from the All India Bar Examination (AIBE XVII - XX).

- **Target Accuracy:** >55%
- **Passing Grade:** 40%

## Project Structure

- `pkg/quant`: NF4 quantization and bit-packing logic.
- `pkg/layers`: Custom QLoRA, RMSNorm, RoPE, and Attention implementations.
- `pkg/model`: Full Gemma 4 architecture and weight loader.
- `pkg/safetensors`: Handcrafted Safetensors IO.
- `pkg/data`: Dataset loading and tokenization.
- `cmd/`: CLI tools for quantization, training, and export.

## License

MIT
