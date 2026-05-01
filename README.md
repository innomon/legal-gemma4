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
- **GoMLX Backends:**
  - **Mac M4:** [go-darwinml](https://github.com/gomlx/go-darwinml) for Metal acceleration.
  - **Raspberry Pi 5:** Default GoMLX XLA/CPU backend.
- **libtokenizers:** Required for the `daulet/tokenizers` bindings.

#### Installing libtokenizers

The project requires `libtokenizers` to be available for CGO linking.

**For Raspberry Pi 5 (Linux ARM64):**
```bash
wget https://github.com/daulet/tokenizers/releases/download/v1.27.0/libtokenizers.linux-arm64.tar.gz
tar -xzf libtokenizers.linux-arm64.tar.gz
sudo cp libtokenizers.a /usr/local/lib/
# Header is in the Go module cache
sudo cp $(go env GOPATH)/pkg/mod/github.com/daulet/tokenizers@v1.27.0/tokenizers.h /usr/local/include/
```

**For Mac M4 (macOS ARM64):**
```bash
curl -LO https://github.com/daulet/tokenizers/releases/download/v1.27.0/libtokenizers.darwin-arm64.tar.gz
tar -xzf libtokenizers.darwin-arm64.tar.gz
sudo cp libtokenizers.a /usr/local/lib/
# Header is in the Go module cache
sudo cp $(go env GOPATH)/pkg/mod/github.com/daulet/tokenizers@v1.27.0/tokenizers.h /usr/local/include/
```

**Environment Setup:**
Ensure `CGO_ENABLED=1` is set when building or running. If you installed the library in a custom location, set:
```bash
export CGO_LDFLAGS="-L/path/to/lib"
export CGO_CFLAGS="-I/path/to/include"
```

### Data Preparation

Download the `viber1/indian-law-dataset` from Hugging Face:

```bash
go run cmd/data/main.go download --output indian_law.jsonl
```

## Quick Start


### Mac M4 (Apple Silicon)
Optimized for Metal acceleration using `go-darwinml`.

**Build:**
```bash
CGO_ENABLED=1 go build -o legal-gemma4 ./cmd/train
```

**Run Training:**
```bash
CGO_ENABLED=1 ./legal-gemma4 \
    --model quantized_model \
    --dataset indian_law.jsonl \
    --tokenizer tokenizer.json
```

### Raspberry Pi 5 (Linux ARM64)
Uses the default XLA CPU backend. Note: Training on Pi 5 is significantly slower than on M4.

**Build:**
```bash
CGO_ENABLED=1 go build -o legal-gemma4 ./cmd/train
```

**Run Training:**
```bash
CGO_ENABLED=1 ./legal-gemma4 \
    --model quantized_model \
    --dataset indian_law.jsonl \
    --tokenizer tokenizer.json
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

## OpenAI API Server

The project includes an OpenAI-compatible API server to serve the model for chat completions.

### Build the Server
```bash
CGO_ENABLED=1 go build -o api ./cmd/api/main.go
```

### Run the Server
```bash
CGO_ENABLED=1 ./api serve \
    --model quantized_model \
    --tokenizer tokenizer.json \
    --port 8080
```

### Verification

**List Models:**
```bash
curl http://localhost:8080/v1/models
```

**Chat Completion:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "legal-gemma-4-e2b",
    "messages": [
      {"role": "user", "content": "What is the penalty for perjury under Indian law?"}
    ]
  }'
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
