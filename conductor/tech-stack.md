# Tech Stack: Legal-Gemma4

## Core Technologies
- **Language:** Go (Golang)
- **Deep Learning Framework:** [GoMLX](https://github.com/gomlx/gomlx)
- **Acceleration Backend:** `go-darwinml` (Apple Silicon), `go` (CPU/Generic)
- **Model:** Gemma-4-E2B (2.3B parameters)

## Dataset & Benchmarks
- **Training Data:** `viber1/indian-law-dataset` (Hugging Face)
- **Evaluation:** AIBE XVII - XX past papers.

## Hardware Targets
- **Primary:** M4 Mac with 16GB RAM.
- **Secondary:** Raspberry Pi 5 with 16GB RAM.
