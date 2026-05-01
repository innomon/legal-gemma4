# Implementation Plan: Training & Dataset

## Phase 1: Data Preparation
- [ ] Implement tokenizer interface (SentencePiece/Gemma-4 compatible).
- [ ] Format dataset into "Advocate" prompts.
- [ ] Create data loader with shuffling and batching.

## Phase 2: Finetuning Loop
- [ ] Initialize model with quantized weights and LoRA adapters.
- [ ] Implement cross-entropy loss function.
- [ ] Implement training loop with periodic checkpointing.

## Phase 3: Monitoring
- [ ] Track loss and accuracy (if validation set exists).
- [ ] Log performance to console or file.
