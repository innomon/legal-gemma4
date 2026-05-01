# Implementation Plan: Evaluation & Export

## Phase 1: Benchmark Implementation
- [ ] Collate AIBE MCQ questions.
- [ ] Implement evaluation harness (prompting + answer parsing).
- [ ] Record results and compare against passing grade (40%).

## Phase 2: Model Merging
- [ ] Implement dequantization of full base model.
- [ ] Add LoRA adapter contributions.
- [ ] Validate merged model weights against original model behavior.

## Phase 3: Safetensor Export
- [ ] Implement safetensor header generation.
- [ ] Write tensor buffers to disk.
- [ ] Verify export with standard tools (e.g., Python `safetensors` library).
