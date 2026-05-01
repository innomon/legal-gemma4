package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/innomon/legal-gemma4/pkg/quant"
	"github.com/innomon/legal-gemma4/pkg/safetensors"
	_ "github.com/gomlx/gomlx/backends/default"
)

func main() {
	inputPath := flag.String("input", "", "Path to .safetensors file or directory containing shards")
	outputPath := flag.String("output", "quantized_model", "Output directory for GoMLX checkpoints")
	blockSize := flag.Int("block_size", 64, "Quantization block size")
	flag.Parse()

	if *inputPath == "" {
		log.Fatal("Please specify --input")
	}

	// 1. Initialize Backend
	backend, err := backends.New()
	if err != nil {
		log.Fatalf("Failed to initialize backend: %v", err)
	}

	ctx := context.New()

	// 2. Open Safetensors
	// Handle single file for now
	st, err := safetensors.Open(*inputPath)
	if err != nil {
		log.Fatalf("Failed to open safetensors: %v", err)
	}
	defer st.Close()

	// 3. Iterate and Quantize
	for name, info := range st.Tensors {
		fmt.Printf("Processing %s (%v)...\n", name, info.DType)

		// Decide whether to quantize
		// Quantize Linear layer weights: usually end with ".weight" and rank 2
		// Skip embeddings, norms, biases
		shouldQuantize := strings.HasSuffix(name, ".weight") && len(info.Shape) == 2 && !strings.Contains(name, "norm") && !strings.Contains(name, "embed")

		data, err := st.TensorFloat32(name)
		if err != nil {
			fmt.Printf("  ⚠️ Skipping %s: %v\n", name, err)
			continue
		}

		if shouldQuantize {
			fmt.Printf("  💎 Quantizing to 4-bit (NF4)...\n")
			n := len(data)
			numBlocks := (n + *blockSize - 1) / *blockSize
			
			packed := make([]uint8, n/2)
			absmaxs := make([]float32, numBlocks)

			for b := 0; b < numBlocks; b++ {
				start := b * (*blockSize)
				end := (b + 1) * (*blockSize)
				if end > n {
					end = n
				}
				block := data[start:end]
				
				// Handle potential padding if block is smaller than blockSize
				if len(block) < *blockSize {
					padded := make([]float32, *blockSize)
					copy(padded, block)
					block = padded
				}

				indices, absmax := quant.QuantizeBlock(block)
				absmaxs[b] = absmax

				for i := 0; i < *blockSize; i += 2 {
					packedIdx := (b * (*blockSize) / 2) + (i / 2)
					if packedIdx < len(packed) {
						packed[packedIdx] = quant.PackIndices(indices[i], indices[i+1])
					}
				}
			}

			// Save to context
			layerCtx := ctx.In(name)
			layerCtx.VariableWithValue("packed_weight", packed)
			layerCtx.VariableWithValue("absmax", absmaxs)
			layerCtx.VariableWithValue("lut", quant.NF4Values)
		} else {
			fmt.Printf("  🚀 Keeping in high precision...\n")
			ctx.In(name).VariableWithValue("weight", data)
		}
	}

	// 4. Save Checkpoint
	fmt.Printf("Saving quantized model to %s...\n", *outputPath)
	if err := os.MkdirAll(*outputPath, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	cp, err := checkpoints.Build(ctx).Dir(*outputPath).Done()
	if err != nil {
		log.Fatalf("Failed to build checkpoint: %v", err)
	}
	if err := cp.Save(); err != nil {
		log.Fatalf("Failed to save checkpoint: %v", err)
	}

	fmt.Println("✅ Quantization complete!")
}
