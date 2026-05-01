package main

import (
	"flag"
	"fmt"
	"log"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/innomon/legal-gemma4/pkg/model"
	_ "github.com/gomlx/gomlx/backends/default"
)

func main() {
	modelDir := flag.String("model", "quantized_model", "Path to quantized model directory (with trained LoRA)")
	outputPath := flag.String("output", "merged_model.safetensors", "Output safetensors file")
	flag.Parse()

	// 1. Initialize Backend
	backend, err := backends.New()
	if err != nil {
		log.Fatalf("Failed to initialize backend: %v", err)
	}

	ctx := context.New()

	// 2. Load Model
	fmt.Printf("📂 Loading model from %s...\n", *modelDir)
	if err := model.LoadQuantizedModel(ctx, *modelDir); err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// 3. Export
	config := model.DefaultGemma4E2BConfig()
	if err := model.ExportMergedModel(backend, ctx, config, *outputPath); err != nil {
		log.Fatalf("Failed to export model: %v", err)
	}

	fmt.Println("✅ Model merged and exported successfully!")
}
