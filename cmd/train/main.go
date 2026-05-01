package main

import (
	"flag"
	"fmt"
	"log"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/innomon/legal-gemma4/pkg/data"
	"github.com/innomon/legal-gemma4/pkg/model"
	"github.com/innomon/legal-gemma4/pkg/train"
	_ "github.com/gomlx/gomlx/backends/default"
)

func main() {
	modelDir := flag.String("model", "quantized_model", "Path to quantized model directory")
	datasetPath := flag.String("dataset", "indian_law.jsonl", "Path to dataset JSONL file")
	tokenizerPath := flag.String("tokenizer", "tokenizer.json", "Path to tokenizer.json file")
	epochs := flag.Int("epochs", 3, "Number of training epochs")
	batchSize := flag.Int("batch_size", 1, "Batch size")
	maxLen := flag.Int("max_len", 2048, "Maximum sequence length")
	flag.Parse()

	// 1. Initialize Backend
	backend, err := backends.New()
	if err != nil {
		log.Fatalf("Failed to initialize backend: %v", err)
	}

	ctx := context.New()

	// 2. Load Model
	fmt.Printf("📂 Loading quantized model from %s...\n", *modelDir)
	if err := model.LoadQuantizedModel(ctx, *modelDir); err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// 3. Load Dataset
	fmt.Printf("📖 Loading dataset from %s...\n", *datasetPath)
	ds, err := data.NewDataset(*datasetPath, *tokenizerPath, *maxLen)
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}

	// 4. Start Training
	config := model.DefaultGemma4E2BConfig()
	train.TrainQLoRA(backend, ctx, config, ds, *epochs, *batchSize)

	fmt.Println("✅ Training complete!")
}
