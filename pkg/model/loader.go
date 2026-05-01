package model

import (
	"fmt"
	"os"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
)

// LoadQuantizedModel loads a model from a GoMLX checkpoint directory.
func LoadQuantizedModel(ctx *context.Context, path string) error {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return fmt.Errorf("checkpoint path %s does not exist", path)
	}

	// We use checkpoints package to load variables into the context.
	// Since we are loading for training, we want the existing variables to be set.
	// But LoRA adapters might not be in the checkpoint yet.
	// We'll use Reuse() so it loads what's there and doesn't panic on extras.
	ctx.Reuse()
	_, err := checkpoints.Build(ctx).Dir(path).Done()
	if err != nil {
		return fmt.Errorf("failed to load checkpoint: %v", err)
	}

	// Lock the frozen weights
	// All variables NOT containing "lora" or "ple" should be frozen.
	// Wait, PLE is trainable in Gemma 4? 
	// The instruction says: "PLE signal unique to each layer index". 
	// Usually PLE is trainable if it's an embedding signal.
	
	ctx.EnumerateVariables(func(v *context.Variable) {
		name := v.ScopeAndName()
		if !strContains(name, "lora") && !strContains(name, "ple") {
			v.SetTrainable(false)
		} else {
			v.SetTrainable(true)
		}
	})

	return nil
}

func contains(s, substr string) bool {
	return fmt.Sprintf("%v", s) != "" && (len(s) >= len(substr)) // basic check
}

// Fixed contains
func strContains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
