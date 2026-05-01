package main

import (
	"fmt"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/innomon/legal-gemma4/pkg/model"
	_ "github.com/gomlx/gomlx/backends/default"
)

func main() {
	config := model.DefaultGemma4E2BConfig()
	// Reduce size for verification to fit in memory easily
	config.NumLayers = 2
	config.HiddenDim = 512
	config.NumHeads = 8
	config.HeadDim = 64

	backend, err := backends.New()
	if err != nil {
		panic(err)
	}

	ctx := context.New().Checked(false)

	// Mock the frozen weights in the context so layers don't panic
	mockModelVariables(ctx.In("model"), config)

	// Build graph
	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, tokens *graph.Node) *graph.Node {
		return model.Gemma4Model(ctx, tokens, config)
	})
	if err != nil {
		panic(err)
	}

	// Initialize variables before execution
	err = ctx.InitializeVariables(backend, nil)
	if err != nil {
		panic(err)
	}

	// Mock input: batch 1, seq 4
	tokens := [][]int32{{1, 2, 3, 4}}
	// context.Exec returns ([]*tensors.Tensor, error)
	results, err := exec.Exec(tokens)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Model Output Shape: %v\n", results[0].Shape())
	fmt.Println("✅ Model architecture verified!")
}

func mockModelVariables(ctx *context.Context, config model.Config) {
	h := config.HiddenDim
	intermediate := h * 4

	// 1. Embedding
	ctx.In("embed_tokens").VariableWithShape("embeddings", shapes.Make(dtypes.Float32, config.VocabSize, h))

	// 2. Transformer Blocks
	for i := 0; i < config.NumLayers; i++ {
		blockScope := ctx.In(fmt.Sprintf("block_%d", i))
		
		// PLE
		blockScope.In("ple").VariableWithShape("weight", shapes.Make(dtypes.Float32, h))
		
		// Norms
		blockScope.In("input_layernorm").In("rms_norm").VariableWithShape("weight", shapes.Make(dtypes.Float32, h))
		blockScope.In("post_attention_layernorm").In("rms_norm").VariableWithShape("weight", shapes.Make(dtypes.Float32, h))

		// Attention
		attnScope := blockScope.In("attention")
		projs := []string{"q_proj", "k_proj", "v_proj", "o_proj"}
		for _, p := range projs {
			mockQLoRAVariables(attnScope.In(p), h, h, config.BlockSize)
		}
		
		// MLP
		mlpScope := blockScope.In("mlp")
		mockQLoRAVariables(mlpScope.In("gate_proj"), h, intermediate, config.BlockSize)
		mockQLoRAVariables(mlpScope.In("up_proj"), h, intermediate, config.BlockSize)
		mockQLoRAVariables(mlpScope.In("down_proj"), intermediate, h, config.BlockSize)
	}

	// 3. Final Norm
	ctx.In("norm").In("rms_norm").VariableWithShape("weight", shapes.Make(dtypes.Float32, h))

	// 4. LM Head
	ctx.In("lm_head").VariableWithShape("weights", shapes.Make(dtypes.Float32, h, config.VocabSize))
}

func mockQLoRAVariables(ctx *context.Context, inDim, outDim, blockSize int) {
	n := inDim * outDim
	ctx.VariableWithShape("packed_weight", shapes.Make(dtypes.Uint8, n/2))
	ctx.VariableWithShape("absmax", shapes.Make(dtypes.Float32, n/blockSize))
	ctx.VariableWithShape("lut", shapes.Make(dtypes.Float32, 16))
}
