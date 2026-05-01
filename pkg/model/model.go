package model

import (
	"fmt"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	my_layers "github.com/innomon/legal-gemma4/pkg/layers"
)

// Config holds the hyperparameters for the Gemma 4 model.
type Config struct {
	VocabSize    int
	NumLayers    int
	NumHeads     int
	HeadDim      int
	HiddenDim    int
	LoRARank     int
	LoRAAlpha    float32
	BlockSize    int
}

// DefaultGemma4E2BConfig returns the configuration for Gemma-4-E2B.
func DefaultGemma4E2BConfig() Config {
	return Config{
		VocabSize: 256000,
		NumLayers: 24,
		NumHeads:  16,
		HeadDim:   128,
		HiddenDim: 2048,
		LoRARank:  8,
		LoRAAlpha: 32,
		BlockSize: 64,
	}
}

// TransformerBlock implements a single transformer block for Gemma 4.
func TransformerBlock(ctx *context.Context, x *graph.Node, config Config, layerIdx int) *graph.Node {
	g := x.Graph()
	ctx = ctx.In(fmt.Sprintf("block_%d", layerIdx))
	
	// 1. PLE (Per-Layer Embedding)
	// Gemma 4 specific: add a residual embedding signal.
	pleVar := ctx.In("ple").VariableWithShape("weight", shapes.Make(dtypes.Float32, config.HiddenDim))
	pleSignal := pleVar.ValueGraph(g)
	// Reshape for broadcasting [batch, seq, hidden] + [1, 1, hidden]
	pleSignal = graph.Reshape(pleSignal, 1, 1, config.HiddenDim)
	x = graph.Add(x, pleSignal)

	qloraCfg := my_layers.QLoRAConfig{
		Rank:      config.LoRARank,
		Alpha:     config.LoRAAlpha,
		BlockSize: config.BlockSize,
		LayerIdx:  layerIdx,
	}

	// 2. Attention path
	residual := x
	normedX := my_layers.RMSNorm(ctx.In("input_layernorm"), x, 1e-6)
	attnOut := my_layers.Gemma4Attention(ctx, normedX, config.NumHeads, config.HeadDim, qloraCfg)
	x = graph.Add(residual, attnOut)

	// 3. MLP path
	residual = x
	normedX = my_layers.RMSNorm(ctx.In("post_attention_layernorm"), x, 1e-6)
	mlpOut := my_layers.Gemma4MLP(ctx, normedX, qloraCfg)
	x = graph.Add(residual, mlpOut)

	return x
}

// Gemma4Model builds the full Gemma 4 model.
// tokens: [batch, seq] int32
func Gemma4Model(ctx *context.Context, tokens *graph.Node, config Config) *graph.Node {
	ctx = ctx.In("model")

	// 1. Embedding
	// Note: We use standard embedding layer, not QLoRA here.
	x := layers.Embedding(ctx.In("embed_tokens"), tokens, dtypes.Float32, config.VocabSize, config.HiddenDim)

	// 2. Transformer Blocks
	for i := 0; i < config.NumLayers; i++ {
		x = TransformerBlock(ctx, x, config, i)
	}

	// 3. Final Norm
	x = my_layers.RMSNorm(ctx.In("norm"), x, 1e-6)

	// 4. Output Head (LM Head)
	lmHeadCtx := ctx.In("lm_head")
	return layers.Dense(lmHeadCtx, x, false, config.VocabSize)
}
