package layers

import (
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// Gemma4MLP implements the GeGLU MLP for Gemma 4.
func Gemma4MLP(ctx *context.Context, x *graph.Node, config QLoRAConfig) *graph.Node {
	ctx = ctx.In("mlp")
	hiddenDim := x.Shape().Dimensions[len(x.Shape().Dimensions)-1]
	intermediateDim := hiddenDim * 4 // Standard for Gemma

	// Gate and Up projections
	gate := QLoRALinear(ctx.In("gate_proj"), x, intermediateDim, config)
	up := QLoRALinear(ctx.In("up_proj"), x, intermediateDim, config)

	// GeGLU activation
	activated := graph.Mul(activations.Gelu(gate), up)

	// Down projection
	return QLoRALinear(ctx.In("down_proj"), activated, hiddenDim, config)
}
