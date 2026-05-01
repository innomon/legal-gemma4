package layers

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// RMSNorm implements Root Mean Square Layer Normalization.
func RMSNorm(ctx *context.Context, x *graph.Node, eps float32) *graph.Node {
	g := x.Graph()
	ctx = ctx.In("rms_norm")
	hiddenDim := x.Shape().Dimensions[x.Rank()-1]

	// 1. Calculate variance: mean(x^2, axis=-1)
	ms := graph.ReduceMean(graph.Square(x), -1)
	
	// 2. Normalize: x / sqrt(ms + eps)
	invRms := graph.Inverse(graph.Sqrt(graph.Add(ms, graph.Scalar(g, x.DType(), eps))))
	// Expand dims of invRms from [batch, seq] to [batch, seq, 1] to broadcast with [batch, seq, hidden]
	invRms = graph.ExpandDims(invRms, -1)
	normed := graph.Mul(x, invRms)

	// 3. Scale by weight (gamma)
	gamma := ctx.VariableWithShape("weight", shapes.Make(dtypes.Float32, hiddenDim))
	if !gamma.HasValue() {
		gamma.SetValueGraph(graph.Ones(g, gamma.Shape()))
	}
	
	// Reshape gamma from [hidden] to [1, 1, hidden] to broadcast with [batch, seq, hidden]
	gammaValue := graph.Reshape(gamma.ValueGraph(g), 1, 1, hiddenDim)
	return graph.Mul(normed, gammaValue)
}
