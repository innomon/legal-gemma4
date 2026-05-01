package layers

import (
	"math"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// Gemma4Attention implements the attention mechanism for Gemma 4.
func Gemma4Attention(ctx *context.Context, x *graph.Node, numHeads, headDim int, config QLoRAConfig) *graph.Node {
	g := x.Graph()
	ctx = ctx.In("attention")
	shape := x.Shape()
	batchSize := shape.Dimensions[0]
	seqLen := shape.Dimensions[1]
	hiddenDim := numHeads * headDim

	// 1. Projections
	q := QLoRALinear(ctx.In("q_proj"), x, hiddenDim, config)
	k := QLoRALinear(ctx.In("k_proj"), x, hiddenDim, config)
	v := QLoRALinear(ctx.In("v_proj"), x, hiddenDim, config)

	// 2. Reshape and Apply RoPE
	q = graph.Reshape(q, batchSize, seqLen, numHeads, headDim)
	k = graph.Reshape(k, batchSize, seqLen, numHeads, headDim)
	v = graph.Reshape(v, batchSize, seqLen, numHeads, headDim)

	q = ApplyRoPE(q, 10000.0)
	k = ApplyRoPE(k, 10000.0)

	// 3. Scale Dot Product Attention
	// Transpose Q: [batch, heads, seq, dim]
	q = graph.TransposeAllDims(q, 0, 2, 1, 3)
	// Transpose K: [batch, heads, dim, seq]
	k = graph.TransposeAllDims(k, 0, 2, 3, 1)
	// Transpose V: [batch, heads, seq, dim]
	v = graph.TransposeAllDims(v, 0, 2, 1, 3)

	// Scores: [batch, heads, seq, seq]
	scores := graph.Dot(q, k).MatMul()
	scaling := graph.Scalar(g, dtypes.Float32, 1.0/math.Sqrt(float64(headDim)))
	scores = graph.Mul(scores, scaling)

	// Causal Mask
	mask := CausalMask(g, seqLen)
	// Broadcast mask to [batch, heads, seq, seq]
	mask = graph.BroadcastToDims(graph.Reshape(mask, 1, 1, seqLen, seqLen), batchSize, numHeads, seqLen, seqLen)
	// Apply mask: large negative for masked out positions
	bigNeg := graph.Scalar(g, dtypes.Float32, -1e10)
	scores = graph.Where(mask, scores, bigNeg)

	// Softmax
	probs := graph.Softmax(scores, -1)

	// Output: [batch, heads, seq, dim]
	out := graph.Dot(probs, v).MatMul()

	// 4. Combine Heads: [batch, seq, heads * dim]
	out = graph.TransposeAllDims(out, 0, 2, 1, 3)
	out = graph.Reshape(out, batchSize, seqLen, hiddenDim)

	// 5. Output Projection
	return QLoRALinear(ctx.In("o_proj"), out, shape.Dimensions[len(shape.Dimensions)-1], config)
}

// CausalMask returns a boolean tensor of shape [seqLen, seqLen] where true means keep.
func CausalMask(g *graph.Graph, seqLen int) *graph.Node {
	rowIndices := graph.Iota(g, shapes.Make(dtypes.Int32, seqLen, seqLen), 0)
	colIndices := graph.Iota(g, shapes.Make(dtypes.Int32, seqLen, seqLen), 1)
	return graph.LessOrEqual(colIndices, rowIndices)
}
