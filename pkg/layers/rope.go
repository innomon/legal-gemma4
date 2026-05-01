package layers

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// ApplyRoPE applies Rotary Positional Embeddings to a tensor x of shape [batch, seq, heads, dim].
func ApplyRoPE(x *graph.Node, base float64) *graph.Node {
	g := x.Graph()
	shape := x.Shape()
	seqLen := shape.Dimensions[1]
	headDim := shape.Dimensions[3]

	// 1. Generate frequencies: base^(-2i/dim)
	halfDim := headDim / 2
	indices := graph.Iota(g, shapes.Make(dtypes.Int32, halfDim), 0)
	indicesF := graph.ConvertDType(indices, dtypes.Float32)
	exponent := graph.Mul(indicesF, graph.Scalar(g, dtypes.Float32, -2.0/float64(headDim)))
	freqs := graph.Exp(graph.Mul(graph.Log(graph.Scalar(g, dtypes.Float32, base)), exponent))

	// 2. Generate positions [0..seqLen-1]
	t := graph.Iota(g, shapes.Make(dtypes.Int32, seqLen), 0)
	tF := graph.ConvertDType(t, dtypes.Float32)

	// 3. Compute phases [seqLen, halfDim]
	phases := graph.Mul(graph.Reshape(tF, seqLen, 1), graph.Reshape(freqs, 1, halfDim))
	cos := graph.Cos(phases)
	sin := graph.Sin(phases)

	// Broadcast cos/sin to [1, seq, 1, halfDim] for [batch, seq, heads, halfDim]
	cos = graph.Reshape(cos, 1, seqLen, 1, halfDim)
	sin = graph.Reshape(sin, 1, seqLen, 1, halfDim)

	// 4. Apply RoPE
	// x is [batch, seq, heads, headDim]
	x1 := graph.Slice(x, graph.AxisRange(), graph.AxisRange(), graph.AxisRange(), graph.AxisRange(0, halfDim))
	x2 := graph.Slice(x, graph.AxisRange(), graph.AxisRange(), graph.AxisRange(), graph.AxisRange(halfDim, headDim))
	
	xRope1 := graph.Sub(graph.Mul(x1, cos), graph.Mul(x2, sin))
	xRope2 := graph.Add(graph.Mul(x1, sin), graph.Mul(x2, cos))

	return graph.Concatenate([]*graph.Node{xRope1, xRope2}, 3)
}
