package layers

import (
	"fmt"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/innomon/legal-gemma4/pkg/quant"
)

// QLoRAConfig holds configuration for a QLoRA layer.
type QLoRAConfig struct {
	Rank      int
	Alpha     float32
	BlockSize int
	LayerIdx  int
}

// QLoRALinear implements a QLoRA Linear layer.
// x: [batch, seq, in_features]
// ctx: current context, expected to have variables "packed_weight", "absmax", and "lut" (frozen)
// and "lora_a", "lora_b" (trainable).
func QLoRALinear(ctx *context.Context, x *graph.Node, outFeatures int, config QLoRAConfig) *graph.Node {
	g := x.Graph()
	inFeatures := x.Shape().Dimensions[len(x.Shape().Dimensions)-1]

	// --- 1. Frozen Path (NF4) ---
	// Load frozen weights from context
	packedVar := ctx.GetVariable("packed_weight")
	absmaxVar := ctx.GetVariable("absmax")
	lutVar := ctx.GetVariable("lut")

	if packedVar == nil || absmaxVar == nil || lutVar == nil {
		panic(fmt.Sprintf("missing frozen weights in context for QLoRA layer at %s", ctx.Scope()))
	}

	packed := packedVar.ValueGraph(g)
	absmax := absmaxVar.ValueGraph(g)
	lut := lutVar.ValueGraph(g)

	// Dequantize weights
	// W: [outFeatures, inFeatures]
	dequantizedFlat := quant.DequantizeNF4Graph(packed, absmax, lut, config.BlockSize)
	weights := graph.Reshape(dequantizedFlat, outFeatures, inFeatures)

	// Frozen Forward: Y = X * W^T
	yFrozen := graph.Dot(x, graph.Transpose(weights, 0, 1)).MatMul()

	// --- 2. LoRA Path ---
	// LoRA_A: [inFeatures, Rank] - Initialize with Gaussian
	// LoRA_B: [Rank, outFeatures] - Initialize with Zeros
	loraA := ctx.In("lora_a").VariableWithShape("weight", shapes.Make(dtypes.Float32, inFeatures, config.Rank))
	loraB := ctx.In("lora_b").VariableWithShape("weight", shapes.Make(dtypes.Float32, config.Rank, outFeatures))

	// Ensure LoRA_B is initialized to zero if not already
	if !loraB.HasValue() {
		loraB.SetValueGraph(graph.Zeros(g, loraB.Shape()))
	}

	// LoRA Forward: Y_lora = (X * A) * B * (alpha/r)
	yLora := graph.Dot(x, loraA.ValueGraph(g)).MatMul()
	yLora = graph.Dot(yLora, loraB.ValueGraph(g)).MatMul()

	scaling := graph.Scalar(g, dtypes.Float32, config.Alpha/float32(config.Rank))
	yLora = graph.Mul(yLora, scaling)

	// --- 3. Combine ---
	return graph.Add(yFrozen, yLora)
}
