package quant

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
)

// DequantizeNF4Graph implements the NF4 dequantization logic within a GoMLX graph.
// It takes packed uint8 indices, absmax scaling factors, and the NF4 LUT.
// indices: [N/2] uint8 tensor
// absmax: [N/block_size] float32 tensor
// Returns: [N] float32 tensor
func DequantizeNF4Graph(indices, absmax, lut *graph.Node, blockSize int) *graph.Node {
	g := indices.Graph()

	// 1. Unpack indices
	// Low bits: packed & 0x0F
	idx1 := graph.BitwiseAnd(indices, graph.Scalar(g, dtypes.Uint8, 0x0F))
	// High bits: packed >> 4
	idx2 := graph.BitwiseShiftRightLogicalScalar(indices, 4)

	// 2. Stack and Flatten indices to [N]
	// We need to interleave idx1 and idx2.
	// Easiest is to stack them on a new dimension and then reshape.
	stacked := graph.Stack([]*graph.Node{idx1, idx2}, -1)
	flatIndices := graph.Reshape(stacked, -1)

	// 3. Gather from LUT
	// LUT is [16] float32
	// We need to expand flatIndices to [N, 1] for Gather to work correctly on a rank 1 params.
	expandedIndices := graph.ExpandDims(flatIndices, -1)
	dequantized := graph.Gather(lut, graph.ConvertDType(expandedIndices, dtypes.Int32))

	// 4. Apply scaling
	// We need to reshape dequantized to [N/blockSize, blockSize] to multiply by absmax [N/blockSize]
	numBlocks := absmax.Shape().Dimensions[0]
	reshaped := graph.Reshape(dequantized, numBlocks, blockSize)
	
	// Multiply by absmax (broadcasting)
	scaled := graph.Mul(reshaped, graph.Reshape(absmax, numBlocks, 1))

	// Flatten back to original size [N]
	return graph.Reshape(scaled, -1)
}
