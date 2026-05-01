package main

import (
	"fmt"
	"math/rand"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/innomon/legal-gemma4/pkg/quant"
	_ "github.com/gomlx/gomlx/backends/default"
)

func main() {
	blockSize := 64
	numBlocks := 2
	n := blockSize * numBlocks

	// 1. Generate random weights
	weights := make([]float32, n)
	for i := range weights {
		weights[i] = float32(rand.NormFloat64())
	}

	// 2. Quantize and Pack
	packed := make([]uint8, n/2)
	absmaxs := make([]float32, numBlocks)

	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := (b + 1) * blockSize
		block := weights[start:end]
		indices, absmax := quant.QuantizeBlock(block)
		absmaxs[b] = absmax

		for i := 0; i < blockSize; i += 2 {
			packedIdx := (b * blockSize / 2) + (i / 2)
			packed[packedIdx] = quant.PackIndices(indices[i], indices[i+1])
		}
	}

	// 3. Dequantize using GoMLX Graph
	g := graph.New()
	indicesNode := graph.Constant(g, tensors.FromFlat(packed))
	absmaxNode := graph.Constant(g, tensors.FromFlat(absmaxs))
	lutNode := graph.Constant(g, tensors.FromFlat(quant.NF4Values))

	dequantizedNode := quant.DequantizeNF4Graph(indicesNode, absmaxNode, lutNode, blockSize)
	
	// Execute graph
	exec := graph.NewExec(g, dequantizedNode)
	results := exec.Call()
	dequantized := results[0].Value().([]float32)

	// 4. Compare
	fmt.Printf("First 8 original:    %v\n", weights[:8])
	fmt.Printf("First 8 dequantized: %v\n", dequantized[:8])

	mse := float32(0)
	for i := range weights {
		diff := weights[i] - dequantized[i]
		mse += diff * diff
	}
	mse /= float32(n)
	fmt.Printf("Mean Squared Error: %f\n", mse)
}
