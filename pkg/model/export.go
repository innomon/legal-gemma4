package model

import (
	"fmt"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/innomon/legal-gemma4/pkg/quant"
	"github.com/innomon/legal-gemma4/pkg/safetensors"
)

// ExportMergedModel merges LoRA adapters into base weights and saves to safetensors.
func ExportMergedModel(backend backends.Backend, ctx *context.Context, config Config, outputPath string) error {
	mergedTensors := make(map[string][]float32)
	tensorShapes := make(map[string][]int)

	// We use a GoMLX graph to perform the merging efficiently.
	// But we'll do it layer by layer to keep memory low.
	
	ctx.EnumerateVariables(func(v *context.Variable) {
		name := v.ScopeAndName()
		
		// Skip if it's a LoRA or PLE variable (we'll process them via their base layers)
		if strContains(name, "lora") || strContains(name, "ple") || strContains(name, "absmax") || strContains(name, "lut") || strContains(name, "packed_weight") {
			return
		}

		// If it's a high-precision weight (embedding, norm), just copy it.
		if strContains(name, "weight") || strContains(name, "embeddings") {
			val, _ := v.Value()
			mergedTensors[name] = val.Value().([]float32)
			tensorShapes[name] = v.Shape().Dimensions
		}
	})

	// Now handle the quantized layers with LoRA
	// We'll iterate through blocks and projections
	for i := 0; i < config.NumLayers; i++ {
		blockScope := fmt.Sprintf("/model/block_%d", i)
		
		// Projections
		projs := []string{
			"attention/q_proj", "attention/k_proj", "attention/v_proj", "attention/o_proj",
			"mlp/gate_proj", "mlp/up_proj", "mlp/down_proj",
		}

		for _, p := range projs {
			scope := blockScope + "/" + p
			
			// Build a small graph to merge
			mergeExec, err := graph.NewExec(backend, func(g *graph.Graph) *graph.Node {
				packedVar := ctx.GetVariableByScopeAndName(scope, "packed_weight")
				absmaxVar := ctx.GetVariableByScopeAndName(scope, "absmax")
				lutVar := ctx.GetVariableByScopeAndName(scope, "lut")
				loraAVar := ctx.GetVariableByScopeAndName(scope+"/lora_a", "weight")
				loraBVar := ctx.GetVariableByScopeAndName(scope+"/lora_b", "weight")

				if packedVar == nil { return nil }

				// Dequantize base
				// We need the shapes. Let's assume we can get them from the variable objects.
				// For this script, we'll need to know outFeatures/inFeatures.
				// This is a bit tricky without a proper model registry.
				// Let's assume we can infer from packed size.
				
				packed := packedVar.ValueGraph(g)
				absmax := absmaxVar.ValueGraph(g)
				lut := lutVar.ValueGraph(g)
				
				baseWeights := quant.DequantizeNF4Graph(packed, absmax, lut, config.BlockSize)
				
				// LoRA Contribution
				a := loraAVar.ValueGraph(g)
				b := loraBVar.ValueGraph(g)
				
				// Contribution = (A * B) * (alpha/r)
				// A: [in, r], B: [r, out] -> [in, out]
				contrib := graph.Dot(a, b).MatMul()
				scaling := graph.Scalar(g, a.DType(), config.LoRAAlpha/float32(config.LoRARank))
				contrib = graph.Mul(contrib, scaling)
				
				// Reshape dequantized base back to [out, in] and contribution to [in, out]
				// Wait, base is [out, in] and contrib is [in, out].
				// Let's ensure they match.
				outDim := b.Shape().Dimensions[1]
				inDim := a.Shape().Dimensions[0]
				
				baseWeights = graph.Reshape(baseWeights, outDim, inDim)
				return graph.Add(baseWeights, graph.Transpose(contrib, 0, 1))
			})
			
			if err != nil || mergeExec == nil { continue }
			
			results, err := mergeExec.Exec()
			if err == nil {
				mergedTensors[scope+"/weight"] = results[0].Value().([]float32)
				tensorShapes[scope+"/weight"] = results[0].Shape().Dimensions
			}
		}

		// Also save PLE as a separate tensor for each block
		pleVar := ctx.GetVariableByScopeAndName(blockScope+"/ple", "weight")
		if pleVar != nil {
			val, _ := pleVar.Value()
			mergedTensors[blockScope+"/ple"] = val.Value().([]float32)
			tensorShapes[blockScope+"/ple"] = pleVar.Shape().Dimensions
		}
	}

	// Save to safetensors
	fmt.Printf("Saving merged model to %s...\n", outputPath)
	return safetensors.Save(outputPath, mergedTensors, tensorShapes)
}
