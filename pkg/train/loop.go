package train

import (
	"fmt"
	"time"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/innomon/legal-gemma4/pkg/data"
	"github.com/innomon/legal-gemma4/pkg/model"
)

// TrainQLoRA runs the finetuning loop.
func TrainQLoRA(backend backends.Backend, ctx *context.Context, config model.Config, ds *data.Dataset, numEpochs int, batchSize int) {
	// 1. Define Optimizer
	// QLoRA typically uses AdamW with low learning rate.
	opt := optimizers.Adam().LearningRate(1e-4).Done()

	// 2. Define Training Step
	trainExec, err := context.NewExec(backend, ctx, func(ctx *context.Context, inputs, targets *graph.Node) *graph.Node {
		g := inputs.Graph()
		logits := model.Gemma4Model(ctx, inputs, config)

		// Loss: SparseCategoricalCrossEntropyLogits
		// logits: [batch, seq, vocab]
		// targets: [batch, seq]
		loss := losses.SparseCategoricalCrossEntropyLogits([]*graph.Node{targets}, []*graph.Node{logits})
		
		// Apply gradients only to trainable variables (LoRA adapters)
		opt.UpdateGraph(ctx, g, loss)
		
		return loss
	})
	if err != nil {
		fmt.Printf("❌ Failed to create training execution: %v\n", err)
		return
	}

	// 3. Training Loop
	fmt.Printf("🚀 Starting QLoRA finetuning for %d epochs...\n", numEpochs)
	startTime := time.Now()

	for epoch := 0; epoch < numEpochs; epoch++ {
		totalLoss := float32(0)
		numBatches := ds.Len() / batchSize
		
		for b := 0; b < numBatches; b++ {
			batchTokens, batchTargets := ds.GetBatch(b*batchSize, batchSize)
			
			results, err := trainExec.Exec(batchTokens, batchTargets)
			if err != nil {
				fmt.Printf("❌ Error during batch %d: %v\n", b, err)
				continue
			}

			lossVal := results[0].Value().(float32)
			totalLoss += lossVal

			if b%10 == 0 {
				fmt.Printf("  Epoch %d | Batch %d/%d | Loss: %.4f\n", epoch, b, numBatches, lossVal)
			}
		}
		
		fmt.Printf("✅ Epoch %d complete | Avg Loss: %.4f | Elapsed: %v\n", epoch, totalLoss/float32(numBatches), time.Since(startTime))
	}
}
