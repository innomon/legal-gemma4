package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/daulet/tokenizers"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/innomon/legal-gemma4/pkg/model"
)

// Server handles OpenAI-compatible requests for the Legal-Gemma4 model.
type Server struct {
	Backend       backends.Backend
	Context       *context.Context
	Port          int
	ModelDir      string
	TokenizerPath string
	Config        model.Config
	tokenizer     *tokenizers.Tokenizer
	exec          *context.Exec
}

// Model represents a simple OpenAI model object.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ChatCompletionRequest follows the OpenAI chat completions schema.
type ChatCompletionRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionResponse follows the OpenAI chat completions response schema.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Choices []Choice `json:"choices"`
}

type Choice struct {
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
	Index        int     `json:"index"`
}

// Start launches the HTTP server.
func (s *Server) Start() error {
	// 1. Load Tokenizer
	tk, err := tokenizers.FromFile(s.TokenizerPath)
	if err != nil {
		return fmt.Errorf("failed to load tokenizer: %w", err)
	}
	s.tokenizer = tk

	// 2. Load Model Checkpoint
	fmt.Printf("📂 Loading quantized model from %s...\n", s.ModelDir)
	if err := model.LoadQuantizedModel(s.Context, s.ModelDir); err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	// 3. Initialize Variables
	if err := s.Context.InitializeVariables(s.Backend, nil); err != nil {
		return fmt.Errorf("failed to initialize variables: %w", err)
	}

	// 4. Setup Execution
	s.exec, err = context.NewExec(s.Backend, s.Context, func(ctx *context.Context, tokens *graph.Node) *graph.Node {
		logits := model.Gemma4Model(ctx, tokens, s.Config)
		// Return only the last token's logits for inference
		// logits: [batch, seq, vocab_size]
		return graph.Slice(logits, graph.AxisRange(), graph.AxisRange(-1), graph.AxisRange())
	})
	if err != nil {
		return fmt.Errorf("failed to setup execution: %w", err)
	}

	// 5. Setup Routes
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", s.handleListModels)
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)

	fmt.Printf("🚀 Legal-Gemma4 API server listening on :%d\n", s.Port)
	return http.ListenAndServe(fmt.Sprintf(":%d", s.Port), mux)
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	models := []Model{
		{ID: "legal-gemma-4-e2b", Object: "model", Created: time.Now().Unix(), OwnedBy: "legal-gemma"},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"data": models})
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if len(req.Messages) == 0 {
		http.Error(w, "no messages provided", http.StatusBadRequest)
		return
	}

	// Simple prompt construction: concat messages
	var prompt strings.Builder
	for _, m := range req.Messages {
		if m.Role == "user" {
			prompt.WriteString("### Question: ")
		} else if m.Role == "assistant" {
			prompt.WriteString("### Advocate Analysis: ")
		}
		prompt.WriteString(m.Content)
		prompt.WriteString("\n")
	}
	prompt.WriteString("### Advocate Analysis: ")

	// 1. Tokenize
	tokens, _ := s.tokenizer.Encode(prompt.String(), true)
	inputIDs := make([]int32, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = int32(t)
	}

	// 2. Inference Loop
	generatedTokens := []int32{}
	maxNewTokens := 100
	
	// Start with full context
	currentInput := inputIDs

	for len(generatedTokens) < maxNewTokens {
		currentTokensTensor := tensors.FromFlatDataAndDimensions(currentInput, 1, len(currentInput))
		results, err := s.exec.Exec(currentTokensTensor)
		if err != nil {
			http.Error(w, fmt.Sprintf("inference error: %v", err), http.StatusInternalServerError)
			return
		}

		logitsTensor := results[0]
		// logits: [1, 1, vocab_size]
		// Greedy search for now
		logits := logitsTensor.Value().([][][]float32)[0][0]
		var nextToken int32
		var maxLogit float32 = -1e10
		for i, l := range logits {
			if l > maxLogit {
				maxLogit = l
				nextToken = int32(i)
			}
		}

		generatedTokens = append(generatedTokens, nextToken)
		
		// Stop on EOS (assuming token 1 is EOS as in Gemma)
		if nextToken == 1 {
			break
		}

		// For simplicity, append to current input for next step (full context)
		// In a real server, we would use a KV cache to only pass the last token.
		currentInput = append(currentInput, nextToken)
	}

	// 3. Detokenize
	ids := make([]uint32, len(generatedTokens))
	for i, t := range generatedTokens {
		ids[i] = uint32(t)
	}
	responseContent := s.tokenizer.Decode(ids, true)

	resp := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: responseContent,
				},
				FinishReason: "stop",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
