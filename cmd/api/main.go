package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/innomon/legal-gemma4/pkg/model"
	"github.com/innomon/legal-gemma4/pkg/server"

	// Register default backends. 
	// The user mentioned go-darwinml for M4 Mac. 
	// In GoMLX, the backend is typically registered via side-effect imports.
	_ "github.com/gomlx/gomlx/backends/default"
)

// Command defines a CLI command.
type Command struct {
	Name        string
	Description string
	Run         func(args []string) error
}

func main() {
	// Handcrafted command registry
	commands := []Command{
		{
			Name:        "serve",
			Description: "Start the OpenAI-compatible API server",
			Run:         runServe,
		},
	}

	if len(os.Args) < 2 {
		printUsage(commands)
		return
	}

	cmdName := os.Args[1]
	for _, cmd := range commands {
		if cmd.Name == cmdName {
			if err := cmd.Run(os.Args[2:]); err != nil {
				log.Fatalf("❌ Error running command %s: %v", cmdName, err)
			}
			return
		}
	}

	fmt.Printf("Error: unknown command %q\n", cmdName)
	printUsage(commands)
}

func printUsage(commands []Command) {
	fmt.Println("Usage: api <command> [arguments]")
	fmt.Println("\nAvailable commands:")
	for _, cmd := range commands {
		fmt.Printf("  %-10s %s\n", cmd.Name, cmd.Description)
	}
}

func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	port := fs.Int("port", 8080, "Port to listen on")
	modelDir := fs.String("model", "quantized_model", "Path to quantized model directory")
	tokenizerPath := fs.String("tokenizer", "tokenizer.json", "Path to tokenizer.json file")
	fs.Parse(args)

	// 1. Initialize Backend
	// GoMLX will pick the best available backend (like go-darwinml if registered)
	backend, err := backends.New()
	if err != nil {
		return fmt.Errorf("failed to initialize backend: %w", err)
	}

	// 2. Initialize ML Context
	ctx := context.New()

	// 3. Setup and Start Server
	s := &server.Server{
		Backend:       backend,
		Context:       ctx,
		Port:          *port,
		ModelDir:      *modelDir,
		TokenizerPath: *tokenizerPath,
		Config:        model.DefaultGemma4E2BConfig(),
	}

	return s.Start()
}
