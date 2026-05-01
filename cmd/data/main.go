package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
)

// Command defines a CLI command.
type Command struct {
	Name        string
	Description string
	Run         func(args []string) error
}

const datasetURL = "https://huggingface.co/datasets/viber1/indian-law-dataset/resolve/main/train.jsonl"

func main() {
	commands := []Command{
		{
			Name:        "download",
			Description: "Download the Indian Law training dataset from Hugging Face",
			Run:         runDownload,
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
	fmt.Println("Usage: data <command> [arguments]")
	fmt.Println("\nAvailable commands:")
	for _, cmd := range commands {
		fmt.Printf("  %-10s %s\n", cmd.Name, cmd.Description)
	}
}

func runDownload(args []string) error {
	fs := flag.NewFlagSet("download", flag.ExitOnError)
	output := fs.String("output", "indian_law.jsonl", "Path to save the dataset")
	fs.Parse(args)

	fmt.Printf("📥 Downloading dataset from: %s\n", datasetURL)
	
	resp, err := http.Get(datasetURL)
	if err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(*output)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to save file: %w", err)
	}

	fmt.Printf("✅ Dataset saved to: %s\n", *output)
	return nil
}
