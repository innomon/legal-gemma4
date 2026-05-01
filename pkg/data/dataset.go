package data

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"github.com/daulet/tokenizers"
)

// Example Indian Law entry
type LawEntry struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// Dataset handles loading and tokenizing the indian-law-dataset.
type Dataset struct {
	tokenizer *tokenizers.Tokenizer
	entries   []LawEntry
	maxLen    int
}

// NewDataset creates a new dataset from a JSONL file.
func NewDataset(filePath string, tokenizerPath string, maxLen int) (*Dataset, error) {
	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %v", err)
	}

	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var entries []LawEntry
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var entry LawEntry
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			continue
		}
		entries = append(entries, entry)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to scan dataset file: %v", err)
	}

	return &Dataset{
		tokenizer: tk,
		entries:   entries,
		maxLen:    maxLen,
	}, nil
}

// GetBatch returns a batch of tokenized prompts and targets.
// Returns: tokens [batch, maxLen], targets [batch, maxLen]
func (d *Dataset) GetBatch(startIdx, batchSize int) ([][]int32, [][]int32) {
	if len(d.entries) == 0 {
		return nil, nil
	}
	endIdx := startIdx + batchSize
	if endIdx > len(d.entries) {
		endIdx = len(d.entries)
	}

	batchTokens := make([][]int32, batchSize)
	batchTargets := make([][]int32, batchSize)

	for i := 0; i < batchSize; i++ {
		idx := (startIdx + i) % len(d.entries)
		entry := d.entries[idx]
		
		prompt := fmt.Sprintf("### Question: %s \n### Advocate Analysis: %s", entry.Question, entry.Answer)
		
		tokens, _ := d.tokenizer.Encode(prompt, true)
		ids := make([]int32, d.maxLen)
		targets := make([]int32, d.maxLen)
		
		// Copy IDs and handle padding/truncation
		for j := 0; j < d.maxLen; j++ {
			if j < len(tokens) {
				ids[j] = int32(tokens[j])
				// Target is next token, so shift by 1
				if j+1 < len(tokens) {
					targets[j] = int32(tokens[j+1])
				} else {
					targets[j] = -1 // Padding/EOS indicator
				}
			} else {
				ids[j] = 0 // Pad token
				targets[j] = -1
			}
		}
		batchTokens[i] = ids
		batchTargets[i] = targets
	}

	return batchTokens, batchTargets
}

func (d *Dataset) Len() int {
	return len(d.entries)
}
