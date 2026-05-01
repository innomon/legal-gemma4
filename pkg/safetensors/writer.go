package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
)

// Save saves a map of tensors to a safetensors file.
// For now, only supports float32.
func Save(path string, tensors map[string][]float32, shapes map[string][]int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	header := make(map[string]interface{})
	var currentOffset uint64

	tensorNames := make([]string, 0, len(tensors))
	for name := range tensors {
		tensorNames = append(tensorNames, name)
	}

	for _, name := range tensorNames {
		data := tensors[name]
		shape := shapes[name]
		size := uint64(len(data) * 4)

		header[name] = TensorInfo{
			DType:   "F32",
			Shape:   shape,
			Offsets: []int{int(currentOffset), int(currentOffset + size)},
		}
		currentOffset += size
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return err
	}

	// 1. Write header size
	headerSize := uint64(len(headerJSON))
	if err := binary.Write(f, binary.LittleEndian, headerSize); err != nil {
		return err
	}

	// 2. Write header
	if _, err := f.Write(headerJSON); err != nil {
		return err
	}

	// 3. Write tensor data
	for _, name := range tensorNames {
		data := tensors[name]
		for _, v := range data {
			if err := binary.Write(f, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	}

	return nil
}
