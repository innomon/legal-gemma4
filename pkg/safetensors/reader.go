package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

// TensorInfo describes a tensor in the safetensors file.
type TensorInfo struct {
	DType   string `json:"dtype"`
	Shape   []int  `json:"shape"`
	Offsets []int  `json:"data_offsets"`
}

// Header is the JSON header of a safetensors file.
type Header map[string]interface{}

// Safetensors represents a loaded safetensors file.
type Safetensors struct {
	file   *os.File
	Header Header
	Tensors map[string]TensorInfo
	dataStart int64
}

// Open opens a safetensors file and reads its header.
func Open(path string) (*Safetensors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to read header size: %v", err)
	}

	headerBuf := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBuf); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to read header: %v", err)
	}

	var header Header
	if err := json.Unmarshal(headerBuf, &header); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to unmarshal header: %v", err)
	}

	tensors := make(map[string]TensorInfo)
	for k, v := range header {
		if k == "__metadata__" {
			continue
		}
		data, err := json.Marshal(v)
		if err != nil {
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(data, &info); err == nil {
			tensors[k] = info
		}
	}

	return &Safetensors{
		file:      f,
		Header:    header,
		Tensors:   tensors,
		dataStart: int64(8 + headerSize),
	}, nil
}

// Close closes the underlying file.
func (s *Safetensors) Close() error {
	return s.file.Close()
}

// ReadTensor reads the raw bytes of a tensor.
func (s *Safetensors) ReadTensor(name string) ([]byte, error) {
	info, ok := s.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %s not found", name)
	}

	size := info.Offsets[1] - info.Offsets[0]
	buf := make([]byte, size)
	if _, err := s.file.ReadAt(buf, s.dataStart+int64(info.Offsets[0])); err != nil {
		return nil, fmt.Errorf("failed to read tensor data: %v", err)
	}

	return buf, nil
}

// TensorFloat32 reads a tensor and converts it to []float32.
// It supports F32 and BF16 (converts BF16 to F32).
func (s *Safetensors) TensorFloat32(name string) ([]float32, error) {
	data, err := s.ReadTensor(name)
	if err != nil {
		return nil, err
	}

	info := s.Tensors[name]
	switch info.DType {
	case "F32":
		count := len(data) / 4
		result := make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
			result[i] = math.Float32frombits(bits)
		}
		return result, nil
	case "BF16":
		count := len(data) / 2
		result := make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint16(data[i*2 : (i+1)*2])
			result[i] = BFloat16ToFloat32(bits)
		}
		return result, nil
	default:
		return nil, fmt.Errorf("unsupported dtype: %s", info.DType)
	}
}

// BFloat16ToFloat32 converts a bfloat16 bit representation to a float32.
func BFloat16ToFloat32(bf uint16) float32 {
	return math.Float32frombits(uint32(bf) << 16)
}
