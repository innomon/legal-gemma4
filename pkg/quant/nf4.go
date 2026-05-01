package quant

import (
	"math"
)

// NF4Values are the 16 values of the NormalFloat4 lookup table.
// These are the information-theoretically optimal values for a normal distribution.
var NF4Values = []float32{
	-1.0,
	-0.6941927075386047,
	-0.5121898055076599,
	-0.37393155694007874,
	-0.256186306476593,
	-0.1497316211462021,
	-0.04963508993387222,
	0.0,
	0.04279178008437157,
	0.14118018746376038,
	0.24603383243083954,
	0.3602966070175171,
	0.49410802125930786,
	0.6496285200119019,
	0.8614716529846191,
	1.0,
}

// FindClosestNF4 returns the index (0-15) of the NF4 value closest to the target.
func FindClosestNF4(target float32) uint8 {
	minDist := float32(math.MaxFloat32)
	var bestIdx uint8
	for i, val := range NF4Values {
		dist := float32(math.Abs(float64(target - val)))
		if dist < minDist {
			minDist = dist
			bestIdx = uint8(i)
		}
	}
	return bestIdx
}

// QuantizeBlock quantizes a block of floats (typically 64) into NF4 indices.
// It returns the indices and the absmax scaling factor.
func QuantizeBlock(block []float32) ([]uint8, float32) {
	absMax := float32(0.0)
	for _, v := range block {
		a := float32(math.Abs(float64(v)))
		if a > absMax {
			absMax = a
		}
	}

	indices := make([]uint8, len(block))
	if absMax == 0 {
		return indices, 0
	}

	for i, v := range block {
		normalized := v / absMax
		indices[i] = FindClosestNF4(normalized)
	}

	return indices, absMax
}

// PackIndices packs two 4-bit indices into a single uint8.
func PackIndices(idx1, idx2 uint8) uint8 {
	return (idx1 & 0x0F) | ((idx2 & 0x0F) << 4)
}

// UnpackIndices unpacks a single uint8 into two 4-bit indices.
func UnpackIndices(packed uint8) (uint8, uint8) {
	return packed & 0x0F, (packed >> 4) & 0x0F
}
