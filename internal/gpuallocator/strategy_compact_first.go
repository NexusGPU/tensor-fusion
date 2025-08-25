package gpuallocator

import (
	"fmt"
	"sort"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

// CompactFirst selects GPU with minimum available resources (most utilized)
// to efficiently pack workloads and maximize GPU utilization
type CompactFirst struct {
	cfg *config.GPUFitConfig
}

var _ Strategy = CompactFirst{}

// SelectGPUs selects multiple GPUs from the same node with the least available resources (most packed)
func (c CompactFirst) SelectGPUs(gpus []*tfv1.GPU, count uint) ([]*tfv1.GPU, error) {
	if len(gpus) == 0 {
		return nil, fmt.Errorf("no GPUs available")
	}

	// If count is 1, just find the most packed GPU
	if count <= 1 {
		// Start with the first GPU as the default selected
		selected := gpus[0]
		lowestTflops, _ := selected.Status.Available.Tflops.AsInt64()
		lowestVRAM, _ := selected.Status.Available.Vram.AsInt64()

		// Find the GPU with the lowest available resources (most packed)
		for i := 1; i < len(gpus); i++ {
			gpu := gpus[i]

			currentTflops, _ := gpu.Status.Available.Tflops.AsInt64()
			currentVRAM, _ := gpu.Status.Available.Vram.AsInt64()

			// We prioritize minimizing VRAM, but if VRAM is equal, we choose based on TFlops
			if currentVRAM < lowestVRAM || (currentVRAM == lowestVRAM && currentTflops < lowestTflops) {
				selected = gpu
				lowestTflops = currentTflops
				lowestVRAM = currentVRAM
			}
		}

		return []*tfv1.GPU{selected}, nil
	}

	// For count > 1, we need to find GPUs from the same node
	// Group GPUs by node
	gpusByNode := make(map[string][]*tfv1.GPU)
	for _, gpu := range gpus {
		if gpu.Labels == nil {
			continue
		}

		nodeName, exists := gpu.Labels[constants.LabelKeyOwner]
		if !exists {
			continue
		}

		gpusByNode[nodeName] = append(gpusByNode[nodeName], gpu)
	}

	// Find nodes that have at least 'count' GPUs
	var candidateNodes []string
	for nodeName, nodeGPUs := range gpusByNode {
		if uint(len(nodeGPUs)) >= count {
			candidateNodes = append(candidateNodes, nodeName)
		}
	}

	if len(candidateNodes) == 0 {
		return nil, fmt.Errorf("no node has at least %d available GPUs", count)
	}

	// For each candidate node, calculate the average resource availability
	nodeScores := make(map[string]int64)
	for _, nodeName := range candidateNodes {
		nodeGPUs := gpusByNode[nodeName]

		// Sort GPUs by resource availability (most packed first)
		sort.Slice(nodeGPUs, func(i, j int) bool {
			// Compare VRAM first
			vramI, _ := nodeGPUs[i].Status.Available.Vram.AsInt64()
			vramJ, _ := nodeGPUs[j].Status.Available.Vram.AsInt64()
			if vramI != vramJ {
				return vramI < vramJ // Lower VRAM (more packed) comes first
			}

			// If VRAM is equal, compare TFlops
			tflopsI, _ := nodeGPUs[i].Status.Available.Tflops.AsInt64()
			tflopsJ, _ := nodeGPUs[j].Status.Available.Tflops.AsInt64()
			return tflopsI < tflopsJ // Lower TFlops (more packed) comes first
		})

		// Calculate score based on the first 'count' GPUs (most packed ones)
		var totalVRAM int64
		var totalTFlops int64
		for i := 0; i < int(count); i++ {
			vram, _ := nodeGPUs[i].Status.Available.Vram.AsInt64()
			tflops, _ := nodeGPUs[i].Status.Available.Tflops.AsInt64()
			totalVRAM += vram
			totalTFlops += tflops
		}

		// Lower score is better (more packed)
		nodeScores[nodeName] = totalVRAM*1000 + totalTFlops // Weight VRAM more heavily
	}

	// Find the node with the lowest score (most packed)
	var bestNodeName string
	var bestScore int64 = -1
	for nodeName, score := range nodeScores {
		if bestScore == -1 || score < bestScore {
			bestNodeName = nodeName
			bestScore = score
		}
	}

	// Return the requested number of GPUs from the best node
	result := make([]*tfv1.GPU, count)
	for i := 0; i < int(count); i++ {
		result[i] = gpusByNode[bestNodeName][i]
	}
	return result, nil
}

// Score function is using by Kubernetes scheduler framework
func (c CompactFirst) Score(gpu *tfv1.GPU) int {
	tflopsUsedPercentage := 100 - gpu.Status.Available.Tflops.AsApproximateFloat64()/gpu.Status.Capacity.Tflops.AsApproximateFloat64()*100
	vramUsedPercentage := 100 - gpu.Status.Available.Vram.AsApproximateFloat64()/gpu.Status.Capacity.Vram.AsApproximateFloat64()*100
	return normalizeScore(c.cfg, vramUsedPercentage, tflopsUsedPercentage)
}
