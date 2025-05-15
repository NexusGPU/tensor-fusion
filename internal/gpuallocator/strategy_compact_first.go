package gpuallocator

import (
	"fmt"
	"sort"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

// CompactFirst selects GPU with minimum available resources (most utilized)
// to efficiently pack workloads and maximize GPU utilization
type CompactFirst struct{}

// SelectGPUs selects multiple GPUs from the same node with the least available resources (most packed)
func (c CompactFirst) SelectGPUs(gpus []tfv1.GPU, count uint) ([]*tfv1.GPU, error) {
	if len(gpus) == 0 {
		return nil, fmt.Errorf("no GPUs available")
	}

	// If count is 1, just find the most packed GPU
	if count <= 1 {
		// Start with the first GPU as the default selected
		selected := &gpus[0]
		lowestTflops := selected.Status.Available.Tflops.Value()
		lowestVRAM := selected.Status.Available.Vram.Value()

		// Find the GPU with the lowest available resources (most packed)
		for i := 1; i < len(gpus); i++ {
			gpu := &gpus[i]

			currentTflops := gpu.Status.Available.Tflops.Value()
			currentVRAM := gpu.Status.Available.Vram.Value()

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
	gpusByNode := make(map[string][]tfv1.GPU)
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
			vramI := nodeGPUs[i].Status.Available.Vram.Value()
			vramJ := nodeGPUs[j].Status.Available.Vram.Value()
			if vramI != vramJ {
				return vramI < vramJ // Lower VRAM (more packed) comes first
			}

			// If VRAM is equal, compare TFlops
			tflopsI := nodeGPUs[i].Status.Available.Tflops.Value()
			tflopsJ := nodeGPUs[j].Status.Available.Tflops.Value()
			return tflopsI < tflopsJ // Lower TFlops (more packed) comes first
		})

		// Calculate score based on the first 'count' GPUs (most packed ones)
		var totalVRAM int64
		var totalTFlops int64
		for i := 0; i < int(count); i++ {
			totalVRAM += nodeGPUs[i].Status.Available.Vram.Value()
			totalTFlops += nodeGPUs[i].Status.Available.Tflops.Value()
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
		result[i] = &gpusByNode[bestNodeName][i]
	}
	return result, nil
}
