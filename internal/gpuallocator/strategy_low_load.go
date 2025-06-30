package gpuallocator

import (
	"fmt"
	"sort"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/samber/lo"
)

// LowLoadFirst selects GPU with maximum available resources (least utilized)
// to distribute workloads more evenly across GPUs
type LowLoadFirst struct {
	cfg *config.GPUFitConfig
}

var _ Strategy = LowLoadFirst{}

// SelectGPUs selects multiple GPUs from the same node with the most available resources (least loaded)
func (l LowLoadFirst) SelectGPUs(gpus []tfv1.GPU, count uint) ([]*tfv1.GPU, error) {
	if len(gpus) == 0 {
		return nil, fmt.Errorf("no GPUs available")
	}

	// If count is 1, just find the least loaded GPU
	if count <= 1 {
		// Start with the first GPU as the default selected
		selected := &gpus[0]
		highestTflops := selected.Status.Available.Tflops.Value()
		highestVRAM := selected.Status.Available.Vram.Value()

		// Find the GPU with the highest available resources (least loaded)
		for i := 1; i < len(gpus); i++ {
			gpu := &gpus[i]

			currentTflops := gpu.Status.Available.Tflops.Value()
			currentVRAM := gpu.Status.Available.Vram.Value()

			// We prioritize maximizing VRAM, but if VRAM is equal, we choose based on TFlops
			if currentVRAM > highestVRAM || (currentVRAM == highestVRAM && currentTflops > highestTflops) {
				selected = gpu
				highestTflops = currentTflops
				highestVRAM = currentVRAM
			}
		}

		return []*tfv1.GPU{selected}, nil
	}

	// For count > 1, we need to find GPUs from the same node
	// Group GPUs by node
	validGPUs := lo.Filter(gpus, func(gpu tfv1.GPU, _ int) bool {
		return gpu.Labels != nil && gpu.Labels[constants.LabelKeyOwner] != ""
	})
	gpusByNode := lo.GroupBy(validGPUs, func(gpu tfv1.GPU) string {
		return gpu.Labels[constants.LabelKeyOwner]
	})

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

		// Sort GPUs by resource availability (least loaded first)
		sort.Slice(nodeGPUs, func(i, j int) bool {
			// Compare VRAM first
			vramI := nodeGPUs[i].Status.Available.Vram.Value()
			vramJ := nodeGPUs[j].Status.Available.Vram.Value()
			if vramI != vramJ {
				return vramI > vramJ // Higher VRAM (less loaded) comes first
			}

			// If VRAM is equal, compare TFlops
			tflopsI := nodeGPUs[i].Status.Available.Tflops.Value()
			tflopsJ := nodeGPUs[j].Status.Available.Tflops.Value()
			return tflopsI > tflopsJ // Higher TFlops (less loaded) comes first
		})

		// Calculate score based on the first 'count' GPUs (least loaded ones)
		var totalVRAM int64
		var totalTFlops int64
		for i := 0; i < int(count); i++ {
			totalVRAM += nodeGPUs[i].Status.Available.Vram.Value()
			totalTFlops += nodeGPUs[i].Status.Available.Tflops.Value()
		}

		// Higher score is better (less loaded)
		nodeScores[nodeName] = totalVRAM*1000 + totalTFlops // Weight VRAM more heavily
	}

	// Find the node with the highest score (least loaded)
	var bestNodeName string
	var bestScore int64 = -1
	for nodeName, score := range nodeScores {
		if bestScore == -1 || score > bestScore {
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

// Score function is using by Kubernetes scheduler framework
func (l LowLoadFirst) Score(gpu tfv1.GPU) int {
	tflopsAvailablePercentage := gpu.Status.Available.Tflops.AsApproximateFloat64() / gpu.Status.Capacity.Tflops.AsApproximateFloat64() * 100
	vramAvailablePercentage := gpu.Status.Available.Vram.AsApproximateFloat64() / gpu.Status.Capacity.Vram.AsApproximateFloat64() * 100
	return normalizeScore(l.cfg, vramAvailablePercentage, tflopsAvailablePercentage)
}
