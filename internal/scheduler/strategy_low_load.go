package scheduler

import (
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
)

// LowLoadFirst selects GPU with maximum available resources (least utilized)
// to distribute workloads more evenly across GPUs
type LowLoadFirst struct{}

// SelectGPU implements Strategy interface for LowLoadFirst strategy
// It selects the GPU with the most available resources (least loaded)
func (l LowLoadFirst) SelectGPU(gpus []tfv1.GPU) (*tfv1.GPU, error) {
	if len(gpus) == 0 {
		return nil, fmt.Errorf("no GPUs available")
	}

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

	return selected, nil
}
