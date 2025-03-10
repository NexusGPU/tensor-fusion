package scheduler

import (
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// CompactFirst selects GPU with minimum available resources (most utilized)
// to efficiently pack workloads and maximize GPU utilization
type CompactFirst struct{}

// SelectGPU implements Strategy interface for CompactFirst strategy
// It selects the GPU with the least available resources (most packed)
func (c CompactFirst) SelectGPU(gpus []tfv1.GPU) (*tfv1.GPU, error) {
	if len(gpus) == 0 {
		return nil, fmt.Errorf("no GPUs available")
	}

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

	return selected, nil
}
