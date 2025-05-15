package filter

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/samber/lo"
)

// ResourceFilter filters GPUs based on available resources
type ResourceFilter struct {
	requiredResource tfv1.Resource
}

// NewResourceFilter creates a new ResourceFilter with the specified resource requirements
func NewResourceFilter(required tfv1.Resource) *ResourceFilter {
	return &ResourceFilter{
		requiredResource: required,
	}
}

// Filter implements GPUFilter.Filter
func (f *ResourceFilter) Filter(_ context.Context, gpus []tfv1.GPU) ([]tfv1.GPU, error) {
	return lo.Filter(gpus, func(gpu tfv1.GPU, _ int) bool {
		// Check if GPU has enough resources available
		if gpu.Status.Available == nil {
			return false
		}

		// Check TFlops and VRAM availability
		hasTflops := gpu.Status.Available.Tflops.Cmp(f.requiredResource.Tflops) >= 0
		hasVram := gpu.Status.Available.Vram.Cmp(f.requiredResource.Vram) >= 0

		return hasTflops && hasVram
	}), nil
}
