package filter

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// GPUModelFilter filters GPUs based on their model (e.g., A100, H100)
type GPUModelFilter struct {
	requiredModel string
}

// NewGPUModelFilter creates a new filter that matches GPUs with the specified model
func NewGPUModelFilter(model string) *GPUModelFilter {
	return &GPUModelFilter{
		requiredModel: model,
	}
}

// Filter implements GPUFilter interface
func (f *GPUModelFilter) Filter(ctx context.Context, gpus []tfv1.GPU) ([]tfv1.GPU, error) {
	if f.requiredModel == "" {
		return gpus, nil
	}

	var filtered []tfv1.GPU
	for _, gpu := range gpus {
		if gpu.Status.GPUModel == f.requiredModel {
			filtered = append(filtered, gpu)
		}
	}
	return filtered, nil
}
